"""Real-time RoboBeats: one forward per new frame, reusing the model cache inside a chunk.

Same protocol and prompt as `Gate.stream` (chunks of 100 frames with 10 overlap, p read at the marker after each judged
frame). Within a chunk the prompt up to the newest marker only grows, so a step feeds just the new frame's tokens and
image on top of the cached prefix. At a chunk switch, or when the conditions change, the chunk is encoded from scratch.
If the cached tokens are ever not a prefix of the current prompt, the chunk is re-encoded too, so the output never
depends on the cache being usable (it only gets slower).

Conditions. A session judges either the default unconditional question ("Has a new event occurred?", the released
gate) or free-text conditions of your own, all in one forward per frame. With several conditions the prompt is the
multi-condition protocol the released weights (v8) were trained on: the numbered conditions first, then after every
judged frame the markers "Is condition k true now?" in order. One condition uses the single-condition prompt.
The unconditional question was always trained on its own, so keep it in its own session (as `serve.py` does).

    gate = LiveGate('superbigsaw/RoboBeats-2B')
    events = gate.session()                                         # p(new event)
    watch = gate.session(['A person is drinking.', 'The door is open.'])
    for t, image in camera:                                         # integer seconds, PIL images
        f = gate.frame(t, image)                                    # preprocess once, share between sessions
        p_event = events.push(f)[0]
        p_drink, p_door = watch.push(f)

`python -m robobeats.live --check video.mp4` compares with `Gate.stream` (unconditional) and with a from-scratch forward
at every frame (conditions), and reports the per-frame latency.
"""
import time

from .gate import CHUNK, CONDITION, MAX_PIXELS, MIN_PIXELS, OVERLAP, STRIDE, Gate

PAD = '<|image_pad|>'
class Frame:
    """one preprocessed frame (pixel values, grid, number of image tokens); can be pushed into several sessions.
    tick = frame index in the stream (= the second at 1 fps); pts = its time in seconds (written into the prompt)"""
    __slots__ = ('tick', 'pts', 'pix', 'grid', 'ntok')


class Session:
    """one live stream judged against a fixed list of conditions"""

    def __init__(self, gate, conditions=None, recompute=False):
        self.g, self.recompute = gate, recompute
        self.buf, self.k, self.last_tick = [], 0, None
        self.ids, self.cache, self.n_img, self.cache_k = None, None, 0, None
        self.set_conditions(conditions)

    def set_conditions(self, conditions):
        """change the conditions; the current chunk is re-encoded at the next frame (the frames are kept)"""
        conds = [c.strip() for c in (conditions or [CONDITION]) if c and c.strip()] or [CONDITION]
        self.conditions = conds
        self.marks = self.g._marks(len(conds))
        self.ids = None

    def push(self, frame, image=None):
        """add a frame (a `Frame` from gate.frame, or a tick plus a PIL image) and return [p_k] for every condition at it"""
        if image is not None:
            frame = self.g.frame(frame, image)
        if self.last_tick is not None and frame.tick <= self.last_tick:
            raise ValueError(f'ticks must increase: {frame.tick} after {self.last_tick}')
        self.last_tick = frame.tick
        if len(self.buf) == CHUNK:
            self.buf = self.buf[STRIDE:]; self.k += 1
        self.buf.append(frame)
        return self.g._step(self)

    def close(self):
        self.cache = None; self.buf = []


class LiveGate(Gate):
    def __init__(self, model_id, device='cuda:0', merge_lora=True):
        super().__init__(model_id, device)
        self.model_id = str(model_id)
        torch = self.torch
        if merge_lora and hasattr(self.model, 'merge_and_unload'):
            self.model = self.model.merge_and_unload().eval()   # same weights, no adapter overhead per layer
        # cuDNN attention builds a new plan for every sequence length, and a live prompt changes length every frame
        torch.backends.cuda.enable_cudnn_sdp(False)
        self._find_rope()
        self.pad_id = self.tok.convert_tokens_to_ids(PAD)
        self.merge = self.proc.image_processor.merge_size

    def _find_rope(self):
        self.rope = next(m for m in self.model.modules() if hasattr(m, 'get_rope_index') and type(m).__name__.endswith('Model'))

    def session(self, conditions=None, recompute=False):
        return Session(self, conditions, recompute)

    def frame(self, tick, image, pts=None):
        """preprocess one frame. tick: frame index (strictly increasing within a session); pts: the time in seconds
        written into the prompt, default = tick (the 1 fps protocol of `Gate.stream`)"""
        f = Frame(); f.tick = tick; f.pts = float(tick) if pts is None else float(pts)
        out = self.proc.image_processor(images=[image], return_tensors='pt', max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS)
        f.pix, f.grid = out['pixel_values'], out['image_grid_thw']
        f.ntok = int(f.grid.prod()) // self.merge ** 2
        return f

    def _ids(self, s):
        """token ids of the chunk prompt cut right after the newest frame's last marker, and the K marker positions there"""
        frames = s.buf
        judged = {f.tick for f in (frames if s.k == 0 else frames[OVERLAP:])}
        text = self._text([(f.tick, f.pts) for f in frames], judged, s.conditions)
        parts = text.split(PAD)
        assert len(parts) == len(frames) + 1
        text = parts[0] + ''.join(PAD * f.ntok + p for f, p in zip(frames, parts[1:]))
        ids = self.tok(text, add_special_tokens=False)['input_ids']

        # markers of the newest judged frame, walking back from its last one: marker k must end before marker k+1 starts
        pos, end = [], len(ids)
        for mark in reversed(s.marks):
            m = len(mark)
            start = max(i for i in range(end - m + 1) if ids[i:i + m] == mark)
            pos.append(start + m - 1); end = start
        pos.reverse()
        return ids[:pos[-1] + 1], pos

    def _step(self, s):
        torch = self.torch
        ids, pos = self._ids(s)
        if s.recompute or s.cache_k != s.k or s.ids is None or len(ids) <= len(s.ids) or ids[:len(s.ids)] != s.ids:
            s.cache_k, s.ids, s.cache, s.n_img = s.k, [], None, 0          # new chunk / changed prompt: from scratch
        frames = s.buf
        new, start = frames[s.n_img:], len(s.ids)
        full = torch.tensor([ids], device=self.device)
        mm = (full == self.pad_id).int()
        keep = len(ids) - pos[0]
        with torch.inference_mode():
            rp, _ = self.rope.get_rope_index(full, mm, image_grid_thw=torch.cat([f.grid for f in frames]).to(self.device))
            out = self.model(input_ids=full[:, start:], mm_token_type_ids=mm[:, start:], position_ids=rp[..., start:],
                             pixel_values=torch.cat([f.pix for f in new]).to(self.device) if new else None,
                             image_grid_thw=torch.cat([f.grid for f in new]).to(self.device) if new else None,
                             past_key_values=s.cache, use_cache=True, logits_to_keep=keep)
            z = out.logits[0, [p - pos[0] - keep for p in pos]].float()[:, self.label_ids]
        s.cache, s.ids, s.n_img = out.past_key_values, ids, len(frames)
        return z.softmax(-1)[:, 1].tolist()


def _check(video, model, device, seconds, conditions):
    """unconditional: live vs Gate.stream (the released offline protocol); conditions: cached vs from-scratch at every frame"""
    import json
    from .video import iter_ticks
    frames = [(t, im.convert('RGB')) for t, _, im in iter_ticks(video, 0, seconds)]
    g = LiveGate(model, device, merge_lora=False)
    ref = dict(g.stream(frames))                       # with the adapter unmerged, exactly as released
    g.model = g.model.merge_and_unload().eval(); g._find_rope()
    ev, cc, cr = g.session(), g.session(conditions), g.session(conditions, recompute=True)
    live, a, b, lat_e, lat_c = {}, [], [], [], []
    sync = g.torch.cuda.synchronize
    for t, im in frames:
        f = g.frame(t, im)
        sync(); t0 = time.perf_counter(); live[t] = ev.push(f)[0]; sync(); lat_e.append(1000 * (time.perf_counter() - t0))
        t0 = time.perf_counter(); a.append(cc.push(f)); sync(); lat_c.append(1000 * (time.perf_counter() - t0))
        b.append(cr.push(f))
    med = lambda x: round(sorted(x[5:] or x)[len(x[5:] or x) // 2], 1)
    d = [abs(live[t] - ref[t]) for t in ref]
    dc = [abs(x - y) for pa, pb in zip(a, b) for x, y in zip(pa, pb)]
    print(json.dumps(dict(
        frames=len(d),
        event=dict(vs_stream_max_abs_dp=round(max(d), 4), mean_abs_dp=round(sum(d) / len(d), 5),
                   flips_at_0_5=sum((live[t] > .5) != (ref[t] > .5) for t in ref), step_ms_median=med(lat_e)),
        conditions=dict(K=len(conditions), cached_vs_scratch_max_abs_dp=round(max(dc), 4),
                        flips_at_0_5=sum((x > .5) != (y > .5) for pa, pb in zip(a, b) for x, y in zip(pa, pb)),
                        frames_true=[sum(p[k] > .5 for p in a) for k in range(len(conditions))], step_ms_median=med(lat_c)),
        note='step time = model call + tokenizing, image preprocessing excluded (done once per frame, shared)')))


if __name__ == '__main__':
    import argparse
    a = argparse.ArgumentParser(description='check the live gate on one video')
    a.add_argument('--check', required=True, metavar='VIDEO')
    a.add_argument('--gate-model', default='superbigsaw/RoboBeats-2B')
    a.add_argument('--device', default='cuda:0')
    a.add_argument('--seconds', type=float, default=300)
    a.add_argument('--condition', action='append', help='a condition to test (repeat; default: two examples)')
    a = a.parse_args()
    _check(a.check, a.gate_model, a.device, a.seconds, a.condition or ['A person is holding a cup.', 'A person is standing in the kitchen.'])
