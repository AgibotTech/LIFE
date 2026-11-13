"""RoboBeats streaming gate: at every tick, p(a condition holds now), strictly causal.

Protocol (identical to training and to the reported runs):
  * condition(s) first, then timestamped frames; after every judged frame a fixed marker per condition; p(true) is read at
    the marker's last token, restricted to the 'false' / 'true' tokens. No answer token is written into the sequence, so
    the decision at tick t only depends on frames <= t.
  * frames in chunks of 100 with 10 overlap (stride 90); chunk k >= 1 restates its first 10 frames as context only.
  * one condition uses the single-condition prompt; several use the numbered multi-condition prompt, all judged in one
    forward ("Condition 1: ...", then "Is condition k true now?" after every judged frame).

Two entry points:
  * `Gate.stream(frames)` -- event discovery, p("a new event occurred"), as in SSS-Bench: frame ids = absolute integer
    seconds from the question's observation start.
  * `Gate.track(frames, conditions)` -- condition tracking, as in the paper's condition-tracking evaluations
    (benchmarks/): the stream starts when the conditions are registered; sample numbers count from 0 and times are
    relative to the first frame.
"""
import json
from pathlib import Path

CONDITION = 'Has a new event occurred?'
SYSTEM = ('You watch a live, timestamped video stream and judge one condition. After each new frame you are asked whether the '
          'condition is true at that frame. Use only the frames shown so far; all images are past or current. Answer whether it '
          'is true now, not whether it was true earlier. Do not invent unseen actions. Answer only true or false.')
SYSTEM_MULTI = ('You watch a live, timestamped video stream and judge several numbered conditions. After each new frame you are '
                'asked, for every condition in turn, whether it is true at that frame. Judge each condition on its own. Use only the '
                'frames shown so far; all images are past or current. Answer whether it is true now, not whether it was true earlier. '
                'Do not invent unseen actions. Answer only true or false.')
MARK = 'Is the condition true now?'
CHUNK, OVERLAP = 100, 10
STRIDE = CHUNK - OVERLAP
MAX_PIXELS, MIN_PIXELS = 147456, 3136


def mark_text(k, K):
    return MARK if K == 1 else f'Is condition {k + 1} true now?'


def resolve(model_id):
    p = Path(model_id)
    if p.exists():
        return str(p)
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id)


class Gate:
    def __init__(self, model_id, device='cuda:0'):
        import torch
        from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration
        self.torch, self.device = torch, device
        path = resolve(model_id)
        adapter_cfg = Path(path) / 'adapter_config.json'
        base = resolve(json.loads(adapter_cfg.read_text())['base_model_name_or_path']) if adapter_cfg.exists() else path
        # tokenizer / processor of the base model (the adapter does not change them)
        self.tok = AutoTokenizer.from_pretrained(base)
        self.proc = AutoProcessor.from_pretrained(base)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, device_map={'': device}, attn_implementation='sdpa')
        if adapter_cfg.exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, path)
        model.config.use_cache = False
        self.model = model.eval()
        ids = [self.tok.encode(x, add_special_tokens=False) for x in ('false', 'true')]
        assert all(len(x) == 1 for x in ids)
        self.label_ids = torch.tensor([x[0] for x in ids], device=device)
        nl = self.tok('\n', add_special_tokens=False)['input_ids']
        probe = self.tok('<|vision_end|>' + MARK + '\n', add_special_tokens=False)['input_ids']
        assert probe[0] == self.tok.convert_tokens_to_ids('<|vision_end|>')
        mark = probe[1:]
        self.mark = mark[:-len(nl)] if mark[-len(nl):] == nl and len(mark) > len(nl) else mark
        self._mark_cache = {1: [self.mark]}

    def _marks(self, K):
        """token ids of marker k as it appears in context: marker 1 right after an image block, markers 2..K after a newline"""
        if K in self._mark_cache:
            return self._mark_cache[K]
        tok = self.tok
        nl = tok('\n', add_special_tokens=False)['input_ids']
        strip = lambda ids: ids[:-len(nl)] if ids[-len(nl):] == nl and len(ids) > len(nl) else ids
        probe = tok('<|vision_end|>' + mark_text(0, K) + '\n', add_special_tokens=False)['input_ids']
        assert probe[0] == tok.convert_tokens_to_ids('<|vision_end|>')
        out = [strip(probe[1:])]
        head = tok(mark_text(0, K) + '\n', add_special_tokens=False)['input_ids']
        for k in range(1, K):
            probe = tok(mark_text(0, K) + '\n' + mark_text(k, K) + '\n', add_special_tokens=False)['input_ids']
            assert probe[:len(head)] == head
            out.append(strip(probe[len(head):]))
        self._mark_cache[K] = out
        return out

    def _text(self, ticks, judged, conditions=(CONDITION,)):
        """chat text of one chunk. ticks: frame indices (time = index) or (index, seconds) pairs; judged: indices followed by
        the markers. One condition = the single-condition prompt, several = the numbered prompt."""
        K = len(conditions)
        if K == 1:
            parts = [dict(type='text', text=f'Condition: {conditions[0]}\n')]
        else:
            parts = [dict(type='text', text=''.join(f'Condition {k + 1}: {c}\n' for k, c in enumerate(conditions)))]
        for t in ticks:
            t, sec = t if isinstance(t, tuple) else (t, t)
            parts += [dict(type='text', text=f'Video sample {t}; actual relative time {float(sec):.6f} seconds.'), dict(type='image')]
            if t in judged:
                parts += [dict(type='text', text=mark_text(k, K) + '\n') for k in range(K)]
        msgs = [dict(role='system', content=SYSTEM if K == 1 else SYSTEM_MULTI), dict(role='user', content=parts)]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)

    def _judge(self, ticks, images, judged, conditions=(CONDITION,)):
        """one forward over a chunk -> for every condition, p(true) at each judged frame (in order)"""
        enc = self.proc(text=[self._text(ticks, judged, conditions)], images=images, return_tensors='pt',
                        images_kwargs={'max_pixels': MAX_PIXELS, 'min_pixels': MIN_PIXELS})
        toks = enc['input_ids'][0].tolist()
        pos = []
        for mark in self._marks(len(conditions)):
            m = len(mark)
            pos.append([i + m - 1 for i in range(len(toks) - m + 1) if toks[i:i + m] == mark])
            assert len(pos[-1]) == len(judged), (len(pos[-1]), len(judged))
        keep = sorted(x for p in pos for x in p); at = {x: i for i, x in enumerate(keep)}
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.inference_mode():
            z = self.model(**enc, use_cache=False, logits_to_keep=self.torch.tensor(keep, device=self.device)).logits[0].float()[:, self.label_ids]
        z = z.softmax(-1)[:, 1].tolist()
        return [[z[at[x]] for x in p] for p in pos]

    def _chunk(self, ticks, images, judged):
        return self._judge(ticks, images, judged)[0]

    def stream(self, frames):
        """event discovery. frames: iterable of (tick, PIL image) in order -> [(tick, p)]; chunk k covers frames [90k, 90k+100)"""
        out, buf, k = [], [], 0
        def run():
            judged = buf if k == 0 else buf[OVERLAP:]
            p = self._chunk([t for t, _ in buf], [im for _, im in buf], {t for t, _ in judged})
            out.extend(zip([t for t, _ in judged], p))
        for t, im in frames:
            buf.append((t, im))
            if len(buf) == CHUNK:
                run(); buf = buf[STRIDE:]; k += 1
        if buf and (k == 0 or len(buf) > OVERLAP):
            run()
        return out

    def track(self, frames, conditions):
        """condition tracking. frames: [(time in seconds, PIL image)] in order, from the moment the conditions are registered;
        conditions: list of statements. Returns, for every condition, p(true) at every frame. Sample numbers count from 0 and
        times are relative to the first frame (the protocol of the paper's condition-tracking evaluations)."""
        conditions = list(conditions)
        if not frames:
            return [[] for _ in conditions]
        t0, P, k, s = frames[0][0], [[] for _ in conditions], 0, 0
        while True:
            buf = frames[s:s + CHUNK]
            ids = [k * STRIDE + i for i in range(len(buf))]
            judged = ids if k == 0 else ids[OVERLAP:]
            ticks = [(j, round(t - t0, 6)) for j, (t, _) in zip(ids, buf)]
            for c, p in enumerate(self._judge(ticks, [im for _, im in buf], set(judged), conditions)):
                P[c] += p
            if s + CHUNK >= len(frames):
                break
            s += STRIDE; k += 1
            if len(frames) - s <= OVERLAP:
                break
        return P


def select(probs, start, on=0.5, off=0.3, pre_roll=2):
    """event readout (SSS-Bench): an event opens at p > on and closes at the first tick with p < off; each event keeps its
    frames from (rise - pre_roll) to the closing tick; an event still open at the cutoff keeps frames to the cutoff."""
    active, intervals = None, []
    for t, p in probs:
        if active is None and p > on:
            active = t
        elif active is not None and p < off:
            intervals.append((max(start, active - pre_roll), t)); active = None
    if active is not None:
        intervals.append((max(start, active - pre_roll), probs[-1][0]))
    return [t for t, _ in probs if any(a <= t <= b for a, b in intervals)]


def on_spans(ticks, probs, on=0.5, off=0.3):
    """condition readout (online Charades-STA / SPOT-Bench): a condition turns on at p > on and stays on while p >= off;
    returns the runs of on-ticks as [first, last] tick pairs (the tick where p drops below `off` is not part of a run)."""
    runs, active = [], False
    for t, p in zip(ticks, probs):
        active = p > on or (active and p >= off)
        if active:
            if runs and runs[-1][1] == prev:
                runs[-1][1] = t
            else:
                runs.append([t, t])
        prev = t
    return runs
