"""RoboBeats streaming event gate: p("a new event occurred") at every 1 Hz tick, strictly causal.

Protocol (identical to training and to the reported SSS-Bench runs):
  * condition first, then timestamped frames; after every judged frame a fixed marker; p(true) is read at the
    marker's last token, restricted to the 'false' / 'true' tokens. No answer token is written into the sequence,
    so the decision at tick t only depends on frames <= t.
  * frames in chunks of 100 with 10 overlap (stride 90); chunk k >= 1 restates its first 10 frames as context only.
  * a stream starts at the question's observation start (frame ids = absolute integer seconds).
"""
import json
from pathlib import Path

CONDITION = 'Has a new event occurred?'
SYSTEM = ('You watch a live, timestamped video stream and judge one condition. After each new frame you are asked whether the '
          'condition is true at that frame. Use only the frames shown so far; all images are past or current. Answer whether it '
          'is true now, not whether it was true earlier. Do not invent unseen actions. Answer only true or false.')
MARK = 'Is the condition true now?'
CHUNK, OVERLAP = 100, 10
STRIDE = CHUNK - OVERLAP
MAX_PIXELS, MIN_PIXELS = 147456, 3136


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

    def _text(self, ticks, judged):
        parts = [dict(type='text', text=f'Condition: {CONDITION}\n')]
        for t in ticks:
            parts += [dict(type='text', text=f'Video sample {t}; actual relative time {float(t):.6f} seconds.'), dict(type='image')]
            if t in judged:
                parts.append(dict(type='text', text=MARK + '\n'))
        msgs = [dict(role='system', content=SYSTEM), dict(role='user', content=parts)]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)

    def _chunk(self, ticks, images, judged):
        text = self._text(ticks, judged)
        enc = self.proc(text=[text], images=images, return_tensors='pt', images_kwargs={'max_pixels': MAX_PIXELS, 'min_pixels': MIN_PIXELS})
        toks, m = enc['input_ids'][0].tolist(), len(self.mark)
        pos = [i + m - 1 for i in range(len(toks) - m + 1) if toks[i:i + m] == self.mark]
        assert len(pos) == len(judged), (len(pos), len(judged))
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.inference_mode():
            z = self.model(**enc, use_cache=False, logits_to_keep=self.torch.tensor(pos, device=self.device)).logits[0].float()[:, self.label_ids]
        return z.softmax(-1)[:, 1].tolist()

    def stream(self, frames):
        """frames: iterable of (tick, PIL image) in order -> [(tick, p)]; chunk k covers frames [90k, 90k+100)"""
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


def select(probs, start, on=0.5, off=0.3, pre_roll=2):
    """hysteresis readout: an event opens at p > on and closes at the first tick with p < off; each event keeps its
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
