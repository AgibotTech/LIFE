"""RoboBeats as a model of the official SPOT-Bench harness (github.com/dibschat/SPOT-Bench), task ABD (paper Tables 2 and 14).

Copy this file to SPOT-Bench/model_wrappers/robobeats.py and register it in model_wrappers/__init__.py:
    "RoboBeats": ("model_wrappers.robobeats", "RoboBeats"),
then run the harness with LIFE_ROOT pointing at this repository (see benchmarks/README.md).

Per question: the request is rewritten into a start condition and a finish condition (abd_rewrites.json, written once by the
plain Qwen3.5-2B so every model is judged on the same conditions). Both are judged together, at 2 fps, on a stream that starts
at the moment the question is asked (frames before it are never shown; times relative to it). Readout: hysteresis 0.5 / 0.3,
1 s grace after the question; "start" fires at the first rising edge of the start condition, "finish" at the first falling
edge of the finish condition after it has been on; each word once, emitted at the second of the decision. The stream of a
question stops once all its words have fired. Frames: every round(frame rate / 2)-th decoded frame.
Env: ROBOBEATS_MODEL (default superbigsaw/RoboBeats-2B), ROBOBEATS_FPS (default 2), LIFE_ROOT (this repository).
"""
import json
import os
import re
import sys

from .base_model import ModelStreaming

LIFE = os.environ.get('LIFE_ROOT') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (LIFE, os.path.join(LIFE, 'benchmarks')):
    if p not in sys.path:
        sys.path.insert(0, p)

REWRITES = os.path.join(LIFE, 'benchmarks', 'spot_abd', 'abd_rewrites.json')
RESPOND_RE = re.compile(r"\bRespond\b.*?\bwords?\s*:\s*(?P<words>.+?)\s*$", re.IGNORECASE)
START_RE = re.compile(r"^(?:start|begin|onset|commence)\w*$")


def clean(q):
    return re.sub(r"([.!?])\s*Respond\b.*$", r"\1", " ".join(str(q or "").split()), flags=re.IGNORECASE).strip()


def words_of(question):
    """the words a question asks for, each tagged 'start' or 'end'"""
    m = RESPOND_RE.search(question or ""); out = []
    for chunk in re.split(r"\s*(?:,|\band\b|\bor\b|/)\s*", m.group("words") if m else "start, finish"):
        w = chunk.strip().strip(".,;:").lower()
        if w and all(w != seen for _, seen in out):
            out.append(("start" if START_RE.match(w) else "end", w))
    return out or [("start", "start"), ("end", "finish")]


class RoboBeats(ModelStreaming):
    FPS, TAU, TAU_EXIT, GRACE = int(os.environ.get('ROBOBEATS_FPS', '2')), 0.5, 0.3, 1

    def __init__(self, args):
        super().__init__(stream_fps=self.FPS)
        from robobeats.gate import CHUNK, OVERLAP, STRIDE, Gate
        self.CHUNK, self.OVERLAP, self.STRIDE = CHUNK, OVERLAP, STRIDE
        self.gate = Gate(os.environ.get('ROBOBEATS_MODEL', 'superbigsaw/RoboBeats-2B'))
        self.rewrites = json.load(open(REWRITES))

    def _chunk(self, conds, buf, k, t0):
        ids = [k * self.STRIDE + i for i in range(len(buf))]
        judged = ids if k == 0 else ids[self.OVERLAP:]
        ticks = [(j, round(t - t0, 6)) for j, (t, _) in zip(ids, buf)]
        return judged, self.gate._judge(ticks, [im for _, im in buf], set(judged), conds)

    def inference(self, video_path, turns):
        from common import frames_every_step
        fps = self.FPS; frames = frames_every_step(video_path, float(fps)); n = len(frames); events = []
        for i, t in enumerate(turns):
            tid = f"q{i}"; ask = int(min(max(0, float(t.get('ask_time', 0)) * fps), n - 1)); q = t.get('question', '')
            events.append({"time": ask / fps, "type": "question", "value": q, "turn_id": tid})
            rw = self.rewrites.get(q, {}); labels = words_of(q)
            texts = list(dict.fromkeys(rw.get(role) or clean(q) for role, _ in labels))   # start and finish conditions (one if equal)
            st = {role: dict(active=False, was_on=False, fired=False) for role, _ in labels}
            stream = [(j / fps, frames[j]) for j in range(ask, n)]; s, k = 0, 0
            while stream and not all(v['fired'] for v in st.values()):
                buf = stream[s:s + self.CHUNK]
                judged, P = self._chunk(texts, buf, k, stream[0][0])
                for r, j in enumerate(judged):
                    tick = ask + j
                    for role, word in labels:
                        v = st[role]; p = P[texts.index(rw.get(role) or clean(q))][r]
                        rising = p > self.TAU and not v['active']; falling = v['active'] and p < self.TAU_EXIT
                        if tick - ask >= self.GRACE * fps and not v['fired']:
                            if role == 'start' and rising or role == 'end' and falling and v['was_on']:
                                events.append({"time": tick / fps, "type": "response", "value": word, "turn_id": tid, "p": round(p, 3)})
                                v['fired'] = True
                        v['active'] = True if p > self.TAU else (False if p < self.TAU_EXIT else v['active'])
                        v['was_on'] = v['was_on'] or v['active']
                if s + self.CHUNK >= len(stream):
                    break
                s += self.STRIDE; k += 1
                if len(stream) - s <= self.OVERLAP:
                    break
        events.sort(key=lambda e: e['time'])
        return events
