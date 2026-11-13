"""Run the RoboBeats gate on one video.

  python -m robobeats video.mp4 --gate-model superbigsaw/RoboBeats-2B [--start 0 --end 600] [--out probs.json]
  python -m robobeats video.mp4 --condition "A person is drinking from a cup." [--condition ...] [--out probs.json]

The video is sampled causally at 1 fps on its own clock.
Without --condition (event discovery): every tick gets p("a new event occurred"), and the readout used on SSS-Bench
(open at p > 0.5, close at p < 0.3, 2 s pre-roll) groups the kept ticks into events.
With --condition (condition tracking): every tick gets p for each condition, all judged in one forward, on a stream that
starts at --start; a condition is on from p > 0.5 while p >= 0.3, as in the paper's online condition-tracking readout.
"""
import argparse, json, pathlib

import av

from .gate import Gate, on_spans, select
from .video import iter_ticks


def spans(ticks):
    out = []
    for t in ticks:
        if out and t == out[-1][1] + 1:
            out[-1][1] = t
        else:
            out.append([t, t])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--gate-model', default='superbigsaw/RoboBeats-2B')
    ap.add_argument('--condition', action='append', help='a condition to track (repeat for several); omit for event discovery')
    ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--end', type=float, help='last second to judge (default: end of video)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', help='write the per-second probabilities and the spans to this JSON file')
    a = ap.parse_args()
    if a.end is None:
        with av.open(a.video) as c:
            a.end = c.duration / 1e6
    gate = Gate(a.gate_model, a.device)
    frames = ((t, im.convert('RGB')) for t, _, im in iter_ticks(a.video, a.start, a.end))
    if a.condition:
        frames = list(frames); ticks = [t for t, _ in frames]
        P = gate.track(frames, a.condition)
        result = []
        for c, p in zip(a.condition, P):
            on = on_spans(ticks, p)
            print(f'{c}\n  ' + ('  '.join(f'{s}–{e} s' for s, e in on) if on else 'never on'))
            result.append(dict(condition=c, p=[[t, round(x, 4)] for t, x in zip(ticks, p)], on=on))
        out = dict(video=a.video, gate_model=a.gate_model, conditions=result)
    else:
        probs = gate.stream(frames)
        events = spans(select(probs, a.start))
        for s, e in events:
            print(f'event  {s:7d} s – {e:7d} s')
        print(f'{len(events)} events, {sum(e - s + 1 for s, e in events)}/{len(probs)} ticks kept')
        out = dict(video=a.video, gate_model=a.gate_model, ticks=probs, events=events)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out))


if __name__ == '__main__':
    main()
