"""Run the RoboBeats gate on one video and print the detected events.

  python -m robobeats video.mp4 --gate-model superbigsaw/RoboBeats-2B [--start 0 --end 600] [--out probs.json]

The video is sampled causally at 1 fps on its own clock; every tick gets p("a new event occurred"), and the
hysteresis readout used on SSS-Bench (open at p > 0.5, close at p < 0.3, 2 s pre-roll) groups the kept ticks into events.
"""
import argparse, json, pathlib

import av

from .gate import Gate, select
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
    ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--end', type=float, help='last second to judge (default: end of video)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', help='write [[tick, p], ...] and the event spans to this JSON file')
    a = ap.parse_args()
    if a.end is None:
        with av.open(a.video) as c:
            a.end = c.duration / 1e6
    gate = Gate(a.gate_model, a.device)
    probs = gate.stream((t, im.convert('RGB')) for t, _, im in iter_ticks(a.video, a.start, a.end))
    events = spans(select(probs, a.start))
    for s, e in events:
        print(f'event  {s:7d} s – {e:7d} s')
    print(f'{len(events)} events, {sum(e - s + 1 for s, e in events)}/{len(probs)} ticks kept')
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(dict(video=a.video, gate_model=a.gate_model, ticks=probs, events=events)))


if __name__ == '__main__':
    main()
