"""Score a Charades-STA run (run.py) with the paper's online readout.

online: a query is on from p > 0.5 while p >= 0.3; on-runs closer than 1 s are bridged; the first run lasting at least 1 s
is the answer, so the answer uses no frame after the moment it is given. Tick t covers [t, t + 1) s at 1 fps.
Reported: R@0.3 / R@0.5 / R@0.7 (IoU with the labelled span), mIoU, and the share of queries with no answer.
--offline adds the offline readout (after the clip: the run of p > 0.5 with the largest probability mass).

  python benchmarks/charades_sta/score.py --run runs/charades_2B --annotations charades_sta_test.txt
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import queries   # noqa: E402

GAP_S, MIN_LEN_S = 1.0, 1.0


def merge(ticks, fps):
    runs = []
    for t in ticks:
        if runs and t - runs[-1][1] <= round(GAP_S * fps) + 1:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    return runs


def online(p, fps):
    on, active = [], False
    for t, x in enumerate(p):
        active = x > 0.5 or (active and x >= 0.3)
        if active:
            on.append(t)
    for s, e in merge(on, fps):
        if e - s + 1 >= round(MIN_LEN_S * fps):
            return [s / fps, (e + 1) / fps]
    return None


def offline(p, fps):
    runs = merge([t for t, x in enumerate(p) if x > 0.5], fps)
    if runs:
        s, e = max(runs, key=lambda r: sum(p[r[0]:r[1] + 1]))
        return [s / fps, (e + 1) / fps]
    if not p:
        return None
    k = max(range(len(p)), key=lambda t: p[t]); s = e = k
    while s > 0 and p[s - 1] >= 0.5 * p[k]:
        s -= 1
    while e + 1 < len(p) and p[e + 1] >= 0.5 * p[k]:
        e += 1
    return [s / fps, (e + 1) / fps]


def iou(a, b):
    if a is None:
        return 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--run', required=True); a.add_argument('--annotations', required=True)
    a.add_argument('--offline', action='store_true')
    a = a.parse_args()
    Q = {q['qid']: q for q in queries(a.annotations)}
    ser = {}
    for f in glob.glob(os.path.join(a.run, 'series_shard*.json')):
        ser.update(json.load(open(f)))
    fps = json.load(open(os.path.join(a.run, 'meta.json')))['fps']
    missing = [i for i in Q if i not in ser]
    if missing:
        print(f'warning: {len(missing)} of {len(Q)} queries have no series yet; scoring the {len(Q) - len(missing)} done')
    done = [i for i in Q if i in ser]
    print('| readout | queries | R@0.3 | R@0.5 | R@0.7 | mIoU | no answer |\n|---|---|---|---|---|---|---|')
    for name, fn in [('online', online)] + ([('offline', offline)] if a.offline else []):
        pred = {i: fn(ser[i], fps) for i in done}; ious = [iou(pred[i], Q[i]['gt']) for i in done]
        r = [100 * sum(x >= th for x in ious) / len(ious) for th in (0.3, 0.5, 0.7)]
        print(f'| {name} | {len(done)} | {r[0]:.1f} | {r[1]:.1f} | {r[2]:.1f} | {100 * sum(ious) / len(ious):.1f} | '
              f'{100 * sum(pred[i] is None for i in done) / len(done):.1f}% |')


if __name__ == '__main__':
    main()
