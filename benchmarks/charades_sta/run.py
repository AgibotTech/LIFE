"""Charades-STA, causal online temporal localization with RoboBeats (paper Tables 2 and 12).

Every query is a condition tracked on its video from 0 s. Protocol of the paper's runs: 1 fps, sample i = the first frame
shown at or after i s, passed as JPEG q95; the queries of a video are judged together in seeded random groups of at most 8
(the multi-condition prompt, one forward per chunk per group). Writes {query id: [p per second]} for score.py.

  python benchmarks/charades_sta/run.py --videos Charades_v1_480 --annotations charades_sta_test.txt --out runs/charades_2B \
      [--gate-model superbigsaw/RoboBeats-2B] [--shard 0 --nshard 1]

Run one process per GPU with --shard i --nshard n and the same --out; score.py reads every shard.
"""
import argparse, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import frames_at_presentation_time, jpeg, n_frames_at   # noqa: E402

FPS, MAX_K = 1, 8


def queries(path):
    """Charades-STA test annotations, one '<video> <start> <end>##<sentence>' per line; ids by line number"""
    out = []
    for k, line in enumerate(open(path)):
        if not line.strip():
            continue
        head, q = line.strip().split('##', 1); vid, s, e = head.split()
        out.append(dict(qid=f'sta_{k:04d}', vid=vid, query=q.strip(), gt=[float(s), float(e)]))
    return out


def groups(qs, max_k=MAX_K):
    """per video: queries shuffled with a fixed seed, split into near-equal groups of at most max_k"""
    by = {}
    for q in qs:
        by.setdefault(q['vid'], []).append(q)
    out = []
    for vid, g in sorted(by.items()):
        g = list(g); random.Random(f'sta_multi:{vid}').shuffle(g)
        n = -(-len(g) // max_k); z = -(-len(g) // n)
        out += [g[i * z:(i + 1) * z] for i in range(n)]
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--videos', required=True, help='folder with <video id>.mp4 (Charades_v1_480)')
    a.add_argument('--annotations', required=True, help='charades_sta_test.txt')
    a.add_argument('--out', required=True)
    a.add_argument('--gate-model', default='superbigsaw/RoboBeats-2B')
    a.add_argument('--device', default='cuda:0')
    a.add_argument('--shard', type=int, default=0)
    a.add_argument('--nshard', type=int, default=1)
    a = a.parse_args()
    from robobeats.gate import Gate
    qs = queries(a.annotations); vids = sorted({q['vid'] for q in qs}); mine = set(vids[a.shard::a.nshard])
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(fps=FPS, max_k=MAX_K, gate_model=a.gate_model), open(os.path.join(a.out, 'meta.json'), 'w'))
    dest = os.path.join(a.out, f'series_shard{a.shard}.json')
    res = json.load(open(dest)) if os.path.exists(dest) else {}
    todo = [g for g in groups([q for q in qs if q['vid'] in mine]) if any(q['qid'] not in res for q in g)]
    gate = Gate(a.gate_model, a.device); t0 = time.time()
    for j, g in enumerate(todo, 1):
        path = os.path.join(a.videos, g[0]['vid'] + '.mp4')
        ims = frames_at_presentation_time(path, FPS, n_frames_at(path, FPS))
        frames = [(i / FPS, jpeg(im)) for i, im in enumerate(ims)]
        for q, p in zip(g, gate.track(frames, [q['query'] for q in g])):
            res[q['qid']] = [round(x, 4) for x in p]
        if j % 50 == 0 or j == len(todo):
            json.dump(res, open(dest + '.tmp', 'w')); os.replace(dest + '.tmp', dest)
            print(f'shard {a.shard}: {j}/{len(todo)} groups ({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
