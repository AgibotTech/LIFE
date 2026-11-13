"""StreamingBench Proactive Output with RoboBeats (paper Tables 2 and 13).

Each question reads "When X, output 'Y'". X becomes one condition, tracked at 1 fps on a stream that starts at the
question's time stamp (frames before it are never shown). The answer fires at the first second after the start with
p(X) > 0.5. Frames: every round(frame rate)-th decoded frame. Seconds after ground truth + 4 s are not judged; no reported
window reaches them. Writes one row per question for score.py.

  python benchmarks/streamingbench_po/run.py --bench StreamingBench --out runs/sb_2B [--gate-model superbigsaw/RoboBeats-2B] \
      [--shard 0 --nshard 1]

--bench is the StreamingBench folder with csv/Proactive_Output.csv and proactive_videos/sample_<k>/video.mp4.
"""
import argparse, csv, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import frames_every_step   # noqa: E402

PATTERN = re.compile(r"^\s*When (.+?),\s*output\s*[\"'](.+?)[\"']\s*\.?\s*$", re.S)
TAU = 0.5


def hms(s):
    h, m, sec = s.strip().split(':')
    return int(h) * 3600 + int(m) * 60 + int(sec)


def sample_of(q):
    return int(q['question_id'].split('_sample_')[1].split('_')[0])


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--bench', required=True); a.add_argument('--out', required=True)
    a.add_argument('--gate-model', default='superbigsaw/RoboBeats-2B'); a.add_argument('--device', default='cuda:0')
    a.add_argument('--shard', type=int, default=0); a.add_argument('--nshard', type=int, default=1)
    a = a.parse_args()
    from robobeats.gate import Gate
    qs = sorted(csv.DictReader(open(os.path.join(a.bench, 'csv', 'Proactive_Output.csv'))), key=sample_of)
    qs = [q for q in qs if sample_of(q) % a.nshard == a.shard]
    os.makedirs(a.out, exist_ok=True); dest = os.path.join(a.out, f'rows_shard{a.shard}.json')
    rows = json.load(open(dest)) if os.path.exists(dest) else []; done = {r['question_id'] for r in rows}
    gate = Gate(a.gate_model, a.device); cache, t0 = {}, time.time()
    for q in qs:
        if q['question_id'] in done:
            continue
        path = os.path.join(a.bench, 'proactive_videos', f'sample_{sample_of(q)}', 'video.mp4')
        if path not in cache:
            cache = {path: frames_every_step(path, 1.0)}
        frames = cache[path]
        start, gt = hms(q['time_stamp']), hms(q['ground_truth_time_stamp']); last = min(len(frames) - 1, gt + 4)
        m = PATTERN.match(q['question'])
        cond, output = (m.group(1).strip() + '.', m.group(2)) if m else (q['question'], q['ground_truth_output'])
        ticks = list(range(start, last + 1))
        p = [round(x, 4) for x in gate.track([(t, frames[t]) for t in ticks], [cond])[0]] if ticks else []
        probs = [[t, x] for t, x in zip(ticks, p) if t > start]
        fired = next((t for t, x in probs if x > TAU), None)
        rows.append(dict(question_id=q['question_id'], condition=cond, output=output, parsed=bool(m), start=start,
                         ground_truth=gt, fired=fired, probs=probs))
        json.dump(rows, open(dest + '.tmp', 'w')); os.replace(dest + '.tmp', dest)
        print(f"{q['question_id']}: start {start} truth {gt} fired {fired} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == '__main__':
    main()
