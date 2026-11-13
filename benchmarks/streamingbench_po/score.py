"""Score a StreamingBench Proactive Output run (run.py): first-trigger accuracy, |fired - truth| <= 1 / 2 / 4 s.
The paper reports the +-2 s column (Table 2) and all three (Table 13).

  python benchmarks/streamingbench_po/score.py --run runs/sb_2B
"""
import argparse, glob, json, os


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0]); a.add_argument('--run', required=True); a = a.parse_args()
    rows = [r for f in sorted(glob.glob(os.path.join(a.run, 'rows_shard*.json'))) for r in json.load(open(f))]
    n = len(rows)
    acc = {w: 100 * sum(r['fired'] is not None and abs(r['fired'] - r['ground_truth']) <= w for r in rows) / n for w in (1, 2, 4)}
    print(f'| questions | +-2 s | +-1 s | +-4 s | never fired |\n|---|---|---|---|---|\n'
          f'| {n} | {acc[2]:.1f} | {acc[1]:.1f} | {acc[4]:.1f} | {100 * sum(r["fired"] is None for r in rows) / n:.1f}% |')
    if n != 250:
        print(f'note: the full set has 250 questions; this run has {n}')


if __name__ == '__main__':
    main()
