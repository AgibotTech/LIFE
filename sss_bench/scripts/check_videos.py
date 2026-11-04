"""Check that every source video listed in data/videos.csv exists under $SSS_DATA_ROOT and covers its questions."""
import csv, json, os, pathlib, sys

import av

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.environ['SSS_DATA_ROOT'])
here = pathlib.Path(__file__).resolve().parent.parent
need = {}
for l in (here / 'data/questions.jsonl').read_text().splitlines():
    q = json.loads(l); need[q['video']] = max(need.get(q['video'], 0), q['query_time_s'])
missing, short = [], []
for row in csv.DictReader(open(here / 'data/videos.csv')):
    p = root / row['relative_path']
    if not p.exists():
        missing.append(row['relative_path']); continue
    with av.open(str(p)) as c:
        dur = float(c.duration / 1e6) if c.duration else None
    if dur is not None and dur + 1 < need[row['relative_path']]:
        short.append((row['relative_path'], dur, need[row['relative_path']]))
print(f'videos: {len(need)}  missing: {len(missing)}  too short: {len(short)}')
for m in missing: print('MISSING', m)
for s in short: print('SHORT', *s)
sys.exit(1 if missing or short else 0)
