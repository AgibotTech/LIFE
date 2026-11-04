"""RoboBeats on SSS-Bench: gate (GPU) -> question-free memory writing -> reader -> predictions.jsonl.

  python -m sss_bench.reproduce.pipeline gate   --gate-model superbigsaw/RoboBeats-2B --work runs/rb2b [--shard i --nshard n]
  python -m sss_bench.reproduce.pipeline answer --work runs/rb2b --llm-base-urls http://127.0.0.1:8100/v1,... --llm-model Qwen3.8-27B

Every question is its own stream that starts at observation_start_s; frames after query_time_s are never decoded
into it (the gate is causal and the selection of a view only uses ticks <= query_time_s).
"""
import argparse, collections, concurrent.futures, hashlib, json, os, pathlib, threading, time, traceback

HERE = pathlib.Path(__file__).resolve().parents[1]


def load_questions(bench):
    return [json.loads(l) for l in (pathlib.Path(bench) / 'questions.jsonl').read_text().splitlines() if l.strip()]


def slug(video):
    return video.replace('/', '__').rsplit('.', 1)[0]


def gkey(video, start):
    return hashlib.sha1(json.dumps([video, start]).encode()).hexdigest()[:16]


def save(p, x):
    p = pathlib.Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.tmp{os.getpid()}'); tmp.write_text(json.dumps(x, ensure_ascii=False)); tmp.replace(p)


def stage_gate(a):
    from ..frames import png_bytes
    from robobeats.gate import Gate
    from robobeats.video import iter_ticks
    work = pathlib.Path(a.work)
    groups = collections.defaultdict(float)
    for q in load_questions(a.bench):
        k = (q['video'], q['observation_start_s']); groups[k] = max(groups[k], q['query_time_s'])
    todo = sorted(groups)[a.shard::a.nshard]
    gate = Gate(a.gate_model)
    for i, (video, start) in enumerate(todo):
        dest = work / 'gate' / f'{gkey(video, start)}.json'
        if dest.exists():
            continue
        t0 = time.time(); fdir = work / 'frames' / slug(video); fdir.mkdir(parents=True, exist_ok=True); pts = {}

        def frames():
            for t, p, im in iter_ticks(pathlib.Path(a.data_root) / video, start, groups[(video, start)]):
                png = fdir / f'{t:07d}.png'
                if not png.exists():
                    tmp = png.with_name(png.name + f'.tmp{os.getpid()}'); tmp.write_bytes(png_bytes(im)); tmp.replace(png)
                pts[t] = p
                yield t, im.convert('RGB')
        probs = gate.stream(frames())
        save(dest, dict(video=video, observation_start_s=start, end_s=groups[(video, start)], gate_model=a.gate_model,
                        ticks=[[t, pts[t], p] for t, p in probs]))
        print(f'[gate {a.shard}/{a.nshard}] {i + 1}/{len(todo)} {video} start={start} ticks={len(probs)} {time.time() - t0:.0f}s', flush=True)


def stage_answer(a):
    from ..llm import Client
    from .agent import read, write_memory
    from robobeats.gate import select
    work = pathlib.Path(a.work); llm = Client(a.llm_base_urls, a.llm_model)
    qs = load_questions(a.bench); locks = collections.defaultdict(threading.Lock)
    pred_dir = work / 'predictions' / a.reader_mode

    def job(q):
        dest = pred_dir / f"{q['id']}.json"
        if dest.exists():
            return json.loads(dest.read_text())
        try:
            g = json.loads((work / 'gate' / f"{gkey(q['video'], q['observation_start_s'])}.json").read_text())
            ticks = [x for x in g['ticks'] if x[0] <= q['query_time_s']]
            frames = [dict(t=t, pts=p) for t, p, _ in ticks]
            fdir = work / 'frames' / slug(q['video'])
            load_png = lambda f: (fdir / f"{f['t']:07d}.png").read_bytes()
            vk = gkey(q['video'], [q['observation_start_s'], q['query_time_s']])
            mpath = work / 'memory' / f'{vk}.json'
            with locks[vk]:
                if mpath.exists():
                    mem = json.loads(mpath.read_text())
                else:
                    sel = set(select([(t, p) for t, _, p in ticks], q['observation_start_s']))
                    chosen = [f for f in frames if f['t'] in sel]
                    with concurrent.futures.ThreadPoolExecutor(4) as pool:
                        records = write_memory(llm, chosen, load_png, pool)
                    mem = dict(video=q['video'], observation_start_s=q['observation_start_s'], query_time_s=q['query_time_s'],
                               candidate_frames=len(frames), selected_frames=len(chosen), records=records)
                    save(mpath, mem)
            out = dict(id=q['id'], reader_mode=a.reader_mode, selected_frames=mem['selected_frames'], candidate_frames=mem['candidate_frames'],
                       **read(llm, q, mem['records'], frames, load_png, a.reader_mode))
            save(dest, out); return out
        except Exception as e:
            (work / 'errors').mkdir(parents=True, exist_ok=True)
            (work / 'errors' / f"{q['id']}.txt").write_text(traceback.format_exc())
            return dict(id=q['id'], error=repr(e))

    done = 0; rows = []
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        for r in pool.map(job, qs):
            rows.append(r); done += 1
            if done % 25 == 0 or done == len(qs):
                print(f'[answer] {done}/{len(qs)}', flush=True)
    ok = [r for r in rows if 'answer' in r]
    with open(work / f'predictions_{a.reader_mode}.jsonl', 'w') as f:
        for r in ok:
            f.write(json.dumps(dict(id=r['id'], answer=r['answer']), ensure_ascii=False) + '\n')
    sel = sum(r['selected_frames'] for r in ok); cand = sum(r['candidate_frames'] for r in ok)
    print(json.dumps(dict(answered=len(ok), failed=len(rows) - len(ok), kept_frame_ratio=round(sel / max(1, cand), 4),
                          looked=sum(r['looked'] for r in ok))), flush=True)
    if len(ok) < len(rows):
        print(f'WARNING: {len(rows) - len(ok)} questions failed (see {work}/errors); rerun to retry, otherwise they count as wrong.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['gate', 'answer'])
    ap.add_argument('--work', required=True)
    ap.add_argument('--bench', default=str(HERE / 'data'))
    ap.add_argument('--data-root', default=os.environ.get('SSS_DATA_ROOT'))
    ap.add_argument('--gate-model', default='superbigsaw/RoboBeats-2B')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--llm-base-urls', default=os.environ.get('SSS_LLM_BASE_URLS', 'http://127.0.0.1:8100/v1'))
    ap.add_argument('--llm-model', default=os.environ.get('SSS_LLM_MODEL', 'Qwen3.8-27B'))
    ap.add_argument('--reader-mode', choices=['hard10', 'memoryonly'], default='hard10')
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    (stage_gate if a.stage == 'gate' else stage_answer)(a)


if __name__ == '__main__':
    main()
