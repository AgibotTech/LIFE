"""SSS-Bench evaluator.

predictions: JSONL, one {"id": ..., "answer": ...} per question (answer: number, string or JSON object).
Numeric questions are scored deterministically; semantic questions by a visual judge on an
OpenAI-compatible endpoint (see rubric/GRADING.md). Missing predictions count as wrong.

  python -m sss_bench.evaluate --pred preds.jsonl --data-root $SSS_DATA_ROOT \
      --judge-base-url http://127.0.0.1:8000/v1 --judge-model Qwen3.8-27B --out results/
API key (if the endpoint needs one) is read from the env var SSS_JUDGE_API_KEY.
"""
import argparse, base64, collections, concurrent.futures, json, os, pathlib, threading

from .frames import png_bytes, sample
from .llm import Client
from .number_policy import number

HERE = pathlib.Path(__file__).resolve().parent
PROMPTS = json.loads((HERE / 'rubric/judge_prompts.json').read_text())
_video_locks = collections.defaultdict(threading.Lock)


def load_jsonl(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]


_client = None


def chat(args, messages, max_tokens):
    global _client
    if _client is None:
        _client = Client(args.judge_base_url, args.judge_model)
    return _client.chat(messages, max_tokens, temperature=0, response_format={'type': 'json_object'})[0]


def parse_obj(text):
    return json.loads(text[text.find('{'):text.rfind('}') + 1])


def extract_free_text_number(args, q, answer, tol):
    msg = [dict(role='system', content=PROMPTS['free_text_numeric_extraction_system']),
           dict(role='user', content=json.dumps(dict(question=q['question'], candidate_response=answer), ensure_ascii=False))]
    try:
        ex = parse_obj(chat(args, msg, 80))
    except Exception as e:
        return None, dict(kind='error', error=repr(e))
    num = lambda k: isinstance(ex.get(k), (int, float)) and not isinstance(ex.get(k), bool)
    if ex.get('kind') == 'single' and num('value'):
        return float(ex['value']), ex
    if ex.get('kind') == 'range' and num('low') and num('high'):
        lo, hi = sorted((float(ex['low']), float(ex['high'])))
        return ((lo + hi) / 2 if hi - lo <= 2 * tol else None), ex
    return None, ex


def grade_numeric(args, q, ref, answer):
    tol = float(ref['tolerance'])
    if args.free_text_numeric:
        v, ex = extract_free_text_number(args, q, answer, tol)
        method = 'free_text_reference_blind_extraction'
    else:
        v, ex, method = number(answer), None, 'deterministic_numeric'
    err = None if v is None else abs(v - float(ref['answer']))
    return dict(correct=v is not None and err <= tol, parsed_value=v, absolute_error=err, extraction=ex, grading_method=method)


def evidence_parts(args, q, ref):
    path = pathlib.Path(args.data_root) / q['video']
    secs = [t for t in ref['evidence_seconds'] if q['observation_start_s'] <= t <= q['query_time_s']]
    with _video_locks[str(path)]:
        frames = sample(path, secs)
    parts, seen = [], set()
    for t in sorted(frames):
        pts, im = frames[t]
        if pts in seen:
            continue
        seen.add(pts)
        parts += [dict(type='text', text=f'Original-video seconds: {pts:.6f} (sample tick {t})'),
                  dict(type='image_url', image_url={'url': 'data:image/png;base64,' + base64.b64encode(png_bytes(im)).decode()})]
    if not 1 <= len(seen) <= 80:
        raise ValueError(f"{q['id']}: {len(seen)} evidence frames")
    return parts


def grade_semantic(args, q, ref, answer):
    reference = dict(acceptable_variants=ref['acceptable_variants'], canonical=ref['answer'], contradictions=ref['contradictions'],
                     minimal_core=ref['minimal_core'], mode=ref['mode'], required_facts=ref['required_facts'],
                     rubric_version=ref['rubric_version'], tolerance=ref['tolerance'])
    payload = dict(question=q['question'], question_type=q['question_type'], reference=reference, candidate=answer)
    messages = [dict(role='system', content=PROMPTS['semantic_judge_system']),
                dict(role='user', content=[dict(type='text', text=json.dumps(payload, ensure_ascii=False))] + evidence_parts(args, q, ref))]
    keys = ['correct', 'core_answer_present', 'contradiction', 'reference_issue']
    text = chat(args, messages, 1536)
    try:
        j = parse_obj(text); assert all(type(j[k]) is bool for k in keys)
    except (KeyError, AssertionError, ValueError, TypeError):
        text = chat(args, messages + [dict(role='user', content=PROMPTS['semantic_judge_format_recovery'])], 1536)
        j = parse_obj(text); assert all(type(j[k]) is bool for k in keys)
    return dict(j, grading_method='visual_rubric_judge')


def length_bin(q):
    s = q['query_time_s'] - q['observation_start_s']
    return '<10 min' if s < 600 else '10-15 min' if s < 900 else '15-20 min' if s < 1200 else '>=20 min'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--data-root', default=os.environ.get('SSS_DATA_ROOT'), help='root of the downloaded source videos')
    # --judge-base-url accepts a comma-separated list of endpoints (round-robin)
    ap.add_argument('--bench', default=str(HERE / 'data'))
    ap.add_argument('--judge-base-url', default=os.environ.get('SSS_JUDGE_BASE_URL', 'http://127.0.0.1:8000/v1'))
    ap.add_argument('--judge-model', default=os.environ.get('SSS_JUDGE_MODEL', 'Qwen3.8-27B'))
    ap.add_argument('--free-text-numeric', action='store_true', help='reference-blind judge extraction for free-text numeric answers')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    Q = load_jsonl(f'{args.bench}/questions.jsonl'); A = {r['id']: r for r in load_jsonl(f'{args.bench}/answers.jsonl')}
    P = {r['id']: r['answer'] for r in load_jsonl(args.pred)}
    out = pathlib.Path(args.out); (out / 'per_question').mkdir(parents=True, exist_ok=True)

    def job(q):
        dest = out / 'per_question' / f"{q['id']}.json"
        if dest.exists():
            return json.loads(dest.read_text())
        ref = A[q['id']]; row = dict(id=q['id'], question_type=q['question_type'], dataset=q['dataset'], length_bin=length_bin(q))
        if q['id'] not in P or P[q['id']] in (None, ''):
            row.update(correct=False, grading_method='missing_prediction')
        else:
            row['answer'] = P[q['id']]
            try:
                row.update((grade_numeric if ref['mode'] == 'numeric' else grade_semantic)(args, q, ref, P[q['id']]))
            except Exception as e:  # infrastructure error: not written, rerun to retry
                return dict(row, correct=None, error=repr(e))
        dest.write_text(json.dumps(row, ensure_ascii=False))
        return row

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(job, Q))
    errors = [r for r in rows if r['correct'] is None]
    acc = lambda rs: round(100 * sum(bool(r['correct']) for r in rs) / len(rs), 1)
    summary = dict(n=len(rows), accuracy=acc(rows), errors_counted_wrong=len(errors), free_text_numeric=args.free_text_numeric)
    for k in ('question_type', 'dataset', 'length_bin'):
        g = collections.defaultdict(list)
        for r in rows:
            g[r[k]].append(r)
        summary['by_' + k] = {c: dict(acc=acc(v), n=len(v)) for c, v in sorted(g.items())}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if errors:
        print(f'WARNING: {len(errors)} judge/video errors counted as wrong; rerun the same command to retry them.')


if __name__ == '__main__':
    main()
