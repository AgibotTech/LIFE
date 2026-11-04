"""Memory writer (question-free) and reader for the RoboBeats pipeline; both use the same 27B model.

writer: the gate-selected frames are cut into packs (<= 60 frames and < 60 s each); each pack is summarized by W1
        with source-video timestamps. The writer never sees the question.
reader: gets the whole memory and the question; in hard10 mode it may request ONE window of the observation prefix
        and receives at most 10 frames of it (uniform over the window, endpoints included), then answers.
"""
import base64, bisect, json, math

from . import prompts as P

SAMPLING = dict(temperature=0.1, top_k=-1, min_p=0.0, repetition_penalty=1.0)


def image_parts(frames, load_png):
    out = []
    for f in frames:
        out += [dict(type='text', text=f"Original-video seconds: {f['pts']:.6f} (sample tick {f['t']})"),
                dict(type='image_url', image_url={'url': 'data:image/png;base64,' + base64.b64encode(load_png(f)).decode()})]
    return out


def packs(frames):
    out, buf = [], []
    for f in frames:
        if buf and (len(buf) >= 60 or f['t'] - buf[0]['t'] >= 60 or f['pts'] - buf[0]['pts'] >= 60):
            out.append(buf); buf = []
        buf.append(f)
    return out + ([buf] if buf else [])


def write_pack(llm, pack, target, load_png):
    messages = [dict(role='system', content=P.WRITER_SYSTEM),
                dict(role='user', content=[dict(type='text', text=P.W1 + f' Aim for {target} tokens; preserve event distinctions. Maximum output 1024 tokens.')] + image_parts(pack, load_png))]
    text, fin = llm.chat(messages, 1024, **SAMPLING)
    if fin == 'length':   # 1) rewrite shorter
        text, fin = llm.chat(messages + [dict(role='assistant', content=text), dict(role='user', content=P.WRITER_LENGTH_REWRITE)], 1024, **SAMPLING)
    if fin == 'length':   # 2) fresh schema-bound request
        fresh = json.loads(json.dumps(messages)); fresh[1]['content'][0]['text'] += '\n' + P.WRITER_SCHEMA_RECOVERY
        schema = {'type': 'object', 'properties': {'memory': {'type': 'string', 'minLength': 1, 'maxLength': 2400}}, 'required': ['memory'], 'additionalProperties': False}
        raw, fin = llm.chat(fresh, 1024, structured_outputs={'json': schema}, **SAMPLING)
        try:
            text = json.loads(raw)['memory']
        except Exception:
            fin = 'length'
        if fin != 'length' and len(text) >= 2200:
            fin = 'length'
    if fin == 'length' and len(pack) > 1:   # 3) split the pack in halves
        h = len(pack) // 2
        a, b = write_pack(llm, pack[:h], target, load_png), write_pack(llm, pack[h:], target, load_png)
        return dict(start=pack[0]['pts'], end=pack[-1]['pts'], text=a['text'] + '\n' + b['text'], frames=[f['t'] for f in pack], split=True)
    return dict(start=pack[0]['pts'], end=pack[-1]['pts'], text=text, frames=[f['t'] for f in pack])


def write_memory(llm, selected, load_png, pool):
    batches = packs(selected)
    target = min(220, max(64, 32000 // max(1, len(batches)) - 24))
    return list(pool.map(lambda pk: write_pack(llm, pk, target, load_png), batches))


def answer_schema(q, look):
    fmt = q['answer_format']
    value = {'type': 'number' if fmt == 'number' else 'string'}
    if q['question_type'] == 'event_count':
        value = {'type': 'integer', 'minimum': 0}
    if q['question_type'] == 'event_time':
        value.update(minimum=q['observation_start_s'], maximum=q['query_time_s'])
    if q['question_type'] == 'event_interval':
        value.update(minimum=0, maximum=q['query_time_s'] - q['observation_start_s'])
    a = {'type': 'object', 'properties': {'action': {'const': 'answer'}, 'answer': value}, 'required': ['action', 'answer'], 'additionalProperties': False}
    if not look:
        return a
    lim = {'type': 'number', 'minimum': q['observation_start_s'], 'maximum': q['query_time_s']}
    b = {'type': 'object', 'properties': {'action': {'const': 'look'}, 'start': lim, 'end': lim, 'target': {'type': 'string'}},
         'required': ['action', 'start', 'end', 'target'], 'additionalProperties': False}
    return {'anyOf': [a, b]}


def _valid(j, q, allow_look):
    if not isinstance(j, dict) or j.get('action') not in (['answer', 'look'] if allow_look else ['answer']):
        return False
    if j['action'] == 'look':
        return all(isinstance(j.get(k), (int, float)) and math.isfinite(j[k]) for k in ('start', 'end')) and isinstance(j.get('target'), str)
    a = j.get('answer')
    if q['question_type'] == 'event_count':
        return type(a) is int and a >= 0
    if q['answer_format'] == 'number':
        return type(a) in (int, float) and math.isfinite(a)
    return isinstance(a, str) and bool(a.strip())


def _decide(llm, messages, q, allow_look):
    schema = {'structured_outputs': {'json': answer_schema(q, allow_look)}}
    text, fin = llm.chat(messages, 512, **schema, **SAMPLING)
    try:
        j = json.loads(text); ok = fin != 'length' and _valid(j, q, allow_look)
    except Exception:
        j, ok = None, False
    if not ok:
        rec = P.READER_LENGTH_RECOVERY if fin == 'length' else P.READER_FORMAT_RECOVERY
        if not allow_look:
            rec = rec.replace('final answer or your single allowed look decision', 'final answer')
        text, fin = llm.chat(messages + [dict(role='user', content=rec)], 512, **schema, **SAMPLING)
        j = json.loads(text); assert _valid(j, q, allow_look), text
    return j, text


def review_frames(frames, start, end, q, max_frames=10):
    """<= 10 distinct frames of [start, end]: all if <= 10, else uniform over the window in source time, endpoints included"""
    cand = [f for f in frames if start <= f['pts'] <= end and q['observation_start_s'] <= f['pts'] <= q['query_time_s']]
    times, n = [f['pts'] for f in cand], len(cand)
    if n <= max_frames:
        return cand
    idx = [0]
    for i in range(1, max_frames - 1):
        target = times[0] + (times[-1] - times[0]) * i / (max_frames - 1)
        lo, hi = idx[-1] + 1, n - max_frames + i
        pos = bisect.bisect_left(times, target, lo, hi + 1)
        idx.append(min({max(lo, min(hi, pos)), max(lo, min(hi, pos - 1))}, key=lambda j: (abs(times[j] - target), times[j], j)))
    idx.append(n - 1)
    return [cand[i] for i in idx]


def read(llm, q, memory, all_frames, load_png, mode='hard10'):
    text = '\n\n'.join(f"Observation pack {i + 1}; source seconds {r['start']:.6f}–{r['end']:.6f}:\n{r['text']}" for i, r in enumerate(memory)) or P.EMPTY_MEMORY
    payload = dict(memory=text, question=q['question'], answer_format=q['answer_format'], observation_start_s=q['observation_start_s'],
                   observation_cutoff_s=q['query_time_s'], clock=q['clock'])
    look_ok = mode == 'hard10'
    messages = [dict(role='system', content=P.READER_SYSTEM + (' ' + P.LOOK_RULE if look_ok else P.MEMORY_ONLY_RULE)),
                dict(role='user', content=json.dumps(payload, ensure_ascii=False))]
    j, raw = _decide(llm, messages, q, look_ok)
    if j['action'] == 'look' and not (q['observation_start_s'] <= j['start'] <= j['end'] <= q['query_time_s']):
        raw, _ = llm.chat(messages + [dict(role='user', content=P.READER_WINDOW_RECOVERY)], 512,
                          structured_outputs={'json': answer_schema(q, True)}, **SAMPLING)
        j = json.loads(raw)
        if j['action'] == 'look':
            assert q['observation_start_s'] <= j['start'] <= j['end'] <= q['query_time_s'], raw
    look = None
    if j['action'] == 'look':
        chosen = review_frames(all_frames, j['start'], j['end'], q)
        look = dict(start=j['start'], end=j['end'], target=j['target'], frames=[f['t'] for f in chosen])
        messages += [dict(role='assistant', content=raw),
                     dict(role='user', content=[dict(type='text', text=P.LOOK_INTRO if chosen else P.LOOK_EMPTY)] + image_parts(chosen, load_png))]
        j, raw = _decide(llm, messages, q, False)
    return dict(answer=j['answer'], looked=look is not None, look=look)
