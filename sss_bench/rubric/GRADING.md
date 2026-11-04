# SSS-Bench grading protocol

Each question has a reference entry in `data/answers.jsonl`. `mode` decides how it is graded.

## 1. Numeric questions (`mode = numeric`, 260 questions)

| question_type | count | unit | tolerance |
|---|---|---|---|
| event_time | 163 | original-video seconds (time point) | ±3 s |
| event_interval | 82 | seconds (duration / gap) | ±2 s |
| event_count | 13 | count | exact (0) |
| first_last_time | 2 | original-video seconds | ±3 s |

Correct iff `|parsed - answer| <= tolerance`. Parsing is deterministic (`sss_bench/number_policy.py`):

- A JSON number or `{"answer": ...}` / `{"time_s": ...}` / `{"value": ...}` is used directly.
- Text containing not/never/no counts as no answer.
- A single `mm:ss` / `hh:mm:ss` is converted to seconds.
- Otherwise the text must contain exactly one number; zero or several numbers count as no answer (wrong).

All times are **original-video seconds** (the `clock` field). Answers relative to the observation start are not converted.

### Free-text models (optional `--free-text-numeric`)
For models that cannot follow the output format and answer in free text (e.g. native streaming models), a judge first extracts the committed number **without seeing the reference** (prompt in `judge_prompts.json`):

- a single value → used as is;
- a range → its midpoint, used only if `high - low <= 2 × tolerance`; otherwise no answer;
- no number → wrong.

No unit or clock conversion or offset is applied. Results produced this way must be reported as "free-text extraction".

## 2. Semantic questions (`mode = semantic`, 358 questions)

Types: nth_event_detail 123, target_change 93, next_event 62, object_identity 17, object_destination 17, interlude 13, state_change 13, actor_object_binding 9, incident_response 7, irregular_order 4.

The judge model (Qwen3.8-27B, temperature 0) receives:

1. the question, the candidate answer and the full reference rubric:
   - `answer`: the canonical answer;
   - `minimal_core` / `required_facts`: the minimal core the answer must hit;
   - `acceptable_variants`: equivalent phrasings;
   - `contradictions`: typical wrong answers that make the answer wrong;
2. evidence frames from the original video at `evidence_seconds` (at most 80, 448 px, each labelled with its original-video second).

The judge returns `{correct, core_answer_present, contradiction, reference_issue, reason_zh}`. Score = `correct`. If the JSON is invalid, one format-recovery prompt is sent (same input).

Principles:
- Equivalent wording counts as correct.
- Optional details are not required.
- Unsupported contradictory details are not rewarded.
- Anything that hits `contradictions` is wrong.

## 3. Reporting

- Accuracy = correct / 618.
- Breakdowns by `question_type`, `dataset`, and observation length (`query_time_s - observation_start_s`: <10 / 10–15 / 15–20 / ≥20 min).
- Any failure (no answer, decode failure) counts as wrong. Do not drop questions.

## Notes on the references
Questions and references were built from the videos by models and then independently verified visually by models (`verification_status = independent_model_visual_reviewed`). **They are not human-annotated gold.** Native dataset labels were used only as unverified hints.
