# 📊 SSS-Bench

SSS-Bench measures **sparse streaming video understanding**. A model watches a long video as a stream. When a question is asked, it must answer about when events happened, how often, in what order, and which objects or actors were involved, using only what it has seen so far.

- 618 questions on 171 source videos from 6 public datasets. Median observation length is 18 min, the longest 30 min.
- 14 question types: 260 numeric (time, interval, count) and 358 short-text (semantic).
- All times use the **original-video clock**, i.e. seconds from the first video frame.

| Dataset | Videos | Questions |
|---|---:|---:|
| Toyota Smarthome Untrimmed (TSU) | 103 | 349 |
| EGTEA Gaze+ | 30 | 118 |
| EPIC-KITCHENS-100 | 14 | 68 |
| Ego4D | 13 | 55 |
| UCF-Crime (UCA annotations) | 8 | 22 |
| Perception Test | 3 | 6 |

All commands below run from the repository root. **No video is redistributed.** Download the videos from the original providers under their own licenses (below).

## 1. Download the source videos

Put everything under one directory and set `SSS_DATA_ROOT` to it. The expected layout (also listed in [`data/videos.csv`](data/videos.csv)) is:

```
$SSS_DATA_ROOT/
  tsu/<id>.mp4                          # Toyota Smarthome Untrimmed
  egtea_gaze_plus/<id>.mp4              # EGTEA Gaze+
  epic_kitchens_100/<Pxx>/<Pxx_yy>.MP4  # EPIC-KITCHENS-100
  ego4d_video_540ss/<uid>.mp4           # Ego4D (video_540ss)
  ucf_crime/<name>_x264.mp4             # UCF-Crime videos (as used by UCA)
  perception_test/<video_id>.mp4        # Perception Test
```

| Dataset | Where to get it | Notes |
|---|---|---|
| TSU | request access on the project page: https://project.inria.fr/toyotasmarthome/ | RGB `.mp4`, keep the original file names |
| EGTEA Gaze+ | https://cbs.ic.gatech.edu/fpv/ | raw videos, not the trimmed clips |
| EPIC-KITCHENS-100 | official scripts: https://github.com/epic-kitchens/epic-kitchens-download-scripts, or the community mirror `huggingface-cli download awsaf49/epic_kitchens_100 --repo-type dataset --include "P23/*" "P24/*" "P25/*" "P29/*" --local-dir $SSS_DATA_ROOT/epic_kitchens_100` | check that the mirror's layout matches `<Pxx>/<Pxx_yy>.MP4` |
| Ego4D | sign the Ego4D license, then `ego4d --output_directory <dir> --datasets video_540ss --video_uids $(cat sss_bench/scripts/ego4d_video_uids.txt)` | 540p short side; timestamps equal the full-resolution video |
| UCF-Crime (UCA) | videos: https://www.crcv.ucf.edu/projects/real-world/ (UCA annotations: https://github.com/Xuange923/UCA-dataset) | flatten to `ucf_crime/<name>_x264.mp4` |
| Perception Test | https://github.com/google-deepmind/perception_test (train videos) | |

Then check the download:

```bash
python sss_bench/scripts/check_videos.py $SSS_DATA_ROOT     # all 171 present and long enough
```

> Some decoders used by streaming models cannot read four EPIC-KITCHENS videos (P24_01, P24_02, P24_07, P25_09). Do not drop these questions, because a failure counts as wrong. Re-encoding to H.264 keeps the timestamps.

## 2. Question format

`data/questions.jsonl` has one object per line:

| Field | Meaning |
|---|---|
| `id` | question id |
| `video` | path relative to `$SSS_DATA_ROOT` |
| `observation_start_s`, `query_time_s` | the model may only see frames in `[observation_start_s, query_time_s]` (original-video seconds); the question is asked at `query_time_s` |
| `interval_start_s`, `interval_end_s` | the interval the question refers to |
| `question` / `question_zh` | English question (the evaluated text) / Chinese translation |
| `question_type`, `answer_format` | one of 14 types; `number` or `short text` |
| `dataset`, `source_video_id`, `source_group` | provenance (`source_group` = subject or scene, for grouped analysis) |

The protocol is streaming and causal: no frame after `query_time_s` may be used. Numeric answers are in original-video seconds, or counts.

## 3. Evaluate your own model

Run your model on `data/questions.jsonl` under the protocol above and write predictions as JSONL: `{"id": ..., "answer": ...}`. The answer can be a number, a string, or a JSON object with an `answer` field.

The semantic judge is **Qwen3.8-27B (BF16)** served with vLLM, with the settings from `rubric/judge_prompts.json`:

```bash
vllm serve <Qwen3.8-27B weights> --served-model-name Qwen3.8-27B --dtype bfloat16 \
  --max-model-len 65536 --limit-mm-per-prompt '{"image":80,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":200704,"min_pixels":3136}'

python -m sss_bench.evaluate --pred preds.jsonl --data-root $SSS_DATA_ROOT \
  --judge-base-url http://127.0.0.1:8000/v1 --judge-model Qwen3.8-27B --out results/my_model
# endpoint needs a key:                          export SSS_JUDGE_API_KEY=...
# model answers numeric questions in free text:  add --free-text-numeric (and report it as such)
```

The evaluator writes `per_question/*.json` and `summary.json`. The summary gives overall accuracy, broken down by question type, dataset and observation length. Missing answers count as wrong. Judge and video errors are not cached, so rerunning the same command retries only those.

Grading in brief (the full specification is in [`rubric/GRADING.md`](rubric/GRADING.md)):

- **Numeric (260)**: parsed deterministically. Tolerances are ±3 s for `event_time` and `first_last_time`, ±2 s for `event_interval`, and exact for `event_count`.
- **Semantic (358)**: the judge sees the question, the candidate answer and the reference rubric (canonical answer, minimal core, acceptable variants, contradictions). It also sees up to 80 evidence frames from the original video.

## About the references

Questions and reference answers were drafted with LLM assistance from the videos and then **independently verified by visual model review**. They are not human-annotated. Native dataset labels were used only as unverified hints. If you find an error, please open an issue with the question id.

## Reproducing RoboBeats

```
video (1 fps, causal) ──► RoboBeats gate ──► hysteresis selection ──► 27B writer (question-free memory)
                                                                        │
                                       question ──► 27B reader (memory + ONE look of ≤ 10 frames) ──► answer
```

- **Gate**: [RoboBeats](../robobeats/README.md) with the condition "Has a new event occurred?". Every question is its own stream, starting at `observation_start_s`.
- **Selection**: an event opens at p > 0.5 and closes at the first later frame with p < 0.3; each event keeps its frames from 2 s before onset until it closes.
- **Writer**: Qwen3.8-27B summarizes the kept frames in packs (≤ 60 frames, < 60 s) with source-video timestamps. It never sees the question.
- **Reader** (`hard10`): Qwen3.8-27B receives the memory and the question. It may request one window of the observed prefix and gets at most 10 frames from it, then answers. `memoryonly` gives no look.

One command, with 8 GPUs by default (the gate on GPU 0, one 27B vLLM server on each of GPUs 1–7):

```bash
SSS_DATA_ROOT=/path/to/videos LLM_MODEL=/path/to/Qwen3.8-27B GATE=2B bash sss_bench/scripts/run_robobeats.sh
# GATE=0.8B | 2B | 4B;  READER_MODE=memoryonly for the no-look variant
# existing 27B endpoints:  LLM_BASE_URLS=http://host1:8000/v1,http://host2:8000/v1 bash sss_bench/scripts/run_robobeats.sh
```

The script runs these stages:

1. checks the videos;
2. runs the gate, sharded over `GATE_GPUS`;
3. starts vLLM;
4. writes memories and answers the questions;
5. grades the answers.

Outputs go to `runs/robobeats_<GATE>/`:

- `predictions_<mode>.jsonl` and `results_<mode>/summary.json`;
- the memories, gate probabilities and look windows for each question.

Every stage caches its results, so rerunning resumes after an interruption.

The stages can also be run one by one:

```bash
python -m sss_bench.reproduce.pipeline gate   --work runs/rb2b --gate-model superbigsaw/RoboBeats-2B --shard 0 --nshard 1
python -m sss_bench.reproduce.pipeline answer --work runs/rb2b --llm-base-urls http://127.0.0.1:8100/v1 --llm-model Qwen3.8-27B
python -m sss_bench.evaluate --pred runs/rb2b/predictions_hard10.jsonl --data-root $SSS_DATA_ROOT \
  --judge-base-url http://127.0.0.1:8100/v1 --judge-model Qwen3.8-27B --out runs/rb2b/results
```
