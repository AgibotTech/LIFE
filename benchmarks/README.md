# Condition-tracking benchmarks

Scripts that reproduce RoboBeats' condition-tracking results in the paper (Table 2; full results in Tables 12–14). Each benchmark has a `run` step, which runs the gate and saves per-second probabilities or responses, and a `score` step. To try another readout, re-run only the `score` step.

| Benchmark | Paper | Metric | Frame rate | RoboBeats-0.8B / 2B / 4B |
|---|---|---|---|---|
| [Charades-STA](#charades-sta) | Tables 2, 12 | causal online R@0.5 | 1 fps | 53.1 / 57.3 / 57.2 |
| [StreamingBench Proactive Output](#streamingbench-proactive-output) | Tables 2, 13 | first trigger within ±2 s | 1 fps | 51.2 / 56.0 / 64.0 |
| [SPOT-Bench ABD](#spot-bench-abd) | Tables 2, 14 | T-F1@5 (official scorer) | 2 fps | 40.7 / 43.8 / 45.8 |

All three use the condition-judgment protocol: the request becomes one or more conditions (statements that are true or false at a moment), and the gate judges them causally, second by second, from the moment the request is made. In Python this is `Gate.track(frames, conditions)`; on the command line it is `python -m robobeats video.mp4 --condition "..."`.

No video is redistributed. Download each benchmark from its source under its own license.

## Setup

Install the [requirements](../requirements.txt), then run every command from the repository root. The reported numbers were computed with `flash-linear-attention==0.5.2` installed (it is in the requirements). Without it, transformers falls back to a PyTorch implementation of the linear-attention layers. Per-second probabilities then differ by about 10⁻³ (BF16 rounding), which can move a score by a few tenths.

`--gate-model` takes `superbigsaw/RoboBeats-{0.8B,2B,4B}` or a local adapter folder. To use several GPUs, start one process per GPU with `--shard i --nshard n` and the same `--out`; the score step reads every shard.

## Charades-STA

Data: the Charades videos at 480p ([Charades_v1_480](https://prior.allenai.org/projects/charades)) and the Charades-STA test annotations `charades_sta_test.txt` ([Charades-STA](https://github.com/jiyanggao/TALL), 3,720 queries).

```bash
python benchmarks/charades_sta/run.py --videos /data/Charades_v1_480 --annotations /data/charades_sta_test.txt \
    --out runs/charades_2B --gate-model superbigsaw/RoboBeats-2B
python benchmarks/charades_sta/score.py --run runs/charades_2B --annotations /data/charades_sta_test.txt
```

Each query is a condition tracked on its video from 0 s at 1 fps. A video's queries are judged together, in fixed random groups of at most 8. Online readout: a query is on from p > 0.5 while p ≥ 0.3; on-runs less than 1 s apart are merged; the answer is the first run that lasts at least 1 s. The answer uses no frame after the moment it is given. The paper reports R@0.3 / 0.5 / 0.7 and mIoU (Table 12). For RoboBeats-2B these are 74.4 / 57.3 / 28.9 and 49.7. `--offline` adds the offline readout.

RoboBeats' training data includes Charades-STA training videos. The test videos were not used.

## StreamingBench Proactive Output

Data: [StreamingBench](https://github.com/THUNLP-MT/StreamingBench), `csv/Proactive_Output.csv` and `proactive_videos/` (250 questions).

```bash
python benchmarks/streamingbench_po/run.py --bench /data/StreamingBench --out runs/sb_2B --gate-model superbigsaw/RoboBeats-2B
python benchmarks/streamingbench_po/score.py --run runs/sb_2B
```

Each question reads "When X, output 'Y'". X becomes one condition, tracked at 1 fps on a stream that starts at the question's time stamp. The answer fires at the first later second with p > 0.5. Scored as |fire − truth| ≤ 2 s (Table 2), and also ±1 s and ±4 s (Table 13). For RoboBeats-2B these are 56.0 / 45.6 / 70.8.

## SPOT-Bench ABD

Data and harness: the official [SPOT-Bench](https://github.com/dibschat/SPOT-Bench) repository with its data from [Hugging Face](https://huggingface.co/datasets/cvml-nus/spot-bench) (access requires the Ego4D / Ego-Exo4D agreements). RoboBeats runs as a model of the official harness, and the official scorer computes the results.

```bash
cd /path/to/SPOT-Bench
cp /path/to/LIFE/benchmarks/spot_abd/robobeats_spot.py model_wrappers/robobeats.py
# add to MODEL_REGISTRY in model_wrappers/__init__.py:
#     "RoboBeats": ("model_wrappers.robobeats", "RoboBeats"),
LIFE_ROOT=/path/to/LIFE ROBOBEATS_MODEL=superbigsaw/RoboBeats-2B \
    python inference.py --task ABD --model RoboBeats --annotation_path data/abd.json --video_root data/videos --result_dir results
python score.py --tasks ABD --model RoboBeats --result_dir results --annotation_root data
```

Each request is turned into a start condition and a finish condition. These are judged together at 2 fps (the input rate of the benchmark's streaming baselines) from the moment the question is asked. "start" fires at the first rise of the start condition above 0.5. "finish" fires when the finish condition, having been on, drops below 0.3. Each word fires once, at least 1 s after the question. The rewrites are in [`spot_abd/abd_rewrites.json`](spot_abd/abd_rewrites.json). They were written once by the plain Qwen3.5-2B, so every model is judged on the same conditions. For RoboBeats-2B: T-score@5 42.9, T-F1@5 43.8; T-F1@5 48.5 on Ego-Exo4D and 33.5 on HTStep.

`abd_rewrites.json` is derived from SPOT-Bench questions and is shared under SPOT-Bench's license, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
