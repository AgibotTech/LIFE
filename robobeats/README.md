# 🎵 RoboBeats

RoboBeats is a small, strictly causal VLM gate. It watches a 1 fps video stream and scores every frame with the probability that a new event has just occurred. Only the frames around detected events are written into memory.

## Models

| Model | Base model |
|---|---|
| [🤗 superbigsaw/RoboBeats-0.8B](https://huggingface.co/superbigsaw/RoboBeats-0.8B) | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| [🤗 superbigsaw/RoboBeats-2B](https://huggingface.co/superbigsaw/RoboBeats-2B) | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |
| [🤗 superbigsaw/RoboBeats-4B](https://huggingface.co/superbigsaw/RoboBeats-4B) | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |

Each model is a LoRA adapter; the base model is downloaded automatically. `--gate-model` also accepts a local directory with `adapter_config.json` and `adapter_model.safetensors`.

## Usage

Install the requirements as in the [main README](../README.md#installation), then from the repository root:

```bash
python -m robobeats my_video.mp4 --gate-model superbigsaw/RoboBeats-2B --out probs.json
# --start / --end   judge only part of the video (seconds)
# --device cuda:1   pick a GPU
```

It prints the detected events. `probs.json` holds every `[second, p]` and the event spans.

```python
from robobeats.gate import Gate, select
from robobeats.video import iter_ticks

gate = Gate('superbigsaw/RoboBeats-2B')
probs = gate.stream((t, im.convert('RGB')) for t, _, im in iter_ticks('my_video.mp4', 0, 600))  # [(second, p)]
kept = select(probs, start=0)  # seconds kept for memory
```

## How it works

- **Causal 1 fps sampling.** For every integer second t, the frame is the last decoded frame with timestamp ≤ t.
- **One forward per chunk.** Frames go in chunks of 100 with an overlap of 10. After each judged frame comes a fixed marker, and p is read from the `false` / `true` logits at the marker. No answer token is written back, so p at second t only depends on frames up to t.
- **Hysteresis readout** (`select`). An event opens when p > 0.5 and closes at the first later second with p < 0.3. Each event keeps its frames from 2 s before it opens until it closes.
