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

## Real-time demo (camera or video)

```bash
pip install "fastapi>=0.110" "uvicorn[standard]>=0.29"
python -m robobeats.serve --gate-model superbigsaw/RoboBeats-2B
# open http://localhost:8000, then "Use camera", drop a video file, or click an example
```

The page sends one frame per second to the server and plots p as it comes back. Frames go only to the server you run.
The bundled example is decoded by the server from the original file with the same 1 fps rule as `python -m robobeats`, so its curves match the offline run exactly; the page only plays it.
On a remote GPU machine, forward the port first (`ssh -L 8000:127.0.0.1:8000 <machine>`); browsers only allow the camera on `localhost` or HTTPS.

**Your own conditions.** On the page you can add conditions of your own, such as "A person is drinking from a cup.", next to the new-event curve. They are judged together in one forward per frame, using the multi-condition prompt the released weights were trained with. Write each condition as a statement that is true or false at a given moment. Accuracy on free-text conditions of your own has not been benchmarked and depends on the wording. The new-event question was always trained on its own, so it runs as a separate session. With both on, each frame costs two forwards; switch the new-event gate off on the page to run conditions only.

For your own loop, use `LiveGate`. It runs one forward per new frame and reuses the model cache within a chunk:

```python
from robobeats.live import LiveGate

gate = LiveGate('superbigsaw/RoboBeats-2B')
events = gate.session()                                          # p(new event)
watch = gate.session(['A person is drinking from a cup.', 'The door is open.'])
f = gate.frame(t, image)  # t: integer second (increasing), image: PIL image; preprocessed once
p_event, = events.push(f)
p_drink, p_door = watch.push(f)
```

`LiveGate` follows the same prompt and chunking as `Gate.stream` and gives the same probabilities up to BF16 noise. On one H800 with RoboBeats-2B, we ran a 300 s TSU video:

- **New event vs `Gate.stream`:** max |Δp| 0.016; 1 of 301 frames lands on the other side of 0.5.
- **Three conditions, cached vs recomputed from scratch at every frame:** max |Δp| 0.019; no frame changes side at 0.5.
- **Speed:** a median of about 130 ms per session per frame. This counts the model call and tokenizing, but not image preprocessing, which runs once per frame.

To check your own setup:

```bash
python -m robobeats.live --check my_video.mp4 --gate-model superbigsaw/RoboBeats-2B --condition "A person is holding a cup."
```


## How it works

- **Causal 1 fps sampling.** For every integer second t, the frame is the last decoded frame with timestamp ≤ t.
- **One forward per chunk.** Frames go in chunks of 100 with an overlap of 10. After each judged frame comes a fixed marker, and p is read from the `false` / `true` logits at the marker. No answer token is written back, so p at second t only depends on frames up to t.
- **Hysteresis readout** (`select`). An event opens when p > 0.5 and closes at the first later second with p < 0.3. Each event keeps its frames from 2 s before it opens until it closes.
