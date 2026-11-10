"""Local web demo: stream your webcam or a video file through RoboBeats in real time.

  python -m robobeats.serve --gate-model superbigsaw/RoboBeats-2B [--device cuda:0] [--port 8000]

then open http://localhost:8000 . Frames go only to this server: the page grabs one frame per second from the camera
or from the video as it plays, sends it over a WebSocket, and plots p("a new event occurred").
On a remote GPU box, forward the port (ssh -L 8000:127.0.0.1:8000 box) and open the same address; browsers only allow
the camera on localhost or HTTPS.

Besides p(new event), you can type conditions of your own on the page; they are judged together in one forward
per frame (a second session, since the released weights learned the new-event question on its own). The new-event gate
can be switched off to run the conditions only.

The bundled examples are decoded here from the original file with the same 1 fps rule as `python -m robobeats`; the
page only plays the clip and reports which second it has reached, so the curves match the offline run frame for frame
(seconds the page skipped while busy, or by seeking forward, are judged too, in order).

One GPU serves the connections one frame at a time. For the camera and your own videos a tab never queues frames: it
sends the next frame only after the previous answer has come back, so if the model falls behind, seconds are skipped
instead of drifting from real time.
"""
import argparse
import asyncio
import io
import json
import struct
import time
from pathlib import Path

PAGE = Path(__file__).with_name('static') / 'index.html'


def build_app(gate, max_sessions):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from PIL import Image
    from .video import iter_ticks

    app, lock, live = FastAPI(), asyncio.Lock(), {'n': 0}

    @app.get('/')
    def index():
        return HTMLResponse(PAGE.read_text(encoding='utf-8'))

    from fastapi.staticfiles import StaticFiles
    app.mount('/examples', StaticFiles(directory=PAGE.with_name('examples')), name='examples')   # bundled example clips

    @app.websocket('/ws')
    async def ws(sock: WebSocket):
        """messages from the page (JSON text unless noted); every reply carries the page's stream number `gen`, so the page
        can drop answers that belong to a stream it has already restarted:
          {type: 'reset', gen}               new stream (new video, restarted camera, seek back)
          {type: 'conditions', conditions}   set / change / clear your conditions (takes effect at the next frame)
          {type: 'event', on}                switch the new-event gate on / off
          {type: 'example', file, gen}       play a bundled clip; frames are then decoded here from the original file
          {type: 'tick', t, gen}             the clip reached second t
          binary: uint32 gen, uint32 second, JPEG   a frame grabbed by the page (camera, your own video)"""
        await sock.accept()
        if live['n'] >= max_sessions:
            await sock.send_text(json.dumps(dict(error=f'server busy ({max_sessions} streams already open)')))
            await sock.close(); return
        live['n'] += 1
        await sock.send_text(json.dumps(dict(hello=dict(model=Path(gate.model_id.rstrip('/')).name))))
        events, watch = gate.session(), None          # p(new event) unless switched off; your conditions in a second, joint session
        example, done, gen = None, -1, 0              # bundled clip {second: image}, last second judged on it, stream number

        def join(new, old):                            # a session started mid-stream sees the current chunk's frames so far
            if old is not None and old.buf:
                new.buf, new.k, new.last_tick = list(old.buf), old.k, old.last_tick
            return new

        async def send(**kw):
            await sock.send_text(json.dumps(dict(kw, gen=gen)))

        async def infer(tick, image):
            async with lock:
                t0 = time.perf_counter()
                try:
                    p, pc = await asyncio.to_thread(push, events, watch, tick, image)
                except Exception as e:                 # a bad frame or a GPU error ends this frame, not the connection
                    await send(error=f'frame {tick}: {e}', t=tick); return
                ms = 1000 * (time.perf_counter() - t0)
            out = dict(t=tick, ms=round(ms, 1))
            if p is not None:
                out['p'] = round(p, 4)
            if watch is not None:
                out['conditions'] = [dict(text=c, p=round(q, 4)) for c, q in zip(watch.conditions, pc)]
            await send(**out)

        try:
            while True:
                msg = await sock.receive()
                if msg.get('type') == 'websocket.disconnect':
                    break
                if msg.get('text') is not None:
                    try:
                        cmd = json.loads(msg['text'])
                    except ValueError:
                        continue
                    kind = cmd.get('type')
                    if kind == 'reset':
                        gen, done = int(cmd.get('gen', gen)), -1
                        if events: events.close(); events = gate.session()
                        conds = watch.conditions if watch else None
                        if watch: watch.close()
                        watch = gate.session(conds) if conds else None
                        await send(reset=True)
                    elif kind == 'conditions':
                        conds = [c for c in cmd.get('conditions', []) if isinstance(c, str) and c.strip()]
                        try:
                            if not conds:
                                if watch: watch.close()
                                watch = None
                            elif watch is None:
                                watch = join(gate.session(conds), events)
                            else:
                                watch.set_conditions(conds)
                            await send(conditions=watch.conditions if watch else [])
                        except ValueError as e:
                            await send(error=str(e))
                    elif kind == 'event':
                        if cmd.get('on') and events is None:
                            events = join(gate.session(), watch)
                        elif not cmd.get('on') and events is not None:
                            events.close(); events = None
                        await send(event=events is not None)
                    elif kind == 'example':
                        gen, done = int(cmd.get('gen', gen)), -1
                        try:
                            example = await asyncio.to_thread(load_example, str(cmd.get('file', '')))
                            await send(example=cmd.get('file'), seconds=len(example))
                        except Exception as e:
                            example = None; await send(error=f'example: {e}')
                    elif kind == 'tick':
                        tick = int(cmd.get('t', -1))
                        if example is None or int(cmd.get('gen', gen)) != gen or tick <= done:
                            await send(skipped=tick); continue
                        # judge every second up to t in order, also ones the page skipped while busy or seeking forward,
                        # so the clip is judged on exactly the frames of the offline run
                        for t in range(done + 1, tick + 1):
                            if t in example:
                                await infer(t, example[t])
                        done = tick
                    elif kind == 'live':
                        example = None
                    continue
                data = msg.get('bytes') or b''
                if len(data) < 9:
                    continue
                g, tick = struct.unpack('<II', data[:8])
                if g != gen:
                    await send(skipped=tick); continue
                try:
                    image = Image.open(io.BytesIO(data[8:])).convert('RGB')
                except Exception as e:
                    await send(error=f'frame {tick}: unreadable image ({e})', t=tick); continue
                await infer(tick, image)
        except WebSocketDisconnect:
            pass
        finally:
            if events: events.close()
            if watch: watch.close()
            live['n'] -= 1
            gate.torch.cuda.empty_cache()

    examples = PAGE.with_name('examples')
    cache = {}

    def load_example(name):
        """1 fps frames of a bundled clip with the offline sampling rule (video.iter_ticks), as in `python -m robobeats`"""
        path = (examples / name).resolve()
        if path.parent != examples.resolve() or path.suffix != '.mp4' or not path.exists():
            raise ValueError(f'no such example: {name}')
        if name not in cache:
            import av
            with av.open(str(path)) as c:
                if c.duration is None:
                    raise ValueError(f'{name}: unknown duration')
                dur = c.duration / 1e6
            cache[name] = {t: im.convert('RGB') for t, _, im in iter_ticks(path, 0, dur)}
        return cache[name]

    # decode the bundled clips now: decoding on the first click competes with serving the same file to the page,
    # and the browser's player can stall waiting for it
    listing = examples / 'examples.json'
    if listing.exists():
        for ex in json.loads(listing.read_text(encoding='utf-8')):
            load_example(ex['file'])

    def push(events, watch, tick, image):
        f = gate.frame(tick, image)                                    # preprocess once for both sessions
        p = events.push(f)[0] if events is not None else None
        return p, (watch.push(f) if watch is not None else None)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--gate-model', default='superbigsaw/RoboBeats-2B')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--host', default='127.0.0.1', help='127.0.0.1 keeps the demo on this machine')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--max-sessions', type=int, default=4, help='open streams at once; each keeps its own cache on the GPU')
    a = ap.parse_args()
    import uvicorn
    from PIL import Image
    from .live import LiveGate
    print(f'loading {a.gate_model} on {a.device} ...', flush=True)
    gate = LiveGate(a.gate_model, a.device)
    for conds in (None, ['A person is walking.', 'A door is open.']):   # warm-up, so the first real frame is not slow
        s = gate.session(conds)
        for t in range(3):
            s.push(t, Image.new('RGB', (640, 360), (t * 60, 90, 120)))
        s.close()
    app = build_app(gate, a.max_sessions)
    print(f'ready: http://localhost:{a.port}', flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level='warning')


if __name__ == '__main__':
    main()
