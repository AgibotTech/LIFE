"""Frame sampling shared by the benchmark runners, as used for the numbers in the paper."""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)             # so `import robobeats` works from any working directory


def frames_every_step(path, fps=1.0):
    """every round(frame_rate / fps)-th decoded frame, as PIL images (StreamingBench and SPOT-Bench runs)"""
    import av
    from PIL import Image
    out = []
    with av.open(str(path)) as c:
        st = c.streams.video[0]; st.thread_type = 'AUTO'
        step = max(1, round(float(st.average_rate or fps) / fps))
        for i, fr in enumerate(c.decode(st)):
            if i % step == 0:
                out.append(Image.fromarray(fr.to_ndarray(format='rgb24')))
    if not out:
        raise ValueError(f'no frames decoded from {path}')
    return out


def n_frames_at(path, fps):
    """number of samples at `fps` over the video's duration (Charades-STA runs)"""
    import av
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        dur = float(st.duration * st.time_base) if st.duration else float(c.duration / 1e6)
    return int(dur * fps) + 1


def frames_at_presentation_time(path, fps, n):
    """sample i = the first decoded frame shown at or after i / fps s, for i < n; the last frame is repeated if the video
    ends early (Charades-STA runs)"""
    import av
    from PIL import Image
    out, nxt, i = {}, 0.0, 0
    with av.open(str(path)) as c:
        st = c.streams.video[0]; st.thread_type = 'AUTO'
        for fr in c.decode(st):
            if fr.time is None:
                continue
            while fr.time + 1e-6 >= nxt and i < n:
                out[i] = Image.fromarray(fr.to_ndarray(format='rgb24')); i += 1; nxt = i / fps
            if i >= n:
                break
    if not out:
        raise ValueError(f'no frames decoded from {path}')
    last = out[max(out)]
    return [out.get(j, last) for j in range(n)]


def jpeg(image, quality=95):
    """JPEG round trip (the Charades-STA runs passed frames to the gate as JPEG q95)"""
    from PIL import Image
    b = io.BytesIO(); image.convert('RGB').save(b, 'JPEG', quality=quality)
    return Image.open(io.BytesIO(b.getvalue())).convert('RGB')


def shard(items, i, n):
    return items[i::n]
