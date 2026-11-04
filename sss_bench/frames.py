"""1 fps evidence-frame sampler used by the SSS-Bench judge.

For integer second t (original-video clock, relative to the first video frame PTS)
the sampled frame is the last decoded RGB frame whose PTS <= t. No future frame is used.
"""
import io
from fractions import Fraction

import av

THUMB = (448, 448)


def sample(video_path, seconds):
    """Return {t: (pts_seconds, PIL.Image)} for the requested integer seconds."""
    wanted = sorted(set(int(t) for t in seconds))
    out, prev, origin, i = {}, None, None, 0
    with av.open(str(video_path)) as container:
        st = container.streams.video[0]
        for frame in container.decode(st):
            if frame.pts is None:
                continue
            ts = Fraction(frame.pts) * st.time_base
            origin = ts if origin is None else origin
            while prev is not None and i < len(wanted) and wanted[i] < ts - origin:
                out[wanted[i]] = prev; i += 1
            if i == len(wanted):
                break
            prev = (float(ts - origin), frame.to_image())
        while prev is not None and i < len(wanted) and wanted[i] - prev[0] < 1:
            # final tick: hold the last real frame (short audio tail after the last video PTS)
            out[wanted[i]] = prev; i += 1
    if i != len(wanted):
        raise ValueError(f'{video_path}: cannot sample seconds {wanted[i:]}')
    return out


def png_bytes(im):
    im = im.convert('RGB').copy()
    im.thumbnail(THUMB)
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    return buf.getvalue()
