"""1 Hz causal frame sampling on the original-video clock (same rule as the evaluator's evidence frames)."""
import math
from fractions import Fraction


def iter_ticks(path, start_s, end_s):
    """yield (tick, pts_seconds, PIL image) for integer ticks ceil(start_s)..floor(end_s): the last decoded frame
    whose PTS (relative to the first video frame) is <= tick; at the final tick the last real frame is held if the
    container ends within one second (short audio tail)."""
    import av
    lo, hi = math.ceil(start_s), math.floor(end_s)
    k, prev = lo, None
    with av.open(str(path)) as c:
        st = c.streams.video[0]; st.thread_type = 'AUTO'
        origin = None
        for fr in c.decode(st):
            if fr.pts is None:
                continue
            ts = Fraction(fr.pts) * st.time_base
            origin = ts if origin is None else origin
            while prev is not None and k <= hi and k < ts - origin:
                yield k, prev[0], prev[1]; k += 1
            if k > hi:
                return
            prev = (float(ts - origin), fr.to_image())
        end = float(Fraction(c.duration, 1000000) - origin) if c.duration is not None and origin is not None else None
    while prev is not None and k <= hi and k - prev[0] < 1 and (end is None or k <= end + 1e-6):
        yield k, prev[0], prev[1]; k += 1
    if k <= hi:
        raise ValueError(f'{path}: video ends before tick {k} (needed up to {hi})')
