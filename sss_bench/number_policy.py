"""Deterministic numeric answer parser for SSS-Bench (rules in rubric/GRADING.md)."""
import json, math, re


def number(v):
    """-> float, or None when no single number can be read from the answer"""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, dict):
        for k in ['time_s', 'answer_number', 'answer', 'value']:
            if k in v and v[k] is not None:
                return number(v[k])
        return None
    s = str(v).strip()
    try:
        z = json.loads(s)
        if isinstance(z, (dict, int, float)):
            return number(z)
    except Exception:
        pass
    if re.search(r'\b(?:not|never|no)\b', s, re.I):
        return None
    ts = re.findall(r'(?<!\d)(?:(\d+):)?(\d{1,2}):(\d{2}(?:\.\d+)?)(?!\d)', s)
    if len(ts) == 1:
        hh, mm, ss = ts[0]
        return int(hh or 0) * 3600 + int(mm) * 60 + float(ss)
    xs = re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])', s)
    return float(xs[0]) if len(xs) == 1 else None
