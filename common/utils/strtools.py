import numpy as np


def part(p, q=None) -> str:
    if q is None:
        q = len(p)
        p = int(np.sum(p))
    return f"{p}/{q} ({p/q*100:3.2f}%)"
