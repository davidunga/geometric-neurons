import numpy as np


def reduce_rows(a, win_sz: int, fn='mean'):
    # non-overlapping reduction 0-th dimension of array
    if isinstance(fn, str): fn = getattr(np, fn)
    return np.array([fn(a[i: i + win_sz], axis=0) for i in range(0, len(a) - win_sz + 1, win_sz)])

