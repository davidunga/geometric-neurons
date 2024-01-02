import numpy as np


def reduce_rows(a, win_sz: int, fn='mean'):
    # non-overlapping reduction 0-th dimension of array
    if isinstance(fn, str): fn = getattr(np, fn)
    return np.array([fn(a[i: i + win_sz], axis=0) for i in range(0, len(a) - win_sz + 1, win_sz)])


def scales_and_offsets(X, axis, kind='std'):
    if kind == 'std':
        offset = np.mean(X, axis=axis, keepdims=True)
        scale = np.std(X, axis=axis, keepdims=True)
    elif kind == 'mad':
        offset = np.median(X, axis=axis, keepdims=True)
        scale = np.median(np.abs(X - offset), axis=axis, keepdims=True)
    elif kind == 'max':
        mn = np.min(X, axis=axis, keepdims=True)
        mx = np.max(X, axis=axis, keepdims=True)
        offset = mn
        scale = mx - mn
    else:
        raise ValueError("Unknown normalization kind")
    return scale, offset


def normalize(X, axis, kind: str = 'std', inplace: bool = False):
    scale, offset = scales_and_offsets(X, axis, kind)
    if not inplace:
        X = X.copy()
    X -= offset
    X /= scale
    return X


