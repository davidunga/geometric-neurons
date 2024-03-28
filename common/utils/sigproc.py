import numpy as np


def reduce_rows(a, win_sz: int):
    """
    Args:
        a: 2d array
        win_sz: window size
    Returns:
        2d array of shape (a.shape[0] // win_sz, a.shape[1])
    """
    new_len = a.shape[0] // win_sz
    sample_ixs = np.arange(new_len) * win_sz
    ret = np.stack([np.mean(a[i: i + win_sz], axis=0) for i in sample_ixs], axis=0)
    return ret, sample_ixs


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
    X /= np.maximum(scale, 1e-8)
    return X


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = np.random.default_rng(1).random(size=(20, 30))
    reduce_rows(x, 3)