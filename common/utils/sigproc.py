import numpy as np
from common.utils import stats
from scipy.signal import find_peaks, peak_prominences
import matplotlib.pyplot as plt
from dataclasses import dataclass


class ExtremaDetector:

    def __init__(self, r: float, rheight: float = .1, rprom: float = .5,
                 nmax: int = None, width_rheight: float = .95):
        self.r = r
        self.rheight = rheight
        self.rprom = rprom
        self.nmax = nmax
        self.width_rheight = width_rheight
        self._inliers = stats.Inliers('percentiles', thresh=.05)

    def maximas(self, v, asmask: bool = False, **kwargs):
        inds = self._find_peaks(v, **kwargs)[0]
        return _to_mask(inds, len(v)) if asmask else inds

    def peaks(self, v, **kwargs) -> list[tuple[int, int, int]]:
        inds, props = self._find_peaks(v, **kwargs)
        return [(ind, int(round(l)), int(round(r)))
                for ind, l, r in zip(inds, props['left_ips'], props['right_ips'])]

    def extremas(self, v, **kwargs):
        ret = np.zeros_like(v, int)
        ret[self.maximas(v, **kwargs)] = 1
        ret[self.maximas(-v, **kwargs)] = -1
        return ret

    def _find_peaks(self, v, show: bool = False):

        mn, mx = self._inliers.inlier_bounds(v)
        height = mn + (mx - mn) * self.rheight
        prominence = (mx - mn) * self.rprom

        kws = {'distance': self.r, 'prominence': prominence,
               'height': height, 'width': 1, 'rel_height': self.width_rheight}

        inds, props = find_peaks(v, **kws)

        if self.nmax and len(inds) > self.nmax:
            ii = props['peak_heights'] >= np.sort(props['peak_heights'])[::-1][self.nmax-1] - 1e-10
            inds = inds[ii]
            props = {prop: vals[ii] for prop, vals in props.items()}

        # -----
        if show:
            plt.figure()
            plt.plot(v, 'k')
            plt.plot(inds, v[inds], 'ro')
            plt.plot(props['left_bases'], v[props['left_bases']], '.m')
            plt.plot(props['right_bases'], v[props['right_bases']], '.m')
            plt.plot(props['left_ips'], props['width_heights'], '.c')
            plt.plot(props['right_ips'], props['width_heights'], '.c')
            plt.plot([0, len(v) - 1], [height, height], 'r')
            plt.plot([0, len(v) - 1], [mn, mn], 'y')
            plt.plot([0, len(v) - 1], [mx, mx], 'g')
            for ind, height, prom in zip(inds, props['peak_heights'], props['prominences']):
                prom_p = prom / kws['prominence']
                height_p = height / kws['height']
                plt.annotate(f'P:{prom:2.1f} [{prom_p:2.1%}]\nH:{height:2.1f} [{height_p:2.1%}]',
                             (ind, height), textcoords="offset points", xytext=(0, 10), ha='center')

            plt.show()
        # -----

        return inds, props


#
#
# def first_extermas(y, **kwargs) -> np.ndarray[int]:
#     ret = np.zeros_like(y, int)
#     ret[first_maximas(y, **kwargs)] = 1
#     ret[first_maximas(-y, **kwargs)] = -1
#     return ret
#
#
# def first_maximas(y, x=None, n: int = 8, r: float = None,
#                   relmin: float = .5) -> list[int]:
#     """
#     Args:
#         y: numeric array
#         x: numeric array same size as y, default = indexes
#         n: max number of maximas
#         r: min distance between maximas, in units of x. default = len(y) / (10n)
#         relmin: stop if maxima is lower than relmin * first_maxima
#     Returns:
#         indices of maximas
#     """
#     if r is None:
#         r = len(y) / (10 * n)
#     if x is None:
#         x = np.arange(len(y))
#     assert len(x) == len(y)
#     y = y - y.min()
#     min_val = None
#     ixs = []
#     for _ in range(n):
#         i = np.argmax(y)
#         if min_val is None:
#             min_val = y[i] * relmin
#         if y[i] < min_val:
#             break
#         if np.any(y[max(0, i-1):min(len(y), i+2)] < 0):
#             break
#         y[np.abs(x - x[i]) <= r] = -1.
#         ixs.append(int(i))
#     return ixs


def nonoverlap_reduce(a: np.ndarray, win_sz: int, axis: int, reduce: str = 'mean'):
    """
    reduce array by aggregation within non-overlapping windows
    Args:
        a: nd array
        win_sz: window size
        axis: axis to reduce along
        reduce: reduce method, e.g., mean/median/all/any,..
    Returns:
        - array with:   shape[i] == a.shape[i] if i != axis, and
                        shape[axis] == a.shape[axis] // win_sz
        - indices that were sampled along axis
    """

    reduce_func = getattr(np, reduce)
    a = np.swapaxes(a, 0, axis)
    new_len = a.shape[0] // win_sz
    sample_ixs = np.arange(new_len) * win_sz
    a = np.stack([reduce_func(a[i: i + win_sz], axis=0) for i in sample_ixs], axis=0)
    a = np.swapaxes(a, 0, axis)
    return a, sample_ixs


def loc_and_scale(X, axis: int = None, kind: str = 'std', nnz_scale: bool = True):
    if kind == 'std':
        loc = np.mean(X, axis=axis, keepdims=True)
        scale = np.std(X, axis=axis, keepdims=True)
    elif kind == 'mad':
        loc = np.median(X, axis=axis, keepdims=True)
        scale = np.median(np.abs(X - loc), axis=axis, keepdims=True)
    elif kind == 'max':
        mn = np.min(X, axis=axis, keepdims=True)
        mx = np.max(X, axis=axis, keepdims=True)
        loc = mn
        scale = mx - mn
    else:
        raise ValueError("Unknown normalization kind")
    if nnz_scale:
        scale = np.maximum(scale, np.finfo(float).eps)
    return loc, scale


def normalize(X, axis=None, kind: str = 'std', inplace: bool = False):
    if not inplace:
        X = X.copy()
    if kind != 'none':
        loc, scale = loc_and_scale(X, axis, kind)
        X -= loc
        X /= np.maximum(scale, np.finfo(float).eps)
    return X


def _to_mask(inds, shape, dtype=bool):
    mask = np.zeros(shape, dtype)
    if isinstance(inds, tuple):
        for inds_ in inds:
            mask[inds_] = 1
    else:
        mask[inds] = 1
    return mask
