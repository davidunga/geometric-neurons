import numpy as np
from typing import Sequence


_stat_full_names = ('size', 'sum', 'mean', 'median', 'var', 'std', 'mad', 'se_mean', 'se_median', 'min', 'max')
_aliases = {'size': 'n', 'mean': 'avg', 'median': 'med', 'se_mean': 'se', 'se_median': 'sm'}


class BinSpec:

    def __init__(self, spec: int | Sequence[float | int] = 10, method: str = 'u'):
        """
        Args:
            spec: either number of bins (int), or 1d array of bin edges
            method: binning method, one of {'u','p'}, 'u' = uniform spacings, 'p' = percentiles.
                ignored if bin edges are provided.
        """
        if isinstance(spec, int):
            assert method in ('u', 'p')
            self.n = spec
            self.method = method
            self._edges = None
        else:  # explicit bin edges
            self._edges = np.asarray(spec, float)
            self.method = None
            self.n = len(self._edges) - 1
            assert np.all(np.diff(self._edges) > 0), "Edge locations must be strictly increasing"

    def get_edges(self, x=None) -> np.ndarray[float]:
        return self._edges if self._edges else make_bin_edges(x, n=self.n, method=self.method)


def calc_stats(x, stats: list[str] = None, axis: int = None) -> dict[str, float]:
    if stats is None:
        stats = stat_names()
    mem = {}
    x = np.asarray(x)
    return {stat: _lazy_calc_stat(x, _tofull(stat), mem, axis=axis) for stat in stats}


def calc(x, stat: str) -> float:
    return calc_stats(x, [stat])[stat]


def make_bin_edges(x: Sequence[float | int], n: int, method: str) -> np.ndarray[float]:
    """
    Args:
        x: array to be binned
        n: number of bins
        method: 'u' = uniformly spaced, 'p' = percentiles
    """
    if method == 'u':
        bin_edges = np.linspace(np.min(x), np.max(x), n + 1)
    elif method == 'p':
        bin_edges = np.percentile(x, np.linspace(0, 100, n + 1))
    else:
        raise ValueError("Unknown binning kind")
    bin_edges[0] -= 1e-10
    bin_edges[-1] += 1e-10
    return bin_edges


def safe_digitize(x, bins: BinSpec):
    """ digitize x to indices starting at 0 """
    bin_edges = bins.get_edges(x)
    inds = np.digitize(x, bin_edges) - 1
    assert inds.min() >= 0 and inds.max() < bins.n
    return inds, bin_edges


def zscore(x: float, xs: Sequence[float], robust: bool):
    """ zscore of x (scalar) relative to xs (population) """
    loc, scale = calc_loc_and_scale(xs, robust)
    z = (x - loc) / scale
    return z, loc, scale


def calc_loc_and_scale(x: Sequence[float], robust: bool) -> tuple[float, float]:
    """ location and scale of distribution """
    loc_name = 'median' if robust else 'mean'
    scale_name, _ = get_scale_and_error_names(loc_name)
    loc, scale = calc_stats(x, [loc_name, scale_name]).values()
    return loc, scale


def stat_names(short: bool = True) -> list[str]:
    return [_toshort(name) for name in _stat_full_names] if short else list(_stat_full_names)


def get_scale_and_error_names(loc: str) -> tuple[str, str]:
    loc_full = _tofull(loc)
    assert loc_full in ('mean', 'median')
    scale, err = ('std', 'se_mean') if loc_full == 'mean' else ('mad', 'se_median')
    if loc_full != loc:
        scale, err = _toshort(scale), _toshort(err)
    return scale, err


def calc_binned_stats(x, y, stats: list[str] = None, bins: BinSpec = BinSpec(), drop_nan: bool = False):
    """
    Computes statistics for y within bins defined on x.
    Parameters:
    - x: Numeric vector.
    - y: Numeric vector, same length as x.
    - stats: names of statistics, default = all.
    - bins: Number of bins to split x into, or array of bin edges.
    - kind: Binning kind, used if bins is int:
        'u' = uniform spacing, 'p' = uniform percentiles
    Returns:
        a dict, mapping each stat name to an array of its binned values.
            also contains the x bin centers, under key 'x'.
    """

    if stats is None:
        stats = stat_names()

    bin_inds, bin_edges = safe_digitize(x, bins)
    stats_per_bin = [calc_stats(y[bin_inds == i], stats) for i in range(bins.n)]
    ret = {stat: np.array([ys[stat] for ys in stats_per_bin]) for stat in stats}
    ret['x'] = (bin_edges[1:] + bin_edges[:-1]) / 2

    return ret


class Inliers:

    _defaults = {'iqr': 1.5, 'sigmas': 3, 'percentiles': .01}

    def __init__(self, method: str, thresh=None):
        """
        method: 'iqr' = interquartile range
                'sigmas' = robust zscore
                'percentiles' = percentiles range
        thresh: if method = 'iqr', thresh is in IQR units. default = 1.5
                if method = 'sigmas', thresh is in robust-sigma units. default = 3.
                if method = 'percentiles', thresh is in fraction units. can be either [low, high] pair,
                    or a margin (scalar) to interpret as [margin, 1-margin]. default = .01.
        """
        self.method = method
        thresh = thresh if thresh is not None else self._defaults[method]
        if method == 'percentiles' and not hasattr(thresh, '__len__'):
            thresh = (thresh, 1 - thresh)
        self.thresh = thresh
        self._bounds = None

    def fit(self, x: np.ndarray):
        self._bounds = self.inlier_bounds(x)
        return self

    def filter(self, x: np.ndarray) -> np.ndarray:
        return x[self.is_inlier(x)]

    def is_inlier(self, x: np.ndarray) -> np.ndarray[bool]:
        lb, ub = self.inlier_bounds(x)
        return (x >= lb) & (x <= ub)

    def is_outlier(self, x: np.ndarray) -> np.ndarray[bool]:
        return ~self.is_inlier(x)

    def inlier_bounds(self, x) -> tuple[float, float]:
        if self._bounds:
            return self._bounds
        if self.method == 'iqr':
            assert self.thresh > 0
            q1, q3 = np.percentile(x, [25, 75])
            iqr = q3 - q1
            lb, ub = q1 - self.thresh * iqr, q3 + self.thresh * iqr
        elif self.method == 'sigmas':
            assert self.thresh > 0
            loc, scale = calc_loc_and_scale(x, robust=True)
            lb, ub = loc - self.thresh * scale, loc + self.thresh * scale
        elif self.method == 'percentiles':
            l, h = self.thresh
            assert 0 <= l < h <= 1
            lb, ub = np.percentile(x, [l * 100, h * 100])
        elif self.method == 'none':
            lb, ub = float('-inf'), float('inf')
        else:
            raise ValueError("Unknown method")
        return lb, ub


def _lazy_calc_stat(x: np.ndarray, stat: str, mem: dict[str, float], **kwargs) -> float:
    """ calc stat with memoization """

    if stat not in _stat_full_names:
        raise ValueError(f"Unknown stat {stat}")

    if stat in mem:
        pass
    elif not len(x):
        mem[stat] = 0 if stat == 'size' else np.nan
    elif stat == 'std':
        mem[stat] = np.sqrt(_lazy_calc_stat(x, 'var', mem, **kwargs))
    elif stat == 'mad':
        mem[stat] = 1.4826 * np.median(np.abs(x - _lazy_calc_stat(x, 'median', mem, **kwargs).reshape(x.shape)), axis=axis)
    elif stat == 'se_mean':
        mem[stat] = _lazy_calc_stat(x, 'std', mem, **kwargs) / np.sqrt(max(1, len(x)))
    elif stat == 'se_median':
        mem[stat] = 1.2533 * _lazy_calc_stat(x, 'se_mean', mem, **kwargs)
    else:
        mem[stat] = getattr(np, stat)(x, **kwargs)

    return mem[stat]


_short_to_full = {_aliases.get(full, full): full for full in _stat_full_names}
_full_to_short = {full: _aliases.get(full, full) for full in _stat_full_names}


def _tofull(name):
    name = _short_to_full.get(name, name)
    assert name in _full_to_short
    return name


def _toshort(name):
    name = _full_to_short.get(name, name)
    assert name in _short_to_full
    return name
