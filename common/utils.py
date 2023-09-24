import numpy as np
from hashlib import sha1
import pandas as pd
from pingouin import partial_corr
from datetime import datetime
from utils.dictools import SymDict, dict_sort
import time
from inspect import getframeinfo, stack
from typing import List, Dict, Iterable, Callable, Literal
from geometrik import geometries


def hash_id(s, n=6):
    """ robust short hash """
    if isinstance(s, dict):
        s = str(dict_sort(s))
    if not isinstance(s, str):
        s = str(s)
    return sha1(s.encode('utf-8')).hexdigest()[:n]


def angdiff(a, b):
    """ angular difference between angles a and b (in radians)  """
    return np.pi - abs(np.mod(abs(a - b), 2 * np.pi) - np.pi)


def part2pcnt(p: float):
    return int(round(p * 100))


def clip_by_percentile(x, p: int):
    x = x.copy()
    th_low, th_high = np.percentile(x, [p, 100 - p])
    x[x < th_low] = th_low
    x[x > th_high] = th_high
    return x


def make_seg_pair_dict(pair_ixs, vals=None, symm=False):
    if vals is None:
        vals = list(range(len(pair_ixs)))
    keys = [tuple(seg_pair) for seg_pair in pair_ixs]
    if symm:
        return SymDict(keys, vals)
    return dict(zip(keys, vals))


def get_pair_segments(pair_ixs, seg_ix):
    """ get all pairs of segment [seg_ix] """
    ixs = np.any(pair_ixs == seg_ix, axis=1)
    pair_segs = pair_ixs[ixs].flatten()
    pair_segs = pair_segs[pair_segs != seg_ix]
    return pair_segs, ixs


def parse_geom(var_name, mode: Literal['num', 'long', 'short', 'sym', 'enum'] = 'num'):

    _geom_specs = {
        0: {'long': 'FullAffine', 'sfx': 'Aff', 'short': 'FuAff', 'sym': '**', 'enum': geometries.GEOMETRY.FULL_AFFINE},
        1: {'long': 'EquiAffine', 'sfx': 'EAf', 'short': 'EqAff', 'sym': '*', 'enum': geometries.GEOMETRY.EQUI_AFFINE},
        2: {'long': 'Euclidean',  'sfx': 'Euc', 'short': 'Eucld', 'sym': ' ', 'enum': geometries.GEOMETRY.EUCLIDEAN},
    }

    found_nums = []
    for g in _geom_specs:
        found_nums.append(f'k{g}' in var_name or
                          f's{g}' in var_name or
                          var_name.lower().endswith(_geom_specs[g]['sfx'].lower()) or
                          _geom_specs[g]['long'] in var_name or
                          _geom_specs[g]['short'] in var_name)

    assert sum(found_nums) < 2
    num = found_nums.index(True) if sum(found_nums) == 1 else 2

    if mode == 'num':
        return num

    return _geom_specs[num][mode]


def unpack_segs(segs: List[Dict], filt: (Iterable, Callable[[str], bool]) = None) -> Dict[str, np.ndarray]:
    """
    Unpack segment variables to a dict of variables
    Args:
        segs: segment structure
        filt: filter for variables names. either a list of names, a boolean lambda, or None
    """
    assert isinstance(filt, (Iterable, Callable)) or (filt is None)
    if isinstance(filt, Iterable):
        var_names = filt
    else:
        var_names = set(segs[0].keys()).difference(('trial_ix', 'ixs'))
    if isinstance(filt, Callable):
        var_names = [var_name for var_name in var_names if filt(var_name)]
    return {var_name: np.array([seg[var_name] for seg in segs])
            for var_name in var_names}


def time_str():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def apply_procrustes(X, procrustes, pair_ix):
    return apply_affine_tform(X, procrustes['A'][pair_ix].reshape((2, 3)))


def apply_affine_tform(Y, A):
    return (A[:, :2] @ Y.T).T + A[:, 2]


def procrustes_metric(X, Y):
    return np.sum((Y-X) ** 2) / np.sum((X - X.mean(axis=0)) ** 2)


def _make_dataframe_for_corr(x, y, zs):

    zs = [] if zs is None else zs
    if len(zs) and not hasattr(zs[0], '__len__'):
        zs = [zs]

    X = np.zeros((len(x.squeeze()), 2 + len(zs)))
    X[:, 0] = x.squeeze()
    X[:, 1] = y.squeeze()

    z_cols = []
    for j in range(len(zs)):
        z = zs[j].squeeze()
        if np.all(np.isclose(X[:, 0], z)) or np.all(np.isclose(X[:, 1], z)):
            # avoid a bug where if z == x or y, it doesn't affect the correlation
            z[0] *= 0.95
        X[:, j + 2] = z

        z_cols.append(f'z{j + 1}')

    df = pd.DataFrame(data=X, columns=['x', 'y'] + z_cols)

    return df, z_cols


def partial_spearman_corr(x, y, zs=None):
    df, z_cols = _make_dataframe_for_corr(x, y, zs)
    result = partial_corr(data=df, x='x', y='y', covar=z_cols, method='spearman')
    rho = float(result['r'])
    pval = float(result['p-val'])
    return rho, pval


def uniform_digitize(x, bins=10, method='p'):
    """
        digitize x using [bins] uniform bins, either py percentile ('p') or value ('v')
        bin indices are shifted to start from 0
    """
    if method == 'p':
        bin_edges = np.percentile(x, np.linspace(0, 100, bins + 1))
    elif method == 'v':
        bin_edges = np.linspace(min(x), max(x), bins + 1)
    else:
        raise ValueError("Unknown binning method")
    bin_edges[-1] += 1e-8
    x_binned = np.digitize(x, bin_edges)
    assert x_binned.min() == 1
    assert x_binned.max() == len(bin_edges) - 1
    x_binned -= 1
    return x_binned, bin_edges


def bin_stats(x, bin_ixs):
    """
    Calc statistics of x for each bin defined by bin_ixs
    :param x: np array
    :param bin_ixs: array same size as x. bin_ixs[i] is the bin that x[i] belongs to
    """
    num_bins = np.max(bin_ixs) + 1
    stats = {k: np.zeros(num_bins) + np.nan for k in ('mean', 'median', 'sd', 'min', 'max', 'count', 'sem')}
    for bin_ix in range(num_bins):
        ii = bin_ixs == bin_ix
        if not np.any(ii):
            continue
        xx = x[ii]
        stats['mean'][bin_ix] = np.mean(xx)
        stats['median'][bin_ix] = np.median(xx)
        stats['sd'][bin_ix] = np.std(xx)
        stats['min'][bin_ix] = np.min(xx)
        stats['max'][bin_ix] = np.max(xx)
        stats['count'][bin_ix] = len(xx)
        stats['sem'][bin_ix] = stats['sd'][bin_ix] / np.sqrt(len(xx))
    return stats


class ExecTimer:
    """ A very simple execution timer.
        Each call, reports the time since its last call, and lines where calle occurred
        Example:
            timer = ExecTimer()
            # some lines of code ...
            timer()
            # lines of code ...
            timer()
            # ...
    """

    def __init__(self, active=True):
        self.active = active
        self.times = []
        self.callers = []
        self._toc()

    def _toc(self):
        if not self.active:
            return
        self.times.append(time.time())
        self.callers.append(getframeinfo(stack()[2][0]))
        self.report(only_last=True)

    def __call__(self, *args, **kwargs):
        self._toc()

    def report(self, only_last=False):
        for i in range(len(self.times) - 1 if only_last else 0, len(self.times)):
            if i == 0:
                print(f"Timing start at {self.callers[0].function} {self.callers[0].lineno}")
            else:
                delta = self.times[i] - self.times[i - 1]
                total = self.times[i] - self.times[0]
                print("{:s}[{:d}] - {:s}[{:d}] : Duration={:5.1f}s   Total={:5.1f}s".format(
                    self.callers[i - 1].function, self.callers[i - 1].lineno,
                    self.callers[i].function, self.callers[i].lineno, delta, total))
