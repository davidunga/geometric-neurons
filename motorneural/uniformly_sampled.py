from dataclasses import dataclass
from src.motorneural.typetools import *
import numpy as np


# ----------------------------------


def snap_to_uniform_sampling(fs: float, tm: Vec[float], tol=1e-6) -> NpVec:
    """ Adjust time values [tm] to sampling rate [fs], don't allow adjustments greater than [tol] / [fs] """
    tm = np.array(tm)
    dt = 1 / fs
    result = np.round(tm / dt) * dt
    assert np.max(np.abs(tm - result)) < dt * tol
    return result


def make_time_bins(fs: float, tlims: Pair, tol: float = 1e-6) -> NpVec:
    """ Make time-bin edges with sampling rate [fs] and time limits [tlims] """
    tlims = snap_to_uniform_sampling(fs, tlims, tol)
    nbins = int(.5 + np.diff(tlims) * fs)
    t0 = tlims[0]
    dt = 1 / fs
    bin_edges = t0 + np.arange(nbins + 1) * dt
    bin_edges[0] -= .5 * tol * dt
    bin_edges[-1] += .5 * tol * dt
    return bin_edges


# ----------------------------------

#
# @dataclass
# class UniformTime:
#
#     fs: float
#     t0: float
#     n: int
#
#     _dt: float = None
#
#     @classmethod
#     def from_time(cls, t: Vec[float]):
#         fs = (len(t) - 1) / (t[-1] - t[0])
#         t0 = np.ceil(t[0] * fs) / fs
#         n = int((t[-1] - t0) * fs)
#         return cls(fs=fs, t0=t0, n=n)
#
#     @property
#     def t(self) -> NpVec:
#         return self.t0 + np.arange(self.n) * self.dt
#
#     @property
#     def dt(self) -> float:
#         if self._dt is None:
#             self._dt = 1 / self.fs
#         return self._dt
#
#     @property
#     def tlims(self) -> Tuple[float, float]:
#         return self.t0, self.t0 + (self.n - 1) * self.dt


@dataclass
class UniformlySampled:
    """
    Container for uniformly sampled data
    """

    def __init__(self, fs: float, t0: float, *args, **kwargs):
        """
        UniformlySampled(fs: float, t0: float, d: [str, NpVec])
        UniformlySampled(fs: float, t0: float, keys: NpVec[str], vals: NpMatrix[float])
        Args:
            fs: Sampling rate [Hz]
            t0: Time offset

            d: dict of variables (sampled @ fs)

            vals: 2d numpy array such that vals[i, :] is the i-th variable's values
            keys: keys[i] is the name of the i-th variable
        """

        if len(args) == 0:
            assert set(kwargs.keys()) == {"keys", "vals"}
            keys = kwargs['keys']
            vals = np.array(kwargs['vals'])
        elif len(args) == 1:
            assert len(kwargs) == 0
            vals = np.concatenate([v[:, None] if v.ndim == 1 else v for v in args[0].values()], axis=1).T
            keys = np.concatenate([np.tile(k, 1 if v.ndim == 1 else v.shape[1]) for k, v in args[0].items()])
        else:
            raise ValueError

        assert len(keys) == vals.shape[0]
        self._fs = fs
        self._t0 = t0
        self._t = self._t0 + np.arange(vals.shape[1]) / self._fs
        self._keys = np.array(keys)
        self._vals = vals

    @classmethod
    def from_json(cls, obj):
        return UniformlySampled(fs=obj["fs"], t0=obj["t0"], keys=obj["keys"], vals=obj["vals"])

    def index(self, tms: NpVec) -> NpVec[float]:
        """ Index of time values """
        return np.floor((tms - self.t[0]) * self.fs).astype(int)

    def to_json(self):
        return {
            "fs": self._fs,
            "t0": self._t0,
            "keys": self._keys.tolist(),
            "vals": [v.tolist() for v in self._vals]
        }

    def _as_array(self) -> NpMatrix[float]:
        """ Returns values array arr, such that arr[i,:] is the i-th variable """
        return self._vals

    def _set_array(self, arr: NpMatrix[float]):
        """ Sets values array, with array of the same size """
        assert arr.shape == self._vals.shape
        self._vals = arr

    @property
    def t(self) -> NpVec:
        return self._t

    @property
    def fs(self) -> float:
        return self._fs

    @property
    def num_samples(self) -> int:
        return len(self.t)

    def get_slice(self, slc: slice):
        return UniformlySampled(self._fs, self._t0, keys=self._keys, vals=self._vals[:, slc])

    def __getitem__(self, item: str) -> NDArray:
        assert item in self._keys
        return self._vals[self._keys == item, :].T.squeeze()

    def __getattr__(self, item):
        assert item in self._keys
        return self._vals[self._keys == item, :].T.squeeze()

