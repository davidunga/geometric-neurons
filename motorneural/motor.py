"""
Motor and kinematic data processing + containers
"""
import geometrik as gk
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
from . npdataframe import NpDataFrame
from . typetools import *


# ----------------------------------


def calc_stat(stat: str, x, units: str = 'none', **kwargs):
    assert units in ('none', 'deg', 'rad')
    stat = {'med': 'median', 'avg': 'mean'}.get(stat.lower(), stat.lower())
    stat_fcn = getattr(np, stat)
    if units in ('deg', 'rag') and stat not in ('min', 'max'):
        if units == 'deg':
            x = np.radians(x)
        ret = np.arctan2(stat_fcn(np.sin(x), **kwargs), stat_fcn(np.cos(x), **kwargs)) % (2 * np.pi)
        if units == 'deg':
            ret = np.degrees(ret)
    else:
        ret = stat_fcn(x, **kwargs)
    return ret


class KinData(NpDataFrame):

    @classmethod
    def from_dict(cls, kin: dict[str, Sequence[float]], t: Sequence[float]):
        df = pd.DataFrame.from_dict(kin)
        return cls(df, t=t, aliases={'X': ['PosX', 'PosY']})

    @property
    def num_vars(self):
        return self.shape[1]

    def __str__(self):
        return f"KinData: {self.num_vars} variables, {len(self)} bins"

    def __repr__(self):
        return str(self)

#
# class KinData2(UniformlySampled):
#     """ uniformly sampled kinematic data """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._stat_fcns = {'Med': np.median, 'Avg': np.mean, 'Std': np.std}
#         self._cache = {}
#
#     def get_slice(self, slc: slice):
#         return KinData(self._fs, self._t0, keys=self._keys, vals=self._vals[:, slc])
#
#     def __getitem__(self, item):
#         if item[:3] not in self._stat_fcns:
#             return super().__getitem__(item)
#         if item not in self._cache:
#             stat_fcn = self._stat_fcns[item[:3]]
#             val = super().__getitem__(item[3:])
#             if item.endswith('Ang'):
#                 val = np.radians(val)
#                 radians_stat = np.arctan2(stat_fcn(np.sin(val)), stat_fcn(np.cos(val)))
#                 self._cache[item] = np.degrees(radians_stat) % 360
#             else:
#                 self._cache[item] = stat_fcn(val)
#         return self._cache[item]
#
#     def __getattr__(self, item):
#         return self.__getitem__(item)
#
#     def __str__(self):
#         return "KinData:" + super()._base_str()
#
#     def __repr__(self):
#         return str(self)

# ----------------------------------

def cart2polar(xy):
    rho = np.linalg.norm(xy, axis=1)
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    return rho, theta


def numdrv(X: NpPoints, t: NpVec, n=1) -> list[NpPoints]:
    """
    Simple numeric derivative(s) of X wrt t.
    Args:
        X: np array, points x dims
        t: 1d np array, same length as X.
        n: order of derivative.
    Returns:
        a list of np arrays, same size as X: [d/dt X, (d/dt)^2 X, .., (d/dt)^n X],
    """

    shape = X.shape
    if X.ndim == 1:
        X = X[:, None]

    drvs = []
    for _ in range(n):
        X = np.copy(X)
        for j in range(X.shape[1]):
            X[:, j] = np.gradient(X[:, j], t, edge_order=1)
        drvs.append(X.reshape(shape))
    return drvs


def kinematics(X: NpPoints, t: NpVec, dst_t: NpVec, dx: float = None, smooth_dur: float = 0):

    # ----
    # defaults

    if dx is None:
        dx = np.median(np.abs(np.diff(X, axis=0)))

    # ----
    # t0 and fs of dst_t:

    fs = (len(dst_t) - 1) / (dst_t[-1] - dst_t[0])
    t0 = dst_t[0]
    deviation_from_uniform = np.max(np.abs(dst_t - (t0 + np.arange(len(dst_t)) / fs)))
    max_deviation = .01  # in dt units
    assert deviation_from_uniform * fs < max_deviation
    assert dst_t[-1] <= t[-1]

    # ----

    crv = gk.spcurve_factory.make_numeric_curve(X=X, t=t, dx=dx, dst_t=dst_t)

    if smooth_dur > 0:
        smooth_sigma = smooth_dur * fs
        X_smooth = gaussian_filter1d(crv.pos(dst_t), sigma=smooth_sigma, axis=0, mode='mirror')
        crv = gk.spcurve_factory.NumericCurve(X_smooth, dst_t)

    invars = gk.invariants.geometric_invariants(crv)

    t = dst_t
    PosX, PosY = crv.pos(t).T
    spd, spd_ang = cart2polar(crv.vel(t))
    spd, spd_ang = cart2polar(crv.vel(t))
    acc, acc_ang = cart2polar(crv.acc(t))

    kin = {
           'PosX': PosX,
           'PosY': PosY,

           'EuSpdAng': np.degrees(spd_ang) % 360,
           'EuAcc': acc,
           'EuAccAng': np.degrees(acc_ang) % 360,

           'AfSpd': numdrv(invars['s0'], t)[0],
           'SaSpd': numdrv(invars['s1'], t)[0],
           'EuSpd': spd,

           'AfCrv': np.abs(invars['k0']),
           'SaCrv': np.abs(invars['k1']),
           'EuCrv': np.abs(invars['k2'])}

    return KinData.from_dict(kin, t)
