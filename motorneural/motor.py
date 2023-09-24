"""
Motor and kinematic data processing + containers
"""

from src.motorneural.typetools import *
from src.motorneural.uniformly_sampled import UniformlySampled
import geometrik as gk

# ----------------------------------


class KinData(UniformlySampled):
    """ uniformly sampled kinematic data """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# ----------------------------------


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


def kinematics(X: NpPoints, t: NpVec, dst_t: NpVec, dx: float = None):

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

    # ----

    crv = gk.spcurve_factory.make_numeric_curve(X=X, t=t, dx=dx, dst_t=dst_t)
    invars = gk.invariants.geometric_invariants(crv)

    t = dst_t
    vel = crv.vel(t)
    acc = crv.acc(t)

    kin = {'X': crv.pos(),
           'velx': vel[:, 0],
           'vely': vel[:, 1],
           'accx': acc[:, 0],
           'accy': acc[:, 1],
           'spd2': np.linalg.norm(vel, axis=1),
           'acc2': np.linalg.norm(acc, axis=1),
           'spd0': numdrv(invars['s0'], t)[0],
           'spd1': numdrv(invars['s1'], t)[0],
           'crv0': invars['k0'],
           'crv1': invars['k1'],
           'crv2': invars['k2'],
           }

    return KinData(fs, t0, kin)
