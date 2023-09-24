import numpy as np
from common.type_utils import *


def random_planar_mtx(seed: int = 0,
                      det: (float, tuple[float, float]) = (.1, 5.),
                      dim: int = 2,
                      ortho: bool = False) -> NpMatrix:

    rng = np.random.default_rng(seed)
    rand_sigma = 5

    if not isinstance(det, float):
        assert len(det) == 2
        det = det[0] + rng.random() * (det[1] - det[0])

    if ortho:
        ang = rng.random() * 2 * np.pi
        cos, sin = np.cos(ang), np.sin(ang)
        R = np.sqrt(det) * np.array([[cos, -sin], [sin, cos]])
    else:
        R = rand_sigma * rng.standard_normal((2, 2))
        R[1, 1] = (det + R[0, 1] * R[1, 0]) / R[0, 0]

    if dim == 2:
        return R

    assert dim == 3
    A = np.eye(3)
    A[:2, :2] = R
    A[:2, -1] = rand_sigma * rng.standard_normal(2)
    return A


class shapesbank:

    @staticmethod
    def parabola(t0: float = -1, t1: float = 1, n: int = 50):
        t = np.linspace(t0, t1, n)
        return np.c_[t, t ** 2]

    @staticmethod
    def ellipse(a: float = 1., b: float = 1., n: int = 50, ang_range: tuple[float] = (0., 359.)):
        t = np.radians(np.linspace(ang_range[0], ang_range[1], n))
        return np.c_[a * np.cos(t), b * np.sin(t)]
