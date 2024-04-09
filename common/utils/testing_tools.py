import numpy as np

from common.utils.typings import *


def random_planar_mtx(seed: int = 0,
                      det: float | tuple[float, float] = None,
                      dim: int = 2,
                      ortho: bool = False) -> NpMatrix:

    rng = np.random.default_rng(seed)
    rand_sigma = 5
    default_det_range = (.1, 5.)

    if det is None:
        det = default_det_range

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


def random_planar_mtx_by_kind(seed: int, kind: str, det_range: tuple[float, float] = None):
    match kind:
        case 'affine':
            ortho = False
        case 'ortho':
            ortho = True
        case 'rigid':
            ortho = True
            det_range = 1.0
        case _:
            raise ValueError("Unknown kind")
    return random_planar_mtx(seed=seed, dim=3, det=det_range, ortho=ortho)


class shapesbank:

    @staticmethod
    def parabola(t0: float = -1, t1: float = 1, n: int = 50):
        t = np.linspace(t0, t1, n)
        return np.c_[t, t ** 2]

    @staticmethod
    def ellipse(a: float = 1., b: float = 1., n: int = 50, ang_min: float = 0., ang_max: float = 359.):
        t = np.radians(np.linspace(ang_min, ang_max, n))
        return np.c_[a * np.cos(t), b * np.sin(t)]
