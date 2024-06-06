import numpy as np

from common.utils.typings import *


_modulos = {'r': np.pi, 'd': 180}


def _get_modulo(mod):
    return _modulos.get(mod, mod)


def _as_radians(rad=None, deg=None):
    if deg is not None:
        assert rad is None
        return np.radians(deg)
    else:
        return rad


def circdiff(t1, t2, mod: float | str = 'r'):
    """ t2 - t1 """
    m = _get_modulo(mod)
    return (t2 - t1 + m) % (2 * m) - m


def is_ccw(thetas):
    assert len(thetas) == 3
    (x1, x2, x3), (y1, y2, y3) = np.cos(thetas), np.sin(thetas)
    det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    return det > 0


def average_theta(thetas):
    s = np.sum(np.sin(thetas))
    c = np.sum(np.cos(thetas))
    avg = np.arctan2(s, c)
    return avg


def rotate_points(*args, **kwargs):
    return rotate(*args, **kwargs)


def rotate(x: np.ndarray | float, y: np.ndarray | float, rad: float = None, deg: float = None, ax=None):
    if hasattr(x, '__len__'):
        x = np.asarray(x)
        y = np.asarray(y)
    if deg is not None:
        assert rad is None
        rad = deg * np.pi / 180
    if ax is not None:
        x, y = x - ax[0], y - ax[1]
    c, s = np.cos(rad), np.sin(rad)
    x, y = c * x - s * y, s * x + c * y
    if ax is not None:
        x += ax[0]
        y += ax[1]
    return x, y


class planar:

    @staticmethod
    def transform(pts: np.ndarray, scale: float = 1., ang: float = .0, offset=.0, reflect: str = 'none'):
        A = planar.build(b=scale, ang=ang, t=offset, reflect=reflect)
        return planar.apply(A, pts)

    @staticmethod
    def _check_mtx(A):
        """ check that matrix is either 2x2, or homogeneous 3x3 """
        assert A.shape in ((2, 2), (3, 3))
        if A.shape == (3, 3):
            assert abs(A[2, 0]) < 1e-8 and abs(A[2, 1]) < 1e-8 and abs(A[2, 2] - 1) < 1e-8

    @staticmethod
    def to_homogeneous(R: NpMatrix, t=0.0) -> NpMatrix:
        planar._check_mtx(R)
        if R.shape == (3, 3):
            assert t == 0
            return R
        A = np.eye(3)
        A[:2, :2] = R
        A[:2, -1] = t
        return A

    @staticmethod
    def build(b=1.0, ang=0.0, t=0.0, reflect: str = 'none') -> NpMatrix:
        c = b * np.cos(np.radians(ang))
        s = b * np.sin(np.radians(ang))
        A = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        if reflect != 'none':
            assert reflect in ('x', 'y')
            A[int(reflect == 'y')] *= -1
        A[:2, -1] = t
        return A

    @staticmethod
    def apply(A: NpMatrix, X: NpPoints) -> NpPoints:
        planar._check_mtx(A)
        return (A @ _pad_to_homogeneous(X).T).T[:, :2]

    @staticmethod
    def apply_inv(A: NpMatrix, X: NpPoints) -> NpPoints:
        return planar.apply(np.linalg.inv(A), X)

    @staticmethod
    def decompose(A: NpMatrix, scale_metric: str = 'det'):

        planar._check_mtx(A)

        t = A[:2, -1] if A.shape == (3, 3) else np.zeros(2, float)

        U, s, Vt = np.linalg.svd(A[:2, :2])

        # ortho_score = 1 / condNumber. max = 1, min = 0
        ortho_score = s[1] / s[0] if s[1] < s[0] else s[0] / s[1]

        if scale_metric == 'avg':
            b = s.mean()
        elif scale_metric == 'det':
            b = np.sqrt(np.abs(np.linalg.det(A[:2, :2])))
        else:
            raise ValueError('Unknown scale metric')

        R = U.T @ Vt.T

        is_reflective = np.linalg.det(R) < 0

        ang1 = np.degrees(np.arctan2(-R[0, 1], R[0, 0])) % 360
        ang1 = min(ang1, 360 - ang1)

        ang2 = np.degrees(np.arctan2(R[1, 0], R[1, 1])) % 360
        ang2 = min(ang2, 360 - ang2)

        if is_reflective:
            ang = min(ang1, ang2)
        else:
            assert abs(ang1 - ang2) < 1e-2
            ang = (ang1 + ang2) / 2

        return b, ang, t, is_reflective, ortho_score


def lsqr(x, b):
    """ matrix A s.t. Ax ~ b """
    return _pad_to_homogeneous(b).T @ np.linalg.pinv(_pad_to_homogeneous(x).T)


def _pad_to_homogeneous(X):
    if X.shape[1] == 2:
        return np.c_[X, np.ones(len(X))]
    elif X.shape[1] == 3:
        return X
    else:
        ValueError()


def foo(li):

    from time import sleep
    if li[0] == 3:
        sleep(8)
    else:
        sleep(6)
    print(li)
    return [x * 2 for x in li]

def partest():
    from multiprocessing import Pool

    L1 = [1, 2, 3]
    L2 = [3, 4, 5]
    L3 = [5, 6, 7]

    with Pool(4) as pool:
        pool.map(foo, [L1, L2, L3])
        print("done")

if __name__ == "__main__":
    partest()
    #run__make_and_save()
    pass
