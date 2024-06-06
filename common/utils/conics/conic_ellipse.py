import matplotlib.pyplot as plt
import numpy as np
from common.utils.conics import conic_coeffs, Conic
from common.utils import polytools
from scipy.special import ellipeinc


class ConicEllipse(Conic):

    _kind = 'e'

    @classmethod
    def from_coeffs(cls, coeffs):

        A, B, C, D, E, F = coeffs
        d = B ** 2 - 4 * A * C
        if d >= 0:
            raise conic_coeffs.ConicTypeError(f"Invalid value when computing ellipse parameters. det={d}")

        # axes:
        M = np.array([[A, B / 2], [B / 2, C]])
        eigvals = np.linalg.eigvals(M)[:2]
        k = np.linalg.det(M) * eigvals
        if np.any(k == 0):
            raise conic_coeffs.ConicTypeError(f"Invalid value when computing ellipse axes. det={d}")
        M0 = np.array([[F, D / 2, E / 2], [D / 2, A, B / 2], [E / 2, B / 2, C]])
        v = -np.linalg.det(M0) / k
        if np.any(v < 0):
            raise conic_coeffs.ConicTypeError(f"Invalid value when computing ellipse axes. det={d}")
        a, b = np.sqrt(v)
        a, b = (a, b) if a > b else (b, a)

        # center & rotation:
        center = np.array([2 * C * D - B * E, 2 * A * E - B * D]) / d
        ang = conic_coeffs.rotation_theta(coeffs) * 180 / np.pi

        return cls((a, b), center, ang)

    def coeffs(self) -> tuple[float, ...]:
        a, b = self.m
        coeffs0 = (1 / a ** 2, 0, 1 / b ** 2, 0, 0, -1)
        coeffs = conic_coeffs.translate(conic_coeffs.rotate(coeffs0, self.ang * np.pi / 180), self.loc)
        return coeffs

    def eccentricity(self) -> float:
        a, b = self.m
        return np.sqrt(1 - (b / a) ** 2)

    def semi_latus(self) -> float:
        a, b = self.m
        return (b ** 2) / a

    def focus_dist(self) -> float:
        a, b = self.m
        return np.sqrt(a ** 2 - b ** 2)

    def focus_pts(self) -> np.ndarray:
        c = self.focus_dist()
        return np.c_[self.transform(([-c, c], [0, 0]))]

    def vertex_pts(self) -> np.ndarray:
        a = self.m[0]
        return np.c_[self.transform(([-a, a], [0, 0]))]

    def t_to_p(self, t):
        return np.asarray(t) * 180 / np.pi

    def p_to_t(self, p):
        return np.asarray(p) * np.pi / 180

    def default_bounds(self) -> tuple[float, float]:
        return -180, 180

    def arclen(self, t=None, p=None):
        if p is not None:
            assert t is None
            t = self.p_to_t(p)
        a, b = self.m
        e2 = 1 - (a / b) ** 2
        return a * ellipeinc(t, e2)

    def squared_dists(self, pts, **kwargs):
        a, b = self.m
        xx, yy = self.inv_transform(pts)
        tvecs = nearest_ellipse_tvec(a, b, np.c_[xx, yy])
        t = np.arctan2(tvecs[:, 1], tvecs[:, 0])
        dists2 = (a * np.cos(t) - xx) ** 2 + (b * np.sin(t) - yy) ** 2
        return dists2, t


def nearest_ellipse_tvec(a: float, b: float, pts: np.ndarray, n_itrs: int = 3):
    s = (a ** 2 - b ** 2) * np.array([1 / a, -1 / b])
    m = np.array([a, b])
    sgn = np.sign(pts)
    pts = np.abs(pts)
    t = np.zeros_like(pts, float) + .707
    for _ in range(n_itrs):
        e = s * np.power(t, 3)
        r = m * t - e
        q = pts - e
        rq = np.hypot(*r.T) / np.hypot(*q.T)
        t = np.clip((e + q * rq[:, None]) / m, 0, 1)
        t /= np.hypot(*t.T)[:, None]
    return t * sgn


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from common.utils.conics.conic import _test_transform
    _test_transform(ConicEllipse((2, 1), (0, 0), 20))
