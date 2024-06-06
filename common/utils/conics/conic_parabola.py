import numpy as np
from common.utils.conics import conic_coeffs, Conic
from common.utils.linalg import rotate_points
from scipy.optimize import minimize


class ConicParabola(Conic):

    _kind = 'p'

    @classmethod
    def from_coeffs(cls, coeffs):
        ang = conic_coeffs.rotation_theta(coeffs, relto='y') * 180 / np.pi
        A, B, C, D, E, F = conic_coeffs.rotate(coeffs, -ang * np.pi / 180)
        if E == 0:
            raise conic_coeffs.ConicTypeError(f"Invalid value when computing parabola parameters. A={A}, E={E}")
        m, n, k = -A / E, -D / E, -F / E
        if m == 0:
            raise conic_coeffs.ConicTypeError(f"Invalid value when computing parabola parameters. A={A}, E={E}")
        x0 = -n / (2 * m)
        y0 = m * x0 ** 2 + n * x0 + k
        vertex = np.array(rotate_points(x0, y0, deg=ang))
        return cls(m, vertex, ang)

    def coeffs(self) -> tuple[float, ...]:
        x0, y0 = self.loc
        A = self.m
        D = -2 * self.m * x0
        F = self.m * x0 ** 2 + y0
        coeffs = conic_coeffs.rotate((A, 0, 0, D, -1, F), self.ang * np.pi / 180)
        return coeffs

    def eccentricity(self) -> float:
        return 1.

    def semi_latus(self) -> float:
        return 1 / abs(2 * self.m)

    def focus_dist(self) -> float:
        return 1 / abs(4 * self.m)

    def focus_pts(self) -> np.ndarray:
        return np.array(self.transform((0, self.focus_dist())), float, ndmin=2)

    def vertex_pts(self) -> np.ndarray:
        return np.array(self.loc, float, ndmin=2)

    def t_to_p(self, t):
        return t / self.semi_latus()

    def p_to_t(self, p):
        return p * self.semi_latus()

    def default_bounds(self) -> tuple[float, float]:
        return -2, 2

    def arclen(self, t=None, p=None):
        if p is not None:
            assert t is None
            t = self.p_to_t(p)
        term1 = (t / 2) * np.sqrt(1 + 4 * self.m ** 2 * t ** 2)
        term2 = (1 / (4 * self.m)) * np.arcsinh(2 * self.m * t)
        return term1 + term2

    def squared_dists(self, pts, **kwargs) -> tuple[np.ndarray[float], np.ndarray[float]]:
        xx, yy = self.inv_transform(pts)
        t = np.zeros(len(pts), float)
        squared_dists = np.zeros(len(pts), float)
        for i, (x, y) in enumerate(zip(xx, yy)):
            roots = np.roots([2 * self.m ** 2, 0, -2 * self.m * y + 1, -x])
            dists2 = [(xi - x) ** 2 + (self.m * xi ** 2 - y) ** 2 if xi.imag == 0 else np.inf for xi in roots]
            mindist_ix = np.argmin(dists2)
            t[i] = roots[mindist_ix].real
            squared_dists[i] = dists2[mindist_ix].real
        return squared_dists, t


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from common.utils.conics.conic import _test_transform
    _test_transform(ConicParabola(3, (-1, 2), 20))
    #plt.show()
