import numpy as np
from common.utils.conics import conic_coeffs, Conic
from common.utils.linalg import rotate_points
from scipy.optimize import minimize


class ConicParabola(Conic):

    _kind_name = 'Parabola'

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

    def default_bounds(self) -> tuple[float, float]:
        plim = self.t_to_p(2)
        return -plim, plim

    def squared_dists(self, pts, refine: bool = False) -> tuple[np.ndarray[float], np.ndarray[float]]:
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

    def _arclen_convert(self, *, t=None, s=None) -> np.ndarray[float]:
        ff = 4 * abs(self.m)
        f = 1 / ff
        f_log_ff = f * np.log(ff)

        def _t_to_s(t):
            h = .5 * t
            q = np.sqrt(f ** 2 + h ** 2)
            return h * q * ff + np.log(h + q) * f + f_log_ff

        if s is None:
            return _t_to_s(np.asarray(t))

        assert t is None
        sgn = np.sign(s)
        s = np.abs(s)
        res = minimize(fun=lambda x: np.sum((s - _t_to_s(x)) ** 2), x0=s, bounds=[(0, s_) for s_ in s])
        return sgn * res.x



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    debug_draw(ConicParabola(3, (0, 0), 0))
    plt.show()
