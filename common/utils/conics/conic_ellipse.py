import matplotlib.pyplot as plt
import numpy as np
from common.utils.conics import conic_coeffs, Conic
from scipy.spatial.distance import cdist
from common.utils import polytools
from scipy.interpolate import interp1d



class ConicEllipse(Conic):

    _kind_name = 'Ellipse'

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

    def default_bounds(self) -> tuple[float, float]:
        plim = self.t_to_p(np.pi)
        return -plim, plim

    def squared_dists(self, pts, refine: bool = False):
        tt = np.linspace(0, 2 * np.pi, 180)
        ellipse_pts = self.parametric_pts(t=tt)
        squared_dists = cdist(pts, ellipse_pts, 'sqeuclidean')
        mindist_ixs = np.argmin(squared_dists, axis=1)
        squared_dists = squared_dists[np.arange(len(pts)), mindist_ixs]
        t = tt[mindist_ixs]
        if refine:
            dthetas = np.linspace(-1, 1, 45) * np.pi / len(tt)
            for i, (ti, pti) in enumerate(zip(t, pts)):
                ellipse_pts = self.parametric_pts(t=ti + dthetas)
                d = np.sum((pti - ellipse_pts) ** 2, axis=1)
                mindist_ix = np.argmin(d)
                squared_dists[i] = d[mindist_ix]
                t[i] += dthetas[mindist_ix]
        return squared_dists, t

    def _arclen_convert(self, *, t=None, s=None):
        tt = np.linspace(0, 2 * np.pi, 180)
        a, b = self.m
        s_tt = polytools.arclen(np.c_[a * np.cos(tt), b * np.sin(tt)])
        if t is not None:
            assert s is None
            return np.sign(t) * interp1d(tt, s_tt)(np.abs(t))
        else:
            return np.sign(s) * interp1d(s_tt, tt)(np.abs(s))



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    debug_draw(ConicEllipse((2, 1), (0, 0), 20))
    plt.show()
