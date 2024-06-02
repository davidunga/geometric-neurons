import numpy as np
from common.utils.conics import conic_coeffs
from common.utils.conics.api.conics_api import conic_api
from common.utils.linalg import rotate_points
from scipy.spatial.distance import cdist
from common.utils import strtools
import matplotlib.pyplot as plt
from common.utils import polytools
from scipy.interpolate import interp1d
from scipy.optimize import minimize


class ellipse_api(conic_api):

    @staticmethod
    def kind():
        return 'e'

    @staticmethod
    def kind_name():
        return 'Ellipse'

    @staticmethod
    def str(m, center, ang):
        s = strtools.to_str((m, center, ang), f='2.2')
        return f'{ellipse_api.kind_name()} {s} e={ellipse_api.eccentricity(m):2.2f}'

    @staticmethod
    def parameters(coeffs):

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

        return (a, b), center, ang

    @staticmethod
    def focus_pts(m, center, ang):
        c = m[0] - ellipse_api.focus_to_vertex_dist(m)
        x, y = rotate_points([-c, c], [0, 0], deg=ang)
        return np.c_[x + center[0], y + center[1]]

    @staticmethod
    def vertex_pts(m, center, ang):
        x, y = rotate_points([-m[0], m[0]], [0, 0], deg=ang)
        return np.c_[x + center[0], y + center[1]]

    @staticmethod
    def focus_to_vertex_dist(m):
        a, b = m
        return a - np.sqrt(a ** 2 - b ** 2)

    @staticmethod
    def approx_dist2(m, center, ang, pts, refine: bool = False):
        return ellipse_api._calc_dists_to_pts(m, center, ang, pts, refine)[1]

    @staticmethod
    def nearest_t(m, center, ang, pts, refine: bool = False):
        return ellipse_api._calc_dists_to_pts(m, center, ang, pts, refine)[0]

    @staticmethod
    def nearest_p(m, center, ang, pts):
        return ellipse_api.t_to_p(m, ellipse_api.nearest_t(m, center, ang, pts))

    @staticmethod
    def nearest_contour_pt(m, center, ang, pts, refine: bool = False):
        t = ellipse_api.nearest_t(m, center, ang, pts, refine=refine)
        return ellipse_api.parametric_pts(m, center, ang, t=t)

    @staticmethod
    def parametric_pts(m, center, ang, p=None, t=None):
        if p is not None:
            assert t is None
            t = ellipse_api.p_to_t(m, p)
        x, y = rotate_points(m[0] * np.cos(t), m[1] * np.sin(t), deg=ang)
        return np.c_[x + center[0], y + center[1]]

    @staticmethod
    def coefficients(m, center, ang):
        a, b = m
        coeffs0 = (1 / a ** 2, 0, 1 / b ** 2, 0, 0, -1)
        coeffs = conic_coeffs.translate(conic_coeffs.rotate(coeffs0, ang * np.pi / 180), center)
        return coeffs

    @staticmethod
    def curvature_at_vertex(m):
        a, b = m
        ka = (b ** 2) / (a ** 3)
        kb = (a ** 2) / (b ** 3)
        return ka, kb

    @staticmethod
    def semi_latus(m) -> float:
        a, b = m
        return (b ** 2) / a

    @staticmethod
    def eccentricity(m) -> float:
        a, b = m
        return np.sqrt(1 - (b / a) ** 2)

    @staticmethod
    def radii_ratio_from_eccentricity(e: float) -> float:
        """ b/a ratio """
        return np.sqrt(1 - e ** 2)

    @staticmethod
    def draw(m, center, ang, bounds=None, *args, **kwargs):

        if bounds is None:
            bounds = ellipse_api.t_to_p(m, [-np.pi, np.pi])
        contour_pts = ellipse_api.parametric_pts(m, center, ang, p=np.linspace(*bounds, 100))

        a, b = m
        focus_pts = ellipse_api.focus_pts((a, b), center, ang)

        plt.figure(figsize=(8, 8))
        lines = plt.plot(*contour_pts.T)
        plt.plot(*contour_pts[0], 'o', color=lines[0].get_color())

        major_axis = np.c_[rotate_points([-a, a], [0, 0], deg=ang)] + center
        minor_axis = np.c_[rotate_points([0, 0], [-b, b], deg=ang)] + center
        plt.plot(*major_axis.T, 'r--', label='Major Axis')
        plt.plot(*minor_axis.T, 'g--', label='Minor Axis')

        plt.plot(*center.squeeze(), 'bo', label='Center')

        d = ellipse_api.focus_to_vertex_dist(m)
        l = ellipse_api.semi_latus(m)
        for i, (fx, fy) in enumerate(focus_pts):
            plt.plot(fx, fy, 'mo', label='Focus' if i == 0 else None)
            c = a - d if i else -(a - d)
            x, y = rotate_points([c, c], [-l, l], deg=ang)
            plt.plot(x + center[0], y + center[1], ':y', label='Latus' if i == 0 else None)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(ellipse_api.str(m, center, ang))
        plt.legend()
        plt.axis('equal')
        plt.grid(True)

        pts = np.array([(1, -1), (-1.5, 2), (-1.5, 1)], float)
        t = ellipse_api.nearest_t(m, center, ang, pts)
        # projected_pts = parametric_pts(coeffs, t)
        projected_pts = ellipse_api.nearest_contour_pt(m, center, ang, pts, refine=False)
        for pt, ppt in zip(pts, projected_pts):
            x1, y1 = pt
            x2, y2 = ppt
            plt.plot([x1, x2], [y1, y2], 'ro-')

        projected_pts = ellipse_api.nearest_contour_pt(m, center, ang, pts, refine=True)
        for pt, ppt in zip(pts, projected_pts):
            x1, y1 = pt
            x2, y2 = ppt
            plt.plot([x1, x2], [y1, y2], 'ko-')

        plt.plot(*ellipse_api.parametric_pts(m, center, ang, np.linspace(-20, 10, 100) * np.pi / 180).T, 'y.-')
        tt = np.linspace(min(t), max(t), 500)
        plt.plot(*ellipse_api.parametric_pts(m, center, ang, tt).T, 'c.')

        plt.show()

    # --------
    # parameterization:
    # t = angular parameter, i.e. a*cos(t), b*sin(t)
    # s = arclength
    # p = arclength normalized by semi latus

    @staticmethod
    def arclen_to_p(m, s):
        return s / (np.pi * ellipse_api.semi_latus(m))

    @staticmethod
    def p_to_arclen(m, p):
        return p * np.pi * ellipse_api.semi_latus(m)

    @staticmethod
    def t_to_p(m, t):
        s = ellipse_api._arclen_convert(m, t=t)
        p = ellipse_api.arclen_to_p(m, s)
        return p

    @staticmethod
    def p_to_t(m, p):
        s = ellipse_api.p_to_arclen(m, p)
        t = ellipse_api._arclen_convert(m, s=s)
        return t

    @staticmethod
    def calc_radius_from_eccentricity(m, e):
        r = ellipse_api.radii_ratio_from_eccentricity(e)
        a, b = m
        if a is None:
            m = b / r, b
        elif b is None:
            m = a, a * r
        else:
            raise ValueError("eccentricity provided, but both radii are valid")
        return m

    @staticmethod
    def _arclen_convert(m, *, t=None, s=None):
        tt = np.linspace(0, 2 * np.pi, 180)
        s_tt = polytools.arclen(np.c_[m[0] * np.cos(tt), m[1] * np.sin(tt)])
        if t is not None:
            assert s is None
            return np.sign(t) * interp1d(tt, s_tt)(np.abs(t))
        else:
            return np.sign(s) * interp1d(s_tt, tt)(np.abs(s))

    @staticmethod
    def _calc_dists_to_pts(m, center, ang, pts, refine: bool = False):
        tt = np.linspace(0, 2 * np.pi, 180)
        ellipse_pts = ellipse_api.parametric_pts(m, center, ang, t=tt)
        squared_dists = cdist(pts, ellipse_pts, 'sqeuclidean')
        mindist_ixs = np.argmin(squared_dists, axis=1)
        squared_dists = squared_dists[np.arange(len(pts)), mindist_ixs]
        t = tt[mindist_ixs]
        if refine:
            dthetas = np.linspace(-1, 1, 45) * np.pi / len(tt)
            for i, (ti, pti) in enumerate(zip(t, pts)):
                ellipse_pts = ellipse_api.parametric_pts(m, center, ang, t=ti + dthetas)
                d = np.sum((pti - ellipse_pts) ** 2, axis=1)
                mindist_ix = np.argmin(d)
                squared_dists[i] = d[mindist_ix]
                t[i] += dthetas[mindist_ix]
        return t, squared_dists


def _test_coeff_params_conversions():
    true_params = (3.2, 1), np.array([-1, 2.1]), 30
    coeffs = ellipse_api.coefficients(*true_params)
    (est_a, est_b), est_center, est_ang = ellipse_api.parameters(coeffs)
    est_a = round(est_a, 4)
    est_b = round(est_b, 4)
    est_ang = round(est_ang, 4)
    print(((est_a, est_b), est_center, est_ang))



def _test_polar_conversions():
    m, center, ang = (2, 1), np.array([0, 0.4]), 90
    ellipse_api.draw(m, center, ang)
    #
    # e, l, f = to_polar(coeffs)
    #
    # t = np.linspace(0, 359, 360) * np.pi / 180
    # r = l / (1 + e * np.cos(t))
    # pts = np.c_[r * np.cos(t), r * np.sin(t)]
    # plt.plot(*pts.T, 'k.')
    # plt.show()

#_test_polar_conversions()
#_test_polar_conversions()
if __name__ == "__main__":
    ellipse_api.draw((2, 1), np.array((-.5, .1), float), 310)
