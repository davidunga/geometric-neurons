import numpy as np
from common.utils.conics import conic_coeffs
from common.utils.linalg import rotate_points
from common.utils.conics.api.conics_api import conic_api
from common.utils import strtools
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class parabola_api(conic_api):

    @staticmethod
    def kind():
        return 'p'

    @staticmethod
    def kind_name():
        return 'Parabola'

    @staticmethod
    def str(m, center, ang):
        return parabola_api.kind_name() + ' ' + strtools.to_str((m, center, ang), f='2.2')

    @staticmethod
    def parameters(coeffs):
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
        return m, vertex, ang

    @staticmethod
    def coefficients(m, vertex, ang):
        x0, y0 = vertex
        D = -2 * m * x0
        F = m * x0 ** 2 + y0
        coeffs = conic_coeffs.rotate((m, 0, 0, D, -1, F), ang * np.pi / 180)
        return coeffs

    @staticmethod
    def approx_dist2(m, vertex, ang, pts, refine: bool = False):
        return parabola_api._calc_dists_to_pts(m, vertex, ang, pts)[1]
        # x0, y0 = rotate_points(*vertex, deg=-ang)
        # x, y = rotate_points(*pts.T, deg=-ang)
        # dists = np.square((y - y0) - m * (x - x0) ** 2)
        # return dists

    @staticmethod
    def parametric_pts(m, vertex, ang, p=None, t=None):
        if p is not None:
            assert t is None
            t = parabola_api.p_to_t(m, p)
        yt = m * t ** 2
        x, y = rotate_points(t, yt, deg=ang)
        return np.c_[x + vertex[0], y + vertex[1]]

    @staticmethod
    def nearest_p(m, vertex, ang, pts, refine: bool = False):
        return parabola_api.t_to_p(m, parabola_api.nearest_t(m, vertex, ang, pts))

    @staticmethod
    def nearest_t(m, vertex, ang, pts, refine: bool = False):
        return parabola_api._calc_dists_to_pts(m, vertex, ang, pts)[0]
        # x0, y0 = rotate_points(*vertex, deg=-ang)
        # x, y = rotate_points(*pts.T, deg=-ang)
        # return x - x0

    @staticmethod
    def focus_pts(m, vertex, ang):
        x, y = rotate_points(vertex[0], vertex[1] + parabola_api.focus_to_vertex_dist(m), deg=ang)
        return np.array([x, y])

    @staticmethod
    def vertex_pts(m, vertex, ang):
        return np.array(vertex, float)

    @staticmethod
    def focus_to_vertex_dist(m):
        return 1 / abs(4 * m)

    @staticmethod
    def semi_latus(m) -> float:
        return 1 / (2 * abs(m))

    @staticmethod
    def curvature_at_vertex(m) -> float:
        return 2 * abs(m)

    @staticmethod
    def eccentricity(m) -> float:
        return 1.

    @staticmethod
    def radii_ratio_from_eccentricity(e: float = None) -> float:
        """ b/a ratio """
        return .0

    @staticmethod
    def arclen_to_p(m, s):
        return s / (np.pi * parabola_api.semi_latus(m))

    @staticmethod
    def p_to_arclen(m, p):
        return p * np.pi * parabola_api.semi_latus(m)

    @staticmethod
    def t_to_p(m, t):
        s = parabola_api._arclen_convert(m, t=t)
        p = parabola_api.arclen_to_p(m, s)
        return p

    @staticmethod
    def p_to_t(m, p):
        s = parabola_api.p_to_arclen(m, p)
        t = parabola_api._arclen_convert(m, s=s)
        return t

    @staticmethod
    def nearest_contour_pt(m, center, ang, pts, refine: bool = False):
        t = parabola_api.nearest_t(m, center, ang, pts, refine=refine)
        return parabola_api.parametric_pts(m, center, ang, t=t)

    @staticmethod
    def _arclen_convert(m, *, t=None, s=None):

        ff = 4 * abs(m)
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

    @staticmethod
    def draw(m, vertex, ang, bounds=None, *args, **kwargs):

        if bounds is None:
            bounds = parabola_api.t_to_p(m, [-np.pi, np.pi])

        focus_pt = parabola_api.focus_pts(m, vertex, ang)

        # Generate parametric points for the parabola
        parametric_points = parabola_api.parametric_pts(m, vertex, ang, p=np.linspace(*bounds, 100))

        # Plot the parabola
        plt.figure(figsize=(8, 8))
        plt.plot(parametric_points[:, 0], parametric_points[:, 1], label='Parabola')

        # Plot the vertex
        plt.plot(vertex[0], vertex[1], 'ro', label='Vertex')
        plt.plot(focus_pt[0], focus_pt[1], 'mo', label='Focus')

        d = parabola_api.focus_to_vertex_dist(m)
        l = parabola_api.semi_latus(m)
        x, y = rotate_points([-l, l], [d, d], deg=ang)
        plt.plot(x + vertex[0], y + vertex[1], ':')

        pts = np.array([(1, -1), (-1.5, 2), (-1.5, 1)], float)
        t = parabola_api.nearest_t(m, vertex, ang, pts)
        # projected_pts = parametric_pts(coeffs, t)
        projected_pts = parabola_api.nearest_contour_pt(m, vertex, ang, pts)
        for pt, ppt in zip(pts, projected_pts):
            x1, y1 = pt
            x2, y2 = ppt
            plt.plot([x1, x2], [y1, y2], 'ro-')
            plt.plot([x1], [y1], 'ko')


        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(parabola_api.str(m, vertex, ang))
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        # plt.show()

    @staticmethod
    def _calc_dists_to_pts(m, vertex, ang, pts, refine: bool = False):
        xx, yy = rotate_points(pts[:, 0] - vertex[0], pts[:, 1] - vertex[1], deg=-ang)
        t = np.zeros(len(pts), float)
        squared_dists = np.zeros(len(pts), float)
        for i, (x, y) in enumerate(zip(xx, yy)):
            roots = np.roots([2 * m ** 2, 0, -2 * m * y + 1, -x])
            dists2 = [(xi - x) ** 2 + (m * xi ** 2 - y) ** 2 if xi.imag == 0 else np.inf for xi in roots]
            mindist_ix = np.argmin(dists2)
            t[i] = roots[mindist_ix].real
            squared_dists[i] = dists2[mindist_ix].real
        return t, squared_dists


def _test_dist():
    print("---")
    ang = 95
    params = 1, (0, 0), ang
    pts = np.asarray([(0, 0), (1, 1), (0, 1)], float)
    pts = np.c_[rotate_points(*pts.T, deg=ang)]
    dists = parabola_api.approx_dist(*params, pts)
    for pt, dist in zip(pts, dists):
        print(np.round(pt, 2), dist)


def _test_coeff_params_conversions():
    true_params = 3.2, np.array([-1, 2.1]), .2
    parabola_api.draw(*true_params)
    # coeffs = parabola_api.coefficients(*true_params)
    # est_m, est_center, est_theta = parabola_api.parameters(coeffs)
    # est_m = round(est_m, 4)
    # est_theta = round(est_theta, 4)
    # print((est_m, est_center, est_theta))


def _test_fit():
    true_params = .32, 1*np.array([-10, 2.1]), 10
    x = np.linspace(-2, 2, 100)
    y = true_params[0] * x ** 2
    pts = np.c_[rotate_points(x, y, deg=true_params[-1])] + true_params[1]

    coeffs = conic_coeffs.fit_conic_parabola(pts)
    p = parabola_api.parameters(coeffs)
    print(strtools.to_str(true_params, '2.2'))
    print(strtools.to_str(p, '2.2'))

    plt.plot(*pts.T,'r.')
    plt.plot(*parabola_api.parametric_pts(*p, np.linspace(-20, 20, 100)).T,'.b')
    plt.show()






#_test_coeff_params_conversions()
if __name__ == "__main__":
    parabola_api.draw(m=1, vertex=(0, 0), ang=-20.0)
    plt.show()
    #_test_fit()
    # m = 2
    # t = np.linspace(-4, 3, 100)
    # y = m * t ** 2
    # from common.utils import polytools
    # s_est = parabola_api.p_to_arclen(m, parabola_api.t_to_p(m, t))
    # s_gt = polytools.arclen(np.c_[t, y])
    # s_gt -= s_gt[np.argmin(np.abs(t))]
    # plt.figure()
    # plt.plot(s_gt, 'ro')
    # plt.plot(s_est, 'c.')
    #
    # t_est = parabola_api.p_to_t(m, parabola_api.arclen_to_p(m, s_gt))
    # plt.figure()
    # plt.plot(t, 'ro')
    # plt.plot(t_est, 'c.')
    # plt.show()
