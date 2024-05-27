import numpy as np
from common.utils.conics import conic_coeffs
from common.utils.conics.api.conics_api import conic_api
from common.utils.linalg import rotate_points
from scipy.spatial.distance import cdist
from common.utils import strtools
import matplotlib.pyplot as plt


class ellipse_api(conic_api):

    @staticmethod
    def kind():
        return 'e'

    @staticmethod
    def kind_name():
        return 'Ellipse'

    @staticmethod
    def str(m, center, theta):
        s = strtools.to_str((m, center, theta * 180 / np.pi), f='2.2')
        return f'{ellipse_api.kind_name()} {s} e={ellipse_api.eccentricity(m):2.2f}'

    @staticmethod
    def parameters(coeffs):

        conic_coeffs.raise_strict_conic_type(coeffs, 'e')

        A, B, C, D, E, F = coeffs
        d = B ** 2 - 4 * A * C

        # center:
        center = np.array([2 * C * D - B * E, 2 * A * E - B * D]) / d

        # axes:
        M0 = np.array([[F, D / 2, E / 2], [D / 2, A, B / 2], [E / 2, B / 2, C]])
        M = np.array([[A, B / 2], [B / 2, C]])
        eigvals = np.linalg.eigvals(M)[:2]
        k = np.linalg.det(M) * eigvals
        if np.any(k == 0):
            raise conic_coeffs.ConicTypeError()
        v = -np.linalg.det(M0) / k
        if np.any(v < 0):
            raise conic_coeffs.ConicTypeError()
        a, b = np.sqrt(v)
        a, b = (a, b) if a > b else (b, a)

        theta = conic_coeffs.rotation_theta(coeffs)

        return (a, b), center, theta

    @staticmethod
    def focus(m, center, theta):
        a, b = m
        c = np.sqrt(a ** 2 - b ** 2)
        x, y = rotate_points([-c, c], [0, 0], rad=theta)
        return np.c_[x, y] + center

    @staticmethod
    def approx_dist(m, center, theta, pts):
        return np.sqrt(np.min(_approx_cdist2(m, center, theta, pts)[0], axis=1))

    @staticmethod
    def nearest_parameter(m, center, theta, pts):
        d, t, _ = _approx_cdist2(m, center, theta, pts)
        return t[np.argmin(d, axis=1)]

    @staticmethod
    def nearest_contour_pt(m, center, theta, pts):
        d, _, contour_pts = _approx_cdist2(m, center, theta, pts)
        return contour_pts[np.argmin(d, axis=1)]

    @staticmethod
    def parametric_pts(m, center, theta, t):
        a, b = m
        x, y = rotate_points(a * np.cos(t), b * np.sin(t), rad=theta)
        return np.c_[x + center[0], y + center[1]]

    @staticmethod
    def coefficients(m, center, theta):
        a, b = m
        coeffs0 = (1 / a ** 2, 0, 1 / b ** 2, 0, 0, -1)
        coeffs = conic_coeffs.translate(conic_coeffs.rotate(coeffs0, theta), center)
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
    def draw(m, center, theta, t=None, *args, **kwargs):

        if t is None:
            t = np.linspace(0, 2 * np.pi, 1000)

        a, b = m
        focus_pts = ellipse_api.focus((a, b), center, theta)

        plt.figure(figsize=(8, 8))
        contour_pts = ellipse_api.parametric_pts(m, center, theta, t)
        lines = plt.plot(*contour_pts.T)
        plt.plot(*contour_pts[0], 'o', color=lines[0].get_color())

        major_axis = np.c_[rotate_points([-a, a], [0, 0], theta)] + center
        minor_axis = np.c_[rotate_points([0, 0], [-b, b], theta)] + center
        plt.plot(*major_axis.T, 'r--', label='Major Axis')
        plt.plot(*minor_axis.T, 'g--', label='Minor Axis')

        plt.plot(*center.squeeze(), 'bo', label='Center')

        for i, (fx, fy) in enumerate(focus_pts):
            plt.plot(fx, fy, 'mo', label=f'Focus{i + 1}')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(ellipse_api.str(m, center, theta))
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        #
        # pts = np.array([(1, -1), (-1.5, 2), (-1.5, 1)], float)
        # t = ellipse_api.nearest_parameter(m, center, theta, pts)
        # # projected_pts = parametric_pts(coeffs, t)
        # projected_pts = ellipse_api.nearest_contour_pt(m, center, theta, pts)
        # for pt, ppt in zip(pts, projected_pts):
        #     x1, y1 = pt
        #     x2, y2 = ppt
        #     plt.plot([x1, x2], [y1, y2], 'ro-')
        #
        # plt.plot(*ellipse_api.parametric_pts(m, center, theta, np.linspace(-20, 10, 100) * np.pi / 180).T, 'y.-')
        # tt = np.linspace(min(t), max(t), 500)
        # plt.plot(*ellipse_api.parametric_pts(m, center, theta, tt).T, 'c.')

        # plt.show()


def _approx_cdist2(m, center, theta, pts):
    samples_per_degree = 2
    a, b = m
    circumference = np.pi * (a + b)
    scale = 1 / circumference
    t = np.linspace(0, 2 * np.pi, 360 * samples_per_degree)
    contour = np.stack(rotate_points(a * np.cos(t), b * np.sin(t), rad=theta), axis=1) + center
    d = cdist(pts * scale, contour * scale, metric='sqeuclidean') / (scale ** 2)
    return d, t, contour


def _test_coeff_params_conversions():
    true_params = (3.2, 1), np.array([-1, 2.1]), 1.6708
    coeffs = ellipse_api.coefficients(*true_params)
    (est_a, est_b), est_center, est_theta = ellipse_api.parameters(coeffs)
    est_a = round(est_a, 4)
    est_b = round(est_b, 4)
    est_theta = round(est_theta, 4)
    print(((est_a, est_b), est_center, est_theta))


def _test_t_vs_points_conversions():
    params = (2, 1), np.array([0, 0]), 0
    print(ellipse_api.nearest_parameter(*params, np.array([[0, 1]])))



def _test_polar_conversions():
    m, center, theta = (2, 1), np.array([0, 0.4]), 20 * np.pi / 180
    ellipse_api.draw(m, center, theta)
    #
    # e, l, f = to_polar(coeffs)
    #
    # t = np.linspace(0, 359, 360) * np.pi / 180
    # r = l / (1 + e * np.cos(t))
    # pts = np.c_[r * np.cos(t), r * np.sin(t)]
    # plt.plot(*pts.T, 'k.')
    # plt.show()


#_test_polar_conversions()
