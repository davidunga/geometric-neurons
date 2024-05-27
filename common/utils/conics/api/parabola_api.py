import numpy as np
from common.utils.conics import conic_coeffs
from common.utils.linalg import rotate_points
from common.utils.conics.api.conics_api import conic_api
from common.utils import strtools
import matplotlib.pyplot as plt


class parabola_api(conic_api):

    @staticmethod
    def kind():
        return 'p'

    @staticmethod
    def kind_name():
        return 'Parabola'

    @staticmethod
    def str(m, center, theta):
        return parabola_api.kind_name() + ' ' + strtools.to_str((m, center, theta * 180 / np.pi), f='2.2')

    @staticmethod
    def parameters(coeffs):
        theta = conic_coeffs.rotation_theta(coeffs, relto='y')
        A, B, C, D, E, F = conic_coeffs.rotate(coeffs, -theta)
        m, n, k = -A / E, -D / E, -F / E
        x0 = -n / (2 * m)
        y0 = m * x0 ** 2 + n * x0 + k
        vertex = np.array(rotate_points(x0, y0, rad=theta))
        return m, np.array([x0, y0]), theta

    @staticmethod
    def coefficients(m, vertex, theta):
        x0, y0 = vertex
        D = -2 * m * x0
        F = m * x0 ** 2 + y0
        coeffs = conic_coeffs.rotate((m, 0, 0, D, -1, F), theta)
        return coeffs

    @staticmethod
    def approx_dist(m, vertex, theta, pts):
        x0, y0 = rotate_points(*vertex, rad=-theta)
        x, y = rotate_points(*pts.T, rad=-theta)
        dists = np.abs((y - y0) - m * (x - x0) ** 2)
        return dists

    @staticmethod
    def parametric_pts(m, vertex, theta, t):
        x = parabola_api.semi_latus(m) * t
        y = m * x ** 2
        x, y = rotate_points(x, y, rad=theta)
        return np.c_[x + vertex[0], y + vertex[1]]

    @staticmethod
    def nearest_parameter(m, vertex, theta, pts):
        x0, y0 = rotate_points(*vertex, rad=-theta)
        x, y = rotate_points(*pts.T, rad=-theta)
        t = (x - x0) / parabola_api.semi_latus(m=m)
        return t

    @staticmethod
    def semi_latus(m) -> float:
        return 2 * abs(m)

    @staticmethod
    def curvature_at_vertex(m) -> float:
        return 2 * np.abs(m)

    @staticmethod
    def eccentricity(m) -> float:
        return 1.

    @staticmethod
    def draw(m, vertex, theta, t=None, *args, **kwargs):
        if t is None:
            t = np.linspace(-1, 1, 400)
        # Generate parametric points for the parabola
        parametric_points = parabola_api.parametric_pts(m, vertex, theta, t)

        # Plot the parabola
        plt.figure(figsize=(8, 8))
        plt.plot(parametric_points[:, 0], parametric_points[:, 1], label='Parabola')

        # Plot the vertex
        plt.plot(vertex[0], vertex[1], 'ro', label='Vertex')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(parabola_api.str(m, vertex, theta))
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        # plt.show()


def _test_coeff_params_conversions():
    true_params = 3.2, np.array([-1, 2.1]), .2
    parabola_api.draw(*true_params)
    # coeffs = parabola_api.coefficients(*true_params)
    # est_m, est_center, est_theta = parabola_api.parameters(coeffs)
    # est_m = round(est_m, 4)
    # est_theta = round(est_theta, 4)
    # print((est_m, est_center, est_theta))

#_test_coeff_params_conversions()