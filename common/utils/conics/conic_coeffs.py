import numpy as np
from common.utils.linalg import rotate_points

class ConicTypeError(Exception):
    pass


def evaluate(coeffs, x: np.ndarray, y: np.ndarray):
    A, B, C, D, E, F = coeffs
    z = A * x ** 2 + B * x * y + C * y ** 2 + D * x + E * y + F
    return z


def fit_lsqr(x, y):
    D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T
    _, _, V = np.linalg.svd(D)
    return V[-1, :]


def rotation_theta(coeffs, relto: str = 'x'):
    assert relto in ('x', 'y')
    A, B, C = coeffs[:3]
    if np.abs(A - C) == 0:
        return 0.
    theta = .5 * np.arctan(B / (A - C))
    if abs(A) > abs(C):
        theta += np.pi / 2
    if relto == 'y':
        theta -= np.pi / 2
    return theta


def rotate(coeffs, theta: float):
    A, B, C, D, E, F = coeffs
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    Q = R @ np.array([[A, B / 2], [B / 2, C]]) @ R.T
    L = R @ np.array([D, E])
    A, B, C = Q[0, 0], 2 * Q[0, 1], Q[1, 1]
    D, E = L
    return A, B, C, D, E, F


def translate(coeffs, delta):
    dx, dy = -delta[0], -delta[1]
    A, B, C, D, E, F = coeffs
    dD = 2 * A * dx + B * dy
    dE = 2 * C * dy + B * dx
    dF = A * dx * dx + C * dy * dy + D * dx + E * dy + B * dx * dy
    D += dD
    E += dE
    F += dF
    return A, B, C, D, E, F


def scale(coeffs, sx: float, sy: float = None):
    if sy is None: sy = sx
    sx, sy = 1 / sx, 1 / sy
    scales = [sx ** 2, sx * sy, sy ** 2, sx, sy, 1.]
    return tuple(c * s for c, s in zip(coeffs, scales))


def get_strict_conic_type(coeffs) -> str:
    _eps = np.finfo(float).eps
    A, B, C, D, E, F = coeffs
    d = B ** 2 - 4 * A * C
    if np.abs(d) < _eps:
        return 'p'
    else:
        return 'e' if d < 0 else 'h'


def raise_strict_conic_type(coeffs, kinds):
    if isinstance(kinds, str): kinds = (kinds,)
    assert set(kinds).issubset(('e', 'p', 'h'))
    strict_type = get_strict_conic_type(coeffs)
    if strict_type not in kinds:
        raise ConicTypeError(f"Conic is {strict_type}, not {kinds}")


def fit_conic_parabola(pts, ang_min=0., ang_max=180.):
    steps_per_search = 10
    min_step_size = .25 * np.pi / 180

    def _search(thetas):
        errors = np.zeros_like(thetas, float)
        params = np.zeros((len(thetas), 3), float)
        for i, theta in enumerate(thetas):
            x, y = rotate_points(*pts.T, rad=-theta)
            params[i] = np.polyfit(x, y, deg=2)
            errors[i] = np.sum((np.polyval(params[i], x) - y) ** 2)
        i = np.argmin(errors)
        return thetas[i], params[i]

    thetas = np.radians(np.linspace(ang_min, ang_max, steps_per_search))
    step_size = thetas[1] - thetas[0]
    while step_size > min_step_size:
        theta, params = _search(thetas)
        thetas = np.linspace(theta - step_size, theta + step_size, steps_per_search)
        step_size = thetas[1] - thetas[0]

    A, D, F = params
    coeffs = rotate((A, 0., 0., D, -1., F), theta)
    return coeffs