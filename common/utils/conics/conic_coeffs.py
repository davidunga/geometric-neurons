import numpy as np


class ConicTypeError(Exception):
    pass


def evaluate(coeffs, x: np.ndarray, y: np.ndarray):
    A, B, C, D, E, F = coeffs
    z = A * x ** 2 + B * x * y + C * y ** 2 + D * x + E * y + F
    return z


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
