import numpy as np
import matplotlib.pyplot as plt
from common.utils.linalg import planar, rotate
from contourpy import contour_generator, ContourGenerator
from common.utils import polytools
from scipy.spatial.distance import cdist
from typing import NamedTuple
import warnings

_ECCEN_TOL = 1e-3


class ConicParams(NamedTuple):
    kind: str
    a: float
    b: float | None
    loc: np.ndarray
    theta: float
    e: float


def fit_conic_lsqr(pts, allow_rotation: bool = True, kind: str = 'auto'):

    """ Implementation of
        Halır, R., & Flusser, J. (1998, February). Numerically stable direct least squares fitting of ellipses
        Modified to support parabolas, and no-rotation constraint
    """

    assert kind in ('e', 'p', 'auto')
    x, y = pts.T

    D1 = np.c_[x**2, x*y, y**2]
    if not allow_rotation:
        D1[:, 1] = .0
    D2 = np.c_[x, y, np.ones(len(x))]
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = np.array([[0, 0, .5], [0, -1, 0], [.5, 0, 0]]) @ (S1 + S2 @ T)
    _, eigvecs = np.linalg.eig(M)
    ii = 4 * eigvecs[0] * eigvecs[2] > eigvecs[1] ** 2
    v = eigvecs[:, ii]
    if v.shape[1] > 1:
        v = v[:, :1].real
    coeffs = np.concatenate((v, T @ v)).ravel()

    if len(coeffs) != 6:
        return None

    if kind == 'p' or (kind == 'auto' and abs(1 - ellipse_params(coeffs).e) < _ECCEN_TOL):
        theta = rotation_theta(coeffs, relto='y')
        x, y = rotate(x, y, rad=-theta)
        m, k, n = np.polyfit(x, y, deg=2)
        coeffs = rotate_coeffs((-m, 0., 0., -k, 1., -n), theta)

    return Conic(coeffs, kind=kind)


def fit_conic_ransac(pts,
                     kind: str,
                     tol: float,
                     seed: int = 1,
                     max_iters: int = 1000,
                     sample_sz: int = 6,
                     weights: np.ndarray = None,
                     high_priority_ixs: np.ndarray = None,
                     high_priority_pcnt: float = .5,
                     high_priority_penalty: float = 1000,
                     allow_rotation: bool = True):

    if isinstance(pts, tuple):
        pts = np.c_[pts[0], pts[1]]

    if weights is None:
        weights = np.ones(len(pts))
    assert len(weights) == len(pts)
    rng = np.random.default_rng(seed)
    best_score, best_conic = -1, None
    for itr in range(max_iters):
        ii = rng.permutation(len(pts))[:sample_sz]
        conic = fit_conic_lsqr(pts[ii], kind=kind, allow_rotation=allow_rotation)
        if conic is None:
            continue
        is_inlier = approx_dist_to_conic(conic, pts) < tol
        score = weights[is_inlier].sum()
        if high_priority_ixs is not None and is_inlier[high_priority_ixs].mean() < high_priority_pcnt:
            score /= high_priority_penalty
        if best_score < score:
            best_score = score
            best_conic = conic
    return best_conic


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


def rotate_coeffs(coeffs, theta: float):
    A, B, C, D, E, F = coeffs
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    Q = R @ np.array([[A, B / 2], [B / 2, C]]) @ R.T
    L = R @ np.array([D, E])
    A, B, C = Q[0, 0], 2 * Q[0, 1], Q[1, 1]
    D, E = L
    return A, B, C, D, E, F


def ellipse_params(coeffs) -> ConicParams:

    _check_raise_conic_type(coeffs, 'e')

    A, B, C, D, E, F = coeffs
    d = B ** 2 - 4 * A * C

    # center:
    center = np.array([2 * C * D - B * E, 2 * A * E - B * D]) / d

    # axes:
    M0 = np.array([[F, D / 2, E / 2], [D / 2, A, B / 2], [E / 2, B / 2, C]])
    M = np.array([[A, B / 2], [B / 2, C]])
    eigvals = np.linalg.eigvals(M)[:2]
    a, b = np.sqrt(-np.linalg.det(M0) / (np.linalg.det(M) * eigvals))
    a, b = (a, b) if a > b else (b, a)
    e = np.sqrt(1 - (b / a) ** 2)

    theta = rotation_theta(coeffs)
    return ConicParams(kind='e', a=a, b=b, loc=center, theta=theta, e=e)


def parabola_params(coeffs) -> ConicParams:
    theta = rotation_theta(coeffs, relto='y')
    A, B, C, D, E, F = rotate_coeffs(coeffs, -theta)
    m, n, k = -A / E, -D / E, -F / E
    x0 = -n / (2 * m)
    y0 = m * x0 ** 2 + n * x0 + k
    vertex = np.array(rotate(x0, y0, rad=theta))
    return ConicParams(kind='p', a=m, b=None, loc=vertex, theta=theta, e=1.)


class ConicTypeError(Exception):
    pass


def _check_raise_conic_type(coeffs, kind):
    if kind not in ('e', 'p', 'h'):
        raise ValueError("Unknown conic kind")
    _eps = np.finfo(float).eps
    A, B, C, D, E, F = coeffs
    d = B ** 2 - 4 * A * C
    if kind == 'e' and d > -_eps:
        raise ConicTypeError(f"Conic ({coeffs}) is not an ellipse. Discriminant={d}")
    elif kind == 'p' and np.abs(d) > _eps:
        raise ConicTypeError(f"Conic ({coeffs}) is not a parabola. Discriminant={d}")
    elif kind == 'h' and d < _eps:
        raise ConicTypeError(f"Conic ({coeffs}) is not a hyperbola. Discriminant={d}")


def evaluate_conic(coeffs, x: np.ndarray, y: np.ndarray):
    A, B, C, D, E, F = coeffs
    z = A * x ** 2 + B * x * y + C * y ** 2 + D * x + E * y + F
    return z


def approx_dist_to_conic(conic, pts):

    def _approx_dist_to_ellipse():
        ellipse_pts = polytools.uniform_resample(conic.parametric_pts(np.linspace(0, 2 * np.pi)), n=500)
        dists = np.sqrt(cdist(pts, ellipse_pts, 'sqeuclidean').min(axis=1))
        return dists

    def _approx_dist_to_parbola():
        theta = rotation_theta(conic.coeffs, relto='y')
        params = parabola_params(rotate_coeffs(conic.coeffs, -theta))
        x, y = rotate(*pts.T, rad=-theta)
        x -= params.loc[0]
        y -= params.loc[1]
        dists = np.abs(y - params.a * x ** 2)
        return dists

    if conic.is_parabola:
        return _approx_dist_to_parbola()
    else:
        return _approx_dist_to_ellipse()


class Conic:

    def __init__(self, coeffs, kind: str = 'auto'):
        assert kind in ('auto', 'e', 'p')
        self._coeffs = coeffs
        self._params = None
        self._kind = kind

    @property
    def params(self) -> ConicParams:
        if self._params is None:

            if self._kind in ('auto', 'e'):
                try:
                    self._params = ellipse_params(self.coeffs)
                except ConicTypeError:
                    if self._kind == 'e':
                        raise

            if self._kind != 'e' and (self._params is None or abs(1 - self._params.e) < _ECCEN_TOL):
                self._params = parabola_params(self.coeffs)

        return self._params

    @property
    def coeffs(self):
        return self._coeffs

    @property
    def is_parabola(self):
        return self.params.kind == 'p'

    @property
    def is_circle(self):
        return self.params.kind == 'e' and abs(self.params.e) < self._eccen_tol

    @property
    def kind(self):
        return 'Parabola' if self.is_parabola else 'Ellipse'

    def __repr__(self) -> str:
        coeffs_str = ", ".join([f"{k}={v:2.3f}" for k, v in zip(("A", "B", "C", "D", "E", "F"), self.coeffs)])
        return f"{self.params} ({coeffs_str})"

    def __str__(self) -> str:
        if self.is_parabola:
            return f"{self.kind} {self.params.a:2.3f}"
        else:
            return f"{self.kind} ({self.params.a:2.3f},{self.params.b:2.3f})"

    def draw(self, x, y, color='b', details: bool = False):
        x, y, z = self.eval_grid(x, y)
        plt.contour(x, y, z, levels=[0], colors=color)
        if details:
            x0, y0 = self.params.loc
            plt.plot(x0, y0, 'r*')
            if not self.is_parabola:
                vx, vy = rotate(self.params.a, 0, rad=self.params.theta)
                plt.plot([x0, x0 + vx], [y0, y0 + vy], '-', color='limeGreen')

    def contour(self, x, y, snap_mode: str = 'none'):
        x, y, z = self.eval_grid(x, y)
        pts = contour_generator(x, y, z).lines(0)[0]
        if snap_mode != 'none':
            if snap_mode == 'mindist':
                grid_pts = np.c_[x.flatten(), y.flatten()]
                pts = grid_pts[np.argmin(cdist(pts, grid_pts), axis=1)]
            elif snap_mode == 'round':
                pts = pts.round().astype(int)
            else:
                raise ValueError("Unknown snap mode")
            mask = np.ones(len(pts), bool)
            mask[1:] = np.any(pts[:1] != pts[:-1], axis=1)
            pts = pts[mask]
        return pts

    def eval(self, pts: np.ndarray) -> np.ndarray:
        return evaluate_conic(self.coeffs, *pts.T)

    def eval_grid(self, x, y):
        if isinstance(x, int): x = np.arange(x)
        if isinstance(y, int): y = np.arange(y)
        x, y = np.asarray(x), np.asarray(y)
        assert x.ndim == y.ndim
        if x.ndim == 1:
            x, y = np.meshgrid(x, y)
        z = evaluate_conic(self.coeffs, x, y)
        return x, y, z

    def parametric_pts(self, t: np.ndarray):
        if self.is_parabola:
            x, y = t, self.params.a * t ** 2
        else:
            x, y = self.params.a * np.cos(t), self.params.b * np.sin(t)
        pts = np.stack(rotate(x, y, rad=self.params.theta), axis=1) + self.params.loc
        return pts

    def get_transformed(self, offset=None, ang=0, sx=1, sy=None):
        if sy is None: sy = sx
        coeffs = self.coeffs
        if ang != 0:
            coeffs = rotate_coeffs(coeffs, ang * np.pi / 180)
        if sx != 1 or sy != 1:
            coeffs = scale_coeffs(coeffs, sx, sy)
        if offset is not None:
            coeffs = shift_coeffs(coeffs, offset)
        return Conic(coeffs)

# ----------


def make_conic_points(kind: str, m, start: float = None, stop: float = None, n: int = 200,
                      ang: float = .0, loc=.0, scale: float = 1):
    if kind == 'p':
        if start is None: start = -1.
        if stop is None: stop = 1.
        t = np.linspace(start, stop, n)
        pts = np.c_[t, m * t ** 2]
    elif kind == 'e':
        if start is None: start = 0.
        if stop is None: stop = 359.
        t = np.linspace(start, stop, n)
        a, b = (1, m) if m > 1 else (m, 1)
        t = np.radians(t)
        pts = np.c_[b * np.cos(t), a * np.sin(t)]
    else:
        raise ValueError('Unknown shape kind')
    pts = planar.transform(pts, scale=scale, ang=ang, offset=loc)
    return pts


def _test_fitting():
    from common.utils import polytools

    xlims = 0, 100
    ylims = 0, 100
    max_rel_offset = .2
    ang_lims = 0, 120
    m_lims = .9, 6
    true_kinds = 'p', 'e'
    noise_factor = .0002
    rng = np.random.default_rng(1)

    spans = np.array([xlims[1] - xlims[0], ylims[1] - ylims[0]])
    max_offset = max_rel_offset * spans
    loc0 = np.array([xlims[0] + xlims[1], ylims[0] + ylims[1]]) / 2
    scale = spans.mean()

    kind = 'p'

    loc = loc0 + 2 * (rng.random(size=2) - .5) * max_offset
    loc[1] -= 20
    ang = rng.random() * (ang_lims[1] - ang_lims[0]) + ang_lims[0]
    ang = 5
    m = rng.random() * (m_lims[1] - m_lims[0]) + m_lims[0]

    print(f"True params: kind={kind}, ang={ang:2.1f}, loc=({loc[0]:2.1f},{loc[1]:2.1f}), m={m:2.1f}")

    if kind == 'p':
        pts = make_conic_points(kind, m, start=-1.5, stop=2, ang=ang, scale=1, loc=loc)
    else:
        pts = make_conic_points(kind, m, start=0, stop=350, ang=ang, scale=scale/10, loc=loc)

    ii = np.all(pts > 0, axis=1) & (pts[:, 0] < xlims[1]) & (pts[:, 1] < ylims[1])
    #pts = pts[ii]
    pts = polytools.uniform_resample(pts, 100)

    #pts += noise_factor * np.std(pts, axis=0) * rng.standard_normal(size=pts.shape)

    c = Conic().fit(pts)
    c = Conic(fit_parabola(*pts.T))
    print(c)

    # #dd = approx_signed_dist_from_ellipse(c.coeffs, xx, yy)
    # dd = approx_dist_to_ellipse(c.coeffs, np.c_[xx.flatten(), yy.flatten()], norm=True).reshape(xx.shape)
    # plt.imshow(dd < .2, cmap='jet')

    #conic_pts = c.parametric_pts(np.linspace(-20, 20))

    plt.plot(*pts.T, 'b.')
    #plt.plot(*conic_pts.T, 'g.')
    c.draw(np.linspace(*xlims, 100 * xlims[1]-1), np.linspace(*ylims, 100 * ylims[1]-1), 'r', details=True)

    plt.show()


def shift_coeffs(coeffs, delta):
    dx, dy = -delta
    A, B, C, D, E, F = coeffs
    dD = 2 * A * dx + B * dy
    dE = 2 * C * dy + B * dx
    dF = A * dx * dx + C * dy * dy + D * dx + E * dy + B * dx * dy
    D += dD
    E += dE
    F += dF
    return A, B, C, D, E, F


def scale_coeffs(coeffs, sx: float, sy: float = None):
    if sy is None: sy = sx
    sx, sy = 1 / sx, 1 / sy
    scales = [sx ** 2, sx * sy, sy ** 2, sx, sy, 1.]
    return tuple(c * s for c, s in zip(coeffs, scales))


def _test_dist_funcs():

    kind = 'p'
    xlims = 0, 100
    ylims = 0, 100
    n = 100

    x = np.linspace(*xlims, n)
    y = np.linspace(*ylims, n)

    mid_pt = np.array([sum(xlims) / 2, sum(ylims) / 2], float)
    scale = float(np.mean([np.diff(xlims), np.diff(ylims)])) / 10

    xx, yy = np.meshgrid(x, y)

    m = 2
    ang = 22
    loc = np.array([0, 0]) + mid_pt
    if kind == 'p':
        pts = make_conic_points(kind, m, start=-1.5, stop=2, ang=ang, scale=scale, loc=loc)
    else:
        pts = make_conic_points(kind, m, start=0, stop=350, ang=ang, scale=scale, loc=loc)

    plt.plot(*pts.T, '.r')

    conic = fit_conic_lsqr(pts, kind='auto')
    print(conic)

    d = approx_dist_to_conic(conic, np.c_[xx.flatten(), yy.flatten()]).reshape(xx.shape)
    plt.imshow(d)
    #plt.plot(*conic.parametric_pts(np.linspace(-20, 20, 50)).T, 'co')
    conic.draw(x, y, details=False)
    plt.show()


if __name__ == "__main__":
    _test_dist_funcs()
    #_test_fitting()
    #
    # from common.utils import plotting
    #
    # pts = make_conic_points('p', m=1.2, loc=1*np.array([2, -3]), ang=-30, start=-2, stop=5)
    # x, y = plotting.get_grid_for_points(pts)
    #
    # c = Conic().fit(pts, kind='best')
    # print(c)
    #
    # plt.plot(*pts.T, 'k.')
    # c.draw(x, y, 'r')
    # #
    # # cg = c.get_contour_gen(x, y)
    # # lines = cg.lines(0.01)[0]
    # # plt.plot(lines[:, 0], lines[:, 1], '.c')
    #
    #
    # xy = c.parametric_pts(np.linspace(-1, 5, 100))
    # plt.plot(*xy.T, 'm.')
    # #
    # # c.normalize(scale=True).draw(x, y, 'g')
    # plotting.set_axis_equal()
    # plt.show()

