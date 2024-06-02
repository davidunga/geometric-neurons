import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from common.utils.conics import conic_coeffs, conic_arc
from common.utils.conics import Conic, ConicParabola, ConicEllipse, get_conic
from common.utils.linalg import rotate_points


def eval_conic_fit(conic: Conic, pts: np.ndarray, normdist_thresh: float, refine: bool = False):
    r2 = conic.arclen_scale_factor() ** 2
    norm_dists2 = conic.squared_dists(pts, refine=refine)[0] / r2
    inliers = norm_dists2 < normdist_thresh ** 2
    inlier_count = int(np.sum(inliers))
    mse = np.mean(norm_dists2)
    scores = {'mse': mse,
              'inlier_count': inlier_count,
              'inlier_p': float(inlier_count) / len(pts)}
    return scores, inliers, norm_dists2


def fit_conic_ransac(pts: np.ndarray, fit_indexes: np.ndarray[int] = None,
                     kinds: Sequence[str] = ('e', 'p'), inlier_p_thresh: float = .9,
                     arc: bool = False, seed: int = 1, normdist_thresh: float = .05, n: int = 7,
                     max_itrs: int = 1000, arc_ang: float = 20) -> tuple[Conic, dict]:

    kinds = set(kinds)
    assert kinds.issubset(('e', 'p'))

    inlier_count_thresh = inlier_p_thresh * len(pts)

    def _calc_scores(conic: Conic, refine: bool = False):
        return eval_conic_fit(conic, pts, normdist_thresh=normdist_thresh, refine=refine)[0]

    def _is_better(s1, s2) -> bool:
        """ check if s1 score is better than s2 """
        if s1['inlier_count'] == s2['inlier_count']:
            return s1['mse'] < s2['mse']
        else:
            return s1['inlier_count'] > s2['inlier_count']

    parabola_fit_kws = {}
    if arc:
        assert fit_indexes is None
        arc_props = conic_arc.get_approx_arc_properties(pts)
        fit_indexes = np.arange(arc_props['start_ix'], arc_props['stop_ix'])
        parabola_fit_kws['ang_min'] = arc_props['ang'] - arc_ang
        parabola_fit_kws['ang_max'] = arc_props['ang'] + arc_ang

    if fit_indexes is None:
        fit_indexes = np.arange(len(pts))

    rng = np.random.default_rng(seed)
    best_scores = {'mse': np.inf, 'inlier_count': 0}
    best_conic = None
    best_sample = None
    scores = None
    for itr in range(max_itrs):
        ii = rng.permutation(fit_indexes)[:n]
        conic = get_conic(_fit_lsqr_coeffs(*pts[ii].T))
        scores = _calc_scores(conic)
        if _is_better(scores, best_scores):
            best_scores = scores
            best_conic = conic
            best_sample = ii
            if scores['inlier_count'] >= inlier_count_thresh:
                break

    conic = best_conic
    scores = _calc_scores(conic, refine=True)
    if 'p' in kinds and best_conic.kind != 'p':
        ang0 = conic.ang - 90
        ang_min, ang_max = ang0 - 10, ang0 + 10
        parabola_coeffs = _fit_parabola_coeffs(*pts[best_sample].T, ang_min=ang_min, ang_max=ang_max)
        parabola = ConicParabola.from_coeffs(parabola_coeffs)
        parabola_scores = _calc_scores(parabola)
        if _is_better(parabola_scores, scores) or kinds == {'p'}:
            scores = parabola_scores
            conic = parabola

    lb, ub = conic.nearest_t(pts[[0, -1]], refine=True)
    if lb > ub:
        if conic.kind == 'e':
            lb = - (2 * np.pi - lb)
        elif conic.kind == 'p':
            lb, ub = ub, lb
    conic._bounds = conic.t_to_p([lb, ub])

    return conic, scores


def _fit_parabola_coeffs(x, y, ang_min=0., ang_max=180., steps_per_search: int = 50):
    min_step_size = .01 * np.pi / 180

    def _search(thetas):
        errors = np.zeros_like(thetas, float)
        params = np.zeros((len(thetas), 3), float)
        for i, theta in enumerate(thetas):
            xx, yy = rotate_points(x, y, rad=-theta)
            params[i] = np.polyfit(xx, yy, deg=2)
            errors[i] = np.sum((np.polyval(params[i], xx) - yy) ** 2)
        i = np.argmin(errors)
        return thetas[i], params[i]

    thetas = np.radians(np.linspace(ang_min, ang_max, steps_per_search))
    step_size = thetas[1] - thetas[0]
    pfit, theta = None, None
    while step_size > min_step_size:
        theta, pfit = _search(thetas)
        thetas = np.linspace(theta - step_size, theta + step_size, steps_per_search)
        step_size = thetas[1] - thetas[0]

    assert pfit is not None
    A, D, F = pfit
    coeffs = conic_coeffs.rotate((A, 0., 0., D, -1., F), theta)
    return coeffs


def _fit_lsqr_coeffs(x, y):
    D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T
    _, _, V = np.linalg.svd(D)
    return V[-1, :]


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    conic = ConicParabola(3, loc=(-1, 2), ang=-50, bounds=[-1, 2])
    pts = conic.parametric_pts(n=100)
    pts += np.std(pts, axis=0) * .1 * rng.standard_normal(size=pts.shape)
    normdist_thresh = .08
    fitted_conic, _ = fit_conic_ransac(pts, arc=False, kinds=('e'), max_itrs=1500, normdist_thresh=normdist_thresh)
    scores, inliers, _ = eval_conic_fit(fitted_conic, pts, normdist_thresh=normdist_thresh, refine=True)
    plt.plot(*pts.T, 'k.', label='GT')
    plt.plot(*pts[inliers].T, 'c.', label='IN')
    print(conic)
    print(fitted_conic)
    plt.plot(*fitted_conic.parametric_pts().T, 'r.', label='Fit Inlier={inlier_p:2.3f} MSE={mse:2.5f}'.format(**scores))
    plt.axis('equal')
    plt.legend()
    plt.show()
