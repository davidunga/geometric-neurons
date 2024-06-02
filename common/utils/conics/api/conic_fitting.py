import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from common.utils.conics.api import get_conic_api, \
    digest_and_validate_conic_params, ellipse_api, \
    get_conic_kind_and_params, parabola_api, conic_api, parabola_to_ellipse
from common.utils.conics.api.conic_section import ConicSection
from common.utils.conics.api.conic_ellipse import ConicEllipse
from common.utils.conics.api.conic_parabola import ConicParabola
from common.utils.conics import conic_coeffs, conic_arc
from common.utils import strtools


def get_conic(coeffs) -> ConicSection:
    try:
        conic = ConicEllipse.from_coeffs(coeffs)
    except conic_coeffs.ConicTypeError:
        conic = ConicParabola.from_coeffs(coeffs)
    return conic


def eval_conic_fit(conic: ConicSection, pts: np.ndarray, normdist_thresh: float, refine: bool = False):
    r2 = conic.arclen_scale_factor() ** 2
    norm_dists2 = conic.squared_dists(pts, refine=refine)[0] / r2
    inliers = norm_dists2 < normdist_thresh ** 2
    inlier_count = sum(inliers)
    mse = np.mean(norm_dists2)
    return {'mse': mse, 'inlier_count': inlier_count}, inliers, norm_dists2


def fit_conic_ransac(pts: np.ndarray, fit_indexes: np.ndarray[int] = None,
                     kinds: Sequence[str] = ('e', 'p'), inlier_p_thresh: float = .9,
                     arc: bool = False, seed: int = 1, normdist_thresh: float = .05, n: int = 7,
                     max_itrs: int = 1000, arc_ang: float = 20) -> dict:

    kinds = set(kinds)
    assert kinds.issubset(('e', 'p'))

    inlier_count_thresh = inlier_p_thresh * len(pts)

    def _calc_scores(conic: ConicSection, refine: bool = False):
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
        conic = get_conic(conic_coeffs.fit_lsqr(*pts[ii].T))
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
        parabola_coeffs = conic_coeffs.fit_conic_parabola(pts[best_sample], ang_min=ang_min, ang_max=ang_max)
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

    result = {
        'conic': conic,
        'mse': scores['mse'],
        'inlier_count': scores['inlier_count'],
        'inlier_p': scores['inlier_count'] / len(pts)
    }

    return result



if __name__ == "__main__":
    rng = np.random.default_rng(1)
    conic = ConicParabola(3, loc=(-1, 2), ang=-50, bounds=[-1, 2])

    pts = conic.parametric_pts(n=100)
    pts += np.std(pts, axis=0) * .1 * rng.standard_normal(size=pts.shape)
    normdist_thresh = .08
    fit_result = fit_conic_ransac(pts, arc=False, kinds=('e'), max_itrs=1500, normdist_thresh=normdist_thresh)

    scores, inliers, _ = eval_conic_fit(fit_result['conic'], pts, normdist_thresh=normdist_thresh, refine=True)

    plt.plot(*pts.T, 'k.', label='GT')
    plt.plot(*pts[inliers].T, 'c.', label='IN')
    print(conic)
    print(fit_result['conic'])
    plt.plot(*fit_result['conic'].parametric_pts().T, 'r.', label='Fit Inlier={inlier_p:2.3f} MSE={mse:2.5f}'.format(**fit_result))
    plt.axis('equal')
    plt.legend()
    plt.show()
    #ellipse = Conic((3, 1), bounds=(-.25, .25))
    #
    # d = parabola_api.focus_to_vertex_dist(parabola.m)
    # l = parabola_api.semi_latus(parabola.m) * .999
    #
    # a = (d ** 2) / (2 * d - l)
    # b = np.sqrt(a * l)
    #
    # focus_pt = parabola.api.focus_pts(*parabola.params)
    # v = focus_pt - parabola.loc
    # center = parabola.loc + v / np.linalg.norm(v) * a
    #
    # ellipse = Conic((a, b), loc=center, ang=parabola.ang-90)

    # plt.plot(*parabola.parametric_pts().T, 'r.')
    # plt.plot(*ellipse.parametric_pts().T, 'b.')
    # plt.axis('equal')
    # plt.show()
    #
    # fitted_conic = fit_conic_ransac(pts, kinds=('e', 'e'), arc=False)
    # print("true", str(true_conic))
    # print("fitted", str(fitted_conic))
    #
    # plt.plot(*pts.T, 'k.')
    # plt.plot(*true_conic.parametric_pts().T, 'ro', label='True ' + str(true_conic))
    # #plt.plot(*fitted_conic.parametric_pts().T, 'c-', label='Fit  ' + str(fitted_conic))
    # plt.plot(*approx_ellipse.parametric_pts().T, 'b-', label='Approx  ' + str(approx_ellipse))
    # plt.axis('equal')
    # plt.legend()
    # plt.show()
