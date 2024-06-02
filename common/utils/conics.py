import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from common.utils.conics.api import get_conic_api, \
    digest_and_validate_conic_params, ellipse_api, \
    get_conic_kind_and_params, parabola_api, conic_api, parabola_to_ellipse
from common.utils.conics import conic_coeffs, conic_arc
from common.utils import strtools


class Conic:

    def __init__(self, m, loc=None, ang=.0, bounds=(-1, 1),
                 mse: float = None, inlier_count: int = None, e: float = None):
        if e is not None:
            m = ellipse_api.calc_radius_from_eccentricity(m, e)
        kind, params = digest_and_validate_conic_params(m, loc=(0, 0) if loc is None else loc, ang=ang)
        self.api = get_conic_api(kind)
        self.params = params
        self.bounds = bounds
        self.mse = float('nan') if mse is None else mse
        self.inlier_count = inlier_count

    @property
    def m(self): return self.params[0]

    @property
    def loc(self): return self.params[1]

    @property
    def ang(self): return self.params[2]

    @property
    def kind(self): return self.api.kind()

    def __str__(self) -> str:
        s = self.api.str(*self.params)
        if self.bounds is not None:
            s += ' ' + strtools.to_str(self.bounds, f='2.2')
        return s

    def draw(self, *args, **kwargs):
        return self.api.draw(*self.params, bounds=self.bounds, *args, **kwargs)

    def parametric_pts(self, n: int = 100):
        return self.api.parametric_pts(*self.params, p=np.linspace(*self.bounds, n))


def fit_conic_ransac(pts: np.ndarray, fit_indexes: np.ndarray[int] = None,
                     kinds: Sequence[str] = ('e', 'p'),
                     arc: bool = False, seed: int = 1, thresh: float = .5, n: int = 7,
                     n_itrs: int = 1000, e_thresh: bool = .95, arc_ang: float = 20) -> Conic:

    thresh2 = thresh ** 2

    def _calc_scores(params, kind):
        conic_api = get_conic_api(kind)
        r2 = conic_api.semi_latus(params[0]) ** 2
        norm_dists2 = conic_api.approx_dist2(*params, pts) / r2
        inliers = norm_dists2 < thresh2
        inlier_count = sum(inliers)
        mse = np.mean(norm_dists2)
        return {'mse': mse, 'inlier_count': inlier_count, 'inliers': inliers}

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
    best_params = None
    scores = None
    for itr in range(n_itrs):
        ii = rng.permutation(fit_indexes)[:n]
        coeffs = conic_coeffs.fit_lsqr(*pts[ii].T)
        kind, params = get_conic_kind_and_params(coeffs)
        scores = _calc_scores(params, kind)
        if _is_better(scores, best_scores):
            if 'p' in kinds and kind != 'p':
                ang0 = params[2] - 90
                ang_min, ang_max = ang0 - 20, ang0 + 20
                parabola_params = parabola_api.parameters(
                    conic_coeffs.fit_conic_parabola(pts[ii], ang_min=ang_min, ang_max=ang_max))
                parabola_scores = _calc_scores(parabola_params, 'p')
                if _is_better(parabola_scores, scores):
                    scores = parabola_scores
                    params = parabola_params
            best_scores = scores
            best_params = params

    kind, params = digest_and_validate_conic_params(*best_params)
    conic_api = get_conic_api(kind)
    lb, ub = conic_api.nearest_t(*params, pts[[0, -1]], refine=True)
    if lb > ub:
        if kind == 'e':
            lb = - (2 * np.pi - lb)
        elif kind == 'p':
            lb, ub = ub, lb
    bounds = conic_api.t_to_p(params[0], np.asarray([lb, ub], float))
    conic = Conic(*params, bounds=bounds, mse=best_scores['mse'], inlier_count=best_scores['inlier_count'])
    return conic



if __name__ == "__main__":
    rng = np.random.default_rng(1)
    conic = Conic((5, None), loc=(-1, 2), ang=-50, bounds=[-3, 1], e=.999999)

    pts = conic.parametric_pts(100)
    #pts += np.std(pts, axis=0) * .01 * rng.standard_normal(size=pts.shape)
    fitted_conic = fit_conic_ransac(pts, arc=False, kinds=('p','e'), n_itrs=500)

    plt.plot(*pts.T, 'k-', label='GT')
    print(conic)
    print(fitted_conic)
    plt.plot(*fitted_conic.parametric_pts().T, 'r.', label='Fit')
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
