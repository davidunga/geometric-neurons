import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from common.utils.conics import conic_coeffs, conic_arc
from common.utils.conics import Conic, ConicParabola, ConicEllipse, get_conic
from common.utils.linalg import rotate_points, is_ccw
from common.utils import strtools
from common.utils.linalg import average_theta


def parabola_to_ellipse(parabola: ConicParabola, e: float = .9999) -> ConicEllipse:
    a = parabola.focus_dist() / (1 - e)
    b = a * np.sqrt(1 - e ** 2)
    dx, dy = rotate_points(a, 0, deg=parabola.ang - np.sign(parabola.m) * 90)
    center = np.array([parabola.loc[0] - dx, parabola.loc[1] - dy])
    ellipse = ConicEllipse(m=(a, b), loc=center, ang=parabola.ang + 90)
    return ellipse


from common.utils import polytools


def dist_to_line(line, pt):
    p1, p2 = line
    vec = p2 - p1
    return np.linalg.norm(np.cross(vec, line - pt)) / np.linalg.norm(vec)


class ConicFitEvaluator:

    def __init__(self, pts: np.ndarray, thresh: float = .05):
        self.norm_factor2 = 1 / np.var(pts, axis=0).min()
        self.thresh2 = thresh ** 2
        self.pts = pts
        self._compare_by = ['inl', 'xse']
        self._metrics = {
            'inl': {'higher_is_better': True, 'rtol': .0},
            'inl_count': {'higher_is_better': True, 'rtol': .0},
            'mse': {'higher_is_better': False, 'rtol': .001},
            'xse': {'higher_is_better': False, 'rtol': .001},
            '_e': {'higher_is_better': True, 'rtol': .0},
        }

    def eval(self, conic: Conic):
        norm_dists2, nearest_t = conic.squared_dists(self.pts)
        norm_dists2 *= self.norm_factor2
        inliers = norm_dists2 < self.thresh2
        inlier_count = int(np.sum(inliers))
        mse = np.mean(norm_dists2)
        xse = np.max(norm_dists2)
        result = {
            'mse': mse,
            'xse': xse,
            'inl': float(inlier_count) / len(self.pts),
            'inl_count': inlier_count,
            '_e': conic.eccentricity(),
            '_t': nearest_t,
            '_inl_mask': inliers,
        }
        result['str'] = 'mse={mse:.3f}, xse={xse:.3f}, inl={inl:.2f}'.format(**result)
        return result

    def is_better(self, s1: dict, s2: dict):
        """ check if s1 score is better than s2 """
        for metric in self._compare_by:
            dff = s1[metric] - s2[metric]
            avg = (s1[metric] + s2[metric]) / 2
            if abs(dff) > abs(avg) * self._metrics[metric]['rtol']:
                return self._metrics[metric]['higher_is_better'] == (dff > 0)
        return False


def fit_conic_ransac(pts: np.ndarray, fit_indexes: np.ndarray[int] = None,
                     inlier_p_thresh: float = .95, seed: int = 1, thresh: float = .05, n: int = 7,
                     max_itrs: int = 500, kind: str = None, resample_factor: float = 1) -> tuple[Conic, dict]:

    evaluator = ConicFitEvaluator(pts=pts, thresh=thresh)

    bests = {}

    def _evaluate(conic: Conic):
        return evaluator.eval(conic)

    def _process_conic(conic, sample):
        new_best = False
        fiteval = _evaluate(conic)
        if conic.kind not in bests or evaluator.is_better(fiteval, bests[conic.kind]['eval']):
            bests[conic.kind] = {'conic': conic, 'sample': sample, 'eval': fiteval}
            new_best = True
        return new_best

    if resample_factor != 1:
        assert fit_indexes is None
        pts_for_fitting = polytools.uniform_resample(pts, kind='cubic', n=int(.5 + resample_factor * len(pts)))[0]
    else:
        pts_for_fitting = pts

    if fit_indexes is None:
        fit_indexes = np.arange(len(pts_for_fitting))

    rng = np.random.default_rng(seed)

    conic = ConicParabola.from_coeffs(_fit_parabola_coeffs(*pts_for_fitting.T))
    _process_conic(conic, sample=fit_indexes)

    for itr in range(max_itrs):
        sample = rng.permutation(fit_indexes)[:n]
        conic = get_conic(_fit_lsqr_coeffs(*pts_for_fitting[sample].T))
        new_best = _process_conic(conic, sample=sample)
        if new_best and conic.kind != 'p':
            conic = ConicParabola.from_coeffs(_fit_parabola_coeffs(*pts_for_fitting[sample].T))
            _process_conic(conic, sample=sample)
        if bests[conic.kind]['eval']['inl'] >= inlier_p_thresh:
            print(itr)
            break

    if 'e' in bests:
        sample = bests['e']['sample']
        conic = ConicParabola.from_coeffs(_fit_parabola_coeffs(*pts_for_fitting[sample].T))
        _process_conic(conic, sample=sample)

    if kind is not None:
        chosen_kind = kind
    else:
        kinds = list(bests)
        chosen_kind = kinds[0]
        for kind in kinds[1:]:
            if evaluator.is_better(bests[kind]['eval'], bests[chosen_kind]['eval']):
                chosen_kind = kind

    chosen_conic = bests[chosen_kind]['conic']
    chosen_eval = bests[chosen_kind]['eval']
    ordered_thetas = chosen_eval['_t'][[0, len(pts) // 2, -1]]

    if chosen_conic.kind == 'e':
        if abs(average_theta(ordered_thetas)) > np.pi / 2:
            chosen_conic.ang += 180
            ordered_thetas += np.pi
        set_ellipse_bounds_from_trajectory(chosen_conic, ordered_thetas=ordered_thetas)
        if chosen_conic.bounds[0] > 180:
            chosen_conic._bounds = chosen_conic.bounds[0] - 360, chosen_conic.bounds[1] - 360
    else:
        assert chosen_conic.kind == 'p'
        set_parabola_bounds_from_trajectory(chosen_conic, ordered_thetas=ordered_thetas)

    scores = {k: v for k, v in chosen_eval.items() if not k.startswith('_')}
    return chosen_conic, scores


def _fit_parabola_coeffs(x, y, n_itrs: int = 3, refine_factor: int = 10, bounds=(0, 360)):

    def _brutforce_fit(thetas):
        errors = np.zeros_like(thetas, float)
        params = np.zeros((len(thetas), 3), float)
        for i, theta in enumerate(thetas):
            xx, yy = rotate_points(x, y, rad=-theta)
            params[i] = np.polyfit(xx, yy, deg=2)
            errors[i] = np.sum(np.square(np.polyval(params[i], xx) - yy))
        best_i = np.argmin(errors)
        return thetas[best_i], params[best_i]

    thetas = np.radians(np.arange(*bounds))
    for _ in range(n_itrs):
        best_theta, best_params = _brutforce_fit(thetas)
        step_sz = (thetas[-1] - thetas[0]) / (len(thetas) - 1)
        thetas = np.linspace(best_theta - step_sz, best_theta + step_sz, refine_factor)

    A, D, F = best_params
    coeffs = conic_coeffs.rotate((A, 0., 0., D, -1., F), best_theta)
    return coeffs


def _fit_lsqr_coeffs(x, y):
    D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T
    _, _, V = np.linalg.svd(D)
    return V[-1, :]


def set_ellipse_bounds_from_trajectory(conic: ConicEllipse, pts=None, ordered_thetas=None):
    if pts is not None:
        assert ordered_thetas is None
        ordered_thetas = conic.nearest_t(pts[[0, len(pts) // 2, -1]], refine=True)
    t1, t2, t3 = np.mod(ordered_thetas, 2 * np.pi)
    d = t3 - t1
    if is_ccw([t1, t2, t3]):
        d = d if d >= 0 else 2 * np.pi + d
    else:
        d = d - 2 * np.pi if d > 0 else d
    conic._bounds = conic.t_to_p([t1, t1 + d])


def set_parabola_bounds_from_trajectory(conic: ConicParabola, ordered_thetas=None):
    conic._bounds = conic.t_to_p(ordered_thetas[[0, -1]])


if __name__ == "__main__":

    rng = np.random.default_rng(1)
    gt_conic = ConicParabola(-3, loc=(-1, 2), ang=50, bounds=[-4, 4])
    pts = gt_conic.parametric_pts(n=100)
    pts[:,1] *= -1
    pts += np.std(pts, axis=0) * .1 * rng.standard_normal(size=pts.shape)

    evaluator = ConicFitEvaluator(pts)
    fitted_conic = ConicParabola.from_coeffs(_fit_parabola_coeffs(*pts.T))
    fitted_conic, _ = fit_conic_ransac(pts, kind='p')

    ev = evaluator.eval(fitted_conic)
    ii = ev['_inl_mask']
    t = ev['_t']
    plt.plot(*pts.T, 'k.', label='GT: ' + str(gt_conic))

    plt.plot(*fitted_conic.parametric_pts().T, 'r.', label='Fit: ' + str(fitted_conic))
    #plt.plot(*pts[ii].T, 'c.', label=None)
    plt.plot(*fitted_conic.parametric_pts(t=t).T, 'c.', label='Fit: ' + str(fitted_conic))
    plt.title("Eval: " + ev['str'])
    plt.axis('equal')
    plt.legend()
    plt.show()
