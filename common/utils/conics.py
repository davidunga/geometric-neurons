import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from common.utils.conics.api import get_conic_api, digest_and_validate_conic_params
from common.utils.conics import conic_coeffs, conic_arc


class Conic:

    def __init__(self, params=None, bounds: Sequence[float] | None = None,
                 mse: float = None, inlier_count: int = None):
        kind, params = digest_and_validate_conic_params(*params)
        if bounds is None:
            bounds = [0, 2 * np.pi] if kind == 'e' else [0, 2 * np.pi]
        self.api = get_conic_api(kind=kind)
        self.params = params
        self.bounds = bounds
        self.mse = float('nan') if mse is None else mse
        self.inlier_count = inlier_count

    @classmethod
    def ellipse(cls, m, center=(0, 0), ang=0, bounds=None):
        return cls((m, center, ang * np.pi / 180), bounds)

    @classmethod
    def parabola(cls, m, vertex=(0, 0), ang=0, bounds=None):
        return cls((m, vertex, ang * np.pi / 180), bounds)

    @property
    def kind(self):
        return self.api.kind()

    def __str__(self) -> str:
        return self.api.str(*self.params)

    def draw(self, *args, **kwargs):
        t = np.linspace(*self.bounds, 100)
        return self.api.draw(*self.params, t=t, *args, **kwargs)

    def parametric_pts(self, t=None, n: int = 100):
        if t is None:
            t = np.linspace(*self.bounds, n)
        return self.api.parametric_pts(*self.params, t)


def fit_conic_ransac(pts: np.ndarray, fit_indexes: np.ndarray[int] = None,
                     kinds: Sequence[str] = ('e', 'p'),
                     arc: bool = False, seed: int = 1, thresh: float = .5, n: int = 7,
                     n_itrs: int = 1000, e_thresh: bool = .95) -> Conic:

    parabola_fit_kws = {}
    if arc:
        assert fit_indexes is None
        arc_props = conic_arc.get_approx_arc_properties(pts)
        fit_indexes = np.arange(arc_props['start_ix'], arc_props['stop_ix'])
        ang0 = arc_props['theta'] * 180 / np.pi
        parabola_fit_kws['ang_min'] = ang0 - 20
        parabola_fit_kws['ang_max'] = ang0 + 20

    if fit_indexes is None:
        fit_indexes = np.arange(len(pts))

    rng = np.random.default_rng(seed)
    best = {'params': None, 'mse': np.inf, 'inlier_count': 0}
    try_as_parabola = False
    for itr in range(n_itrs):

        if try_as_parabola:
            # try forcing a parabolic fit
            try_as_parabola = False
            coeffs = conic_coeffs.fit_conic_parabola(pts[ii], **parabola_fit_kws)
        else:
            ii = rng.permutation(fit_indexes)[:n]
            coeffs = conic_coeffs.fit_lsqr(*pts[ii].T)
        conic_type = conic_coeffs.get_strict_conic_type(coeffs)

        if conic_type not in kinds:
            continue

        conic_api = get_conic_api(conic_type)
        conic_params = conic_api.parameters(coeffs)
        r = conic_api.semi_latus(conic_params[0])
        norm_dists = conic_api.approx_dist(*conic_params, pts) / r
        inliers = norm_dists < thresh
        inlier_count = sum(inliers)
        mse = np.mean(norm_dists ** 2)
        if (inlier_count > best['inlier_count']) or (inlier_count == best['inlier_count'] and mse < best['mse']):
            best['params'] = conic_params
            best['mse'] = mse
            best['inlier_count'] = inlier_count
            best['_coeffs'] = coeffs
            best['_conic_type'] = conic_type
            best['_inliers'] = inliers

            if conic_type == 'e' and 'p' in kinds:
                try_as_parabola = abs(conic_api.eccentricity(conic_params[0])) > e_thresh

    conic_api = get_conic_api(best['_conic_type'])
    t = conic_api.nearest_parameter(*best['params'], pts)
    best['bounds'] = [min(t), max(t)]
    conic = Conic(**{k: v for k, v in best.items() if not k.startswith('_')})
    return conic


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    conic = Conic.ellipse((2, 1), ang=0, bounds=2 * np.array([-20, 45]) * np.pi / 180)
    pts = conic.parametric_pts()
    pts += .0005 * np.std(pts, axis=0) * rng.standard_normal(size=pts.shape)
    fitted_conic = fit_conic_ransac(pts, kinds=('p','e'), arc=False)
    plt.plot(*pts.T, '.')
    plt.plot(*fitted_conic.parametric_pts().T, 'r-')
    plt.title(str(fitted_conic))
    plt.axis('equal')
    plt.show()