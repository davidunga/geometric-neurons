from . conic import Conic
from . conic_parabola import ConicParabola
from . conic_ellipse import ConicEllipse
from . conic_coeffs import ConicTypeError
import numpy as np


def get_conic(coeffs=None, **kwargs) -> Conic:
    if coeffs is not None:
        assert not kwargs
        conic = _get_conic_by_coeffs(coeffs)
    else:
        conic = _get_conic_by_kws(**kwargs)
    return conic


def _validate_and_prep_conic_params(*, m, loc, ang, bounds=None):
    if hasattr(m, '__len__'):
        assert len(m) == 2
        m = float(m[0]), float(m[1])
    else:
        m = float(m)
    loc = np.asarray(loc, float)
    ang = float(ang)
    assert len(loc) == 2
    if bounds is not None:
        assert len(bounds) == 2
        # lb, ub = bounds
        # if ub - lb > 180:
        #     delta = ub - lb
        #     lb = ub
        #     ub = lb + delta
        #     bounds = lb, ub
        bounds = tuple(bounds)
    return m, loc, ang, bounds


def _get_conic_by_kws(**kwargs) -> Conic:
    m, loc, ang, bounds = _validate_and_prep_conic_params(**kwargs)
    if isinstance(m, tuple):
        conic = ConicEllipse(m=m, loc=loc, ang=ang, bounds=bounds)
    else:
        conic = ConicParabola(m=m, loc=loc, ang=ang, bounds=bounds)
    return conic


def _get_conic_by_coeffs(coeffs) -> Conic:
    try:
        conic = ConicEllipse.from_coeffs(coeffs)
    except conic_coeffs.ConicTypeError:
        conic = ConicParabola.from_coeffs(coeffs)
    return conic
