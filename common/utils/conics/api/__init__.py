from typing import Type
import numpy as np
from common.utils.linalg import rotate_points
from common.utils.conics.conic_coeffs import ConicTypeError
from common.utils.conics.api import ellipse_api, parabola_api
from common.utils.conics.api.conics_api import conic_api
from common.utils.conics.api.ellipse_api import ellipse_api
from common.utils.conics.api.parabola_api import parabola_api


def parabola_to_ellipse(m, loc, ang, e=.9999):
    a = parabola_api.focus_to_vertex_dist(m) / (1 - e)
    b = a * ellipse_api.radii_ratio_from_eccentricity(e)
    dx, dy = rotate_points(a, 0, deg=ang - 90)
    center = np.array([loc[0] - dx, loc[1] - dy])
    return (a, b), center, ang + 90


def get_conic_kind_and_params(coeffs):
    try:
        kind = 'e'
        params = ellipse_api.parameters(coeffs)
    except ConicTypeError:
        kind = 'p'
        params = parabola_api.parameters(coeffs)
    return kind, params


def digest_and_validate_conic_params(m, loc, ang):
    kind = 'p'
    if hasattr(m, '__len__'):
        assert len(m) == 2
        if m[1] is None:
            m = m[0]
        else:
            kind = 'e'
    if kind == 'e':
        assert m[0] >= m[1], f"Major axis is smaller minor: {m}"
    return kind, (m, np.asarray(loc, float), ang)


def get_conic_api(kind: str) -> Type[conic_api]:
    if kind == 'e':
        return ellipse_api
    elif kind == 'p':
        return parabola_api
    else:
        raise ValueError('Unknown conic kind ' + str(kind))

