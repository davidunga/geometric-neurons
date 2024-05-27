from typing import Type
import numpy as np
from common.utils.conics.conic_coeffs import get_strict_conic_type
from common.utils.conics.api import ellipse_api, parabola_api
from common.utils.conics.api.conics_api import conic_api
from common.utils.conics.api.ellipse_api import ellipse_api
from common.utils.conics.api.parabola_api import parabola_api


def digest_and_validate_conic_params(m, loc, theta):
    kind = 'p'
    if hasattr(m, '__len__'):
        assert len(m) == 2
        if m[1] is None:
            m = m[0]
        else:
            kind = 'e'
    if kind == 'e':
        assert m[0] >= m[1], f"Major axis is smaller minor: {m}"
    loc = np.asarray(loc, float)
    return kind, (m, loc, theta)


def get_conic_api(kind: str) -> Type[conic_api]:
    if kind == 'e':
        return ellipse_api
    elif kind == 'p':
        return parabola_api
    else:
        raise ValueError('Unknown conic kind ' + str(kind))

