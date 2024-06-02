import numpy as np
from common.utils.linalg import rotate_points
from common.utils import strtools
import matplotlib.pyplot as plt

"""
parameterization:
    t = angular parameter, e.g. a*cos(t), b*sin(t)
    s = arclength
    p = arclength normalized by semi latus
"""


def _unpack_points(pts) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(pts, tuple):
        return pts
    else:
        return pts.T


class Conic:

    _kind_name: str = ''

    def __init__(self, m, loc, ang, bounds=None):
        self.m = m
        self.loc = loc
        self.ang = ang
        self._bounds = bounds

    def to_json(self):
        return {'m': list(self.m) if hasattr(self.m, '__len__') else self.m,
                'loc': list(self.loc),
                'ang': float(self.ang),
                'bounds': list(self._bounds) if self._bounds is not None else 'null'}

    @property
    def kind(self):
        return self.kind_name[0].lower()

    @property
    def kind_name(self):
        return self._kind_name

    @property
    def params(self):
        return self.m, self.loc, self.ang

    @property
    def xang(self):
        return self.ang - 90 if self.kind == 'p' else self.ang

    @property
    def unitvec(self):
        return np.array(rotate_points(1, 0, deg=self.xang))

    @property
    def bounds(self) -> tuple[float, float]:
        if self._bounds is None:
            self._bounds = self.default_bounds()
        return tuple(self._bounds)

    def radcurv_at_vertex(self) -> float:
        return self.semi_latus()

    def arclen_scale_factor(self) -> float:
        return self.radcurv_at_vertex()

    def transform(self, pts):
        x, y = rotate_points(*_unpack_points(pts), deg=self.ang)
        return x + self.loc[0], y + self.loc[1]

    def inv_transform(self, pts):
        x, y = _unpack_points(pts)
        x, y = rotate_points(x - self.loc[0], y - self.loc[1], deg=-self.ang)
        return x, y

    def nearest_p(self, pts, **kwargs) -> np.ndarray[float]:
        return self.t_to_p(self.nearest_t(pts, **kwargs))

    def nearest_t(self, pts, **kwargs) -> np.ndarray[float]:
        return self.squared_dists(pts, **kwargs)[1]

    def nearest_contour_pt(self, pts, **kwargs):
        return self.parametric_pts(t=self.nearest_t(pts, **kwargs))

    def parametric_pts(self, *, p=None, t=None, n: int = 100) -> np.ndarray:
        x, y = self.transform(self.nontransformed_parametric_pts(p=p, t=t, n=n))
        return np.c_[x, y]

    def t_to_p(self, t):
        return self._arclen_convert(t=t) / self.arclen_scale_factor()

    def p_to_t(self, p):
        return self._arclen_convert(s=p * self.arclen_scale_factor())

    def default_bounds(self) -> tuple[float, float]:
        raise NotImplementedError()

    def __str__(self):
        s = strtools.to_str(self.params, f='2.2')
        return f'{self.kind_name} {s} e={self.eccentricity():2.2f} {strtools.to_str(list(self.bounds), f=2.2)}'

    def draw(self, *args, **kwargs):

        contour_pts = self.parametric_pts()

        plt.figure(figsize=(8, 8))

        lines = plt.plot(*contour_pts.T, '.-')
        plt.plot(*contour_pts[0], 'o', color=lines[0].get_color())
        plt.plot(*self.loc, 'b+')

        plt.plot(*self.vertex_pts().T, '*', label='Vertex')

        vec = self.semi_latus() * np.array([self.unitvec[1], -self.unitvec[0]])
        for i, pt in enumerate(self.focus_pts()):
            line = np.array([pt - vec, pt + vec])
            plt.plot(*pt, 'mo', label='Focus' if i == 0 else None)
            plt.plot(*line.T, ':y', label='Latus' if i == 0 else None)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(str(self))
        plt.legend()
        plt.axis('equal')
        plt.grid(True)

    @classmethod
    def from_coeffs(cls, coeffs):
        raise NotImplementedError()

    def coeffs(self) -> tuple[float, ...]:
        raise NotImplementedError

    def eccentricity(self) -> float:
        raise NotImplementedError()

    def semi_latus(self) -> float:
        raise NotImplementedError()

    def focus_dist(self) -> float:
        raise NotImplementedError()

    def focus_pts(self) -> np.ndarray:
        raise NotImplementedError

    def vertex_pts(self) -> np.ndarray:
        raise NotImplementedError()

    def nontransformed_parametric_pts(self, *, p=None, t=None, n: int = 100) -> np.ndarray:
        if p is None and t is None:
            p = np.linspace(*self.bounds, n)
        if p is not None:
            assert t is None
            t = self.p_to_t(p)
        if self.kind == 'e':
            return np.c_[self.m[0] * np.cos(t), self.m[1] * np.sin(t)]
        elif self.kind == 'p':
            return np.c_[t, self.m * t ** 2]
        else:
            raise ValueError()

    def squared_dists(self, pts, refine: bool = False) -> tuple[np.ndarray[float], np.ndarray[float]]:
        raise NotImplementedError

    def _arclen_convert(self, *, t=None, s=None) -> np.ndarray[float]:
        raise NotImplementedError


def debug_draw(conic: Conic):
    conic.draw()
    pts = np.array([(1, -1), (-1.5, 2), (-1.5, 1)], float)
    plt.plot(*pts.T, 'ro')
    for refine in (False, True):
        marker = 'r:' if refine else 'k:'
        projected_pts = conic.nearest_contour_pt(pts, refine=refine)
        for j, (pt, ppt) in enumerate(zip(pts, projected_pts)):
            plt.plot([pt[0], ppt[0]], [pt[1], ppt[1]], marker, label=f'Refine={refine}' if j == 0 else None)
