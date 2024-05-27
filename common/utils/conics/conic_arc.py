import numpy as np
from common.utils import polytools
from geometrik.utils import inflection_points
from common.utils.linalg import rotate_points
import matplotlib.pyplot as plt


def get_approx_arc_properties(pts) -> dict:

    margin = .05
    s_rad = 2 * np.pi

    inflection_ixs = inflection_points(pts)

    # find index of maximal curvature
    s = polytools.arclen(pts)
    t = polytools.winding_angle(pts)
    k = np.gradient(t, s)
    s_start, s_stop = margin * s[-1], (1 - margin) * s[-1]
    start = np.nonzero(s >= s_start)[0][0]
    stop = np.nonzero(s < s_stop)[0][-1] + 1

    curv = np.zeros_like(k)
    curv[start: stop] = np.abs(k[start: stop])
    if len(inflection_ixs):
        inflection_ixs = np.r_[0, inflection_ixs, len(pts) - 1]
        i = np.argmax(np.diff(inflection_ixs))
        convex_start = max(start, inflection_ixs[i])
        convex_stop = max(min(inflection_ixs[i + 1], stop), start + 1)
        curv[:convex_start] = 0
        curv[convex_stop:] = 0

    i0 = int(np.argmax(curv))

    # get index range of the arc
    r = 1 / np.abs(k[i0])
    sr = s_rad * r
    s_start = max(s_start, s[i0] - sr)
    s_stop = min(s_stop, s[i0] + sr)
    start = np.nonzero(s >= s_start)[0][0]
    stop = np.nonzero(s < s_stop)[0][-1] + 1

    m = k[i0] / 2
    dx, dy = pts[i0 + 1] - pts[i0 - 1]
    theta = np.arctan2(dy, dx)

    if t[0] > t[-1]:
        theta += np.pi
        m *= -1

    center = pts[i0] + np.array(rotate_points(0, r, rad=theta))
    props = {'r': r, 'center': center, 'm': m, 'theta': theta,
             'inflections_score': len(inflection_ixs) / len(pts),
             'vertex_ix': i0, 'start_ix': start, 'stop_ix': stop,
             'valid_arclen': s[stop] - s[start], 'total_arclen': s[-1],
             'normalized_valid_arclen': (s[stop] - s[start]) / r,
             'normalized_total_arclen': s[-1] / r}
    return props


def draw_arc_properties(pts, arc_props: dict = None):

    if arc_props is None:
        arc_props = get_approx_arc_properties(pts)

    start = arc_props['start_ix']
    stop = arc_props['stop_ix']
    i0 = arc_props['vertex_ix']
    m = arc_props['m']
    center = arc_props['center']
    r = arc_props['r']
    theta = arc_props['theta']

    ts = np.linspace(-np.pi/2, np.pi/2, 100) + theta - np.pi / 2
    osc_circle_pts = center + r * np.c_[np.cos(ts), np.sin(ts)]

    xx = np.linspace(-r, r, 100)
    yy = m * xx ** 2
    osc_parabola_pts = pts[i0] + np.stack(rotate_points(xx, yy, rad=theta), axis=1)

    plt.plot(*pts.T, 'k.')
    plt.plot(*pts[start: stop].T, 'r.')
    plt.plot(pts[i0, 0], pts[i0, 1], 'c*')
    plt.plot(*osc_circle_pts.T, 'b-')
    plt.plot(*osc_parabola_pts.T, 'g-')
    plt.gca().set_aspect('equal')


