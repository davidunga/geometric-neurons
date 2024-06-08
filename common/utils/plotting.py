from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import pylab
from screeninfo import get_monitors, Monitor
import seaborn as sns
from common.utils import stats
from typing import Sequence
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse
from sklearn.covariance import MinCovDet
from common.utils import gaussians

_marks_markers = {0: '*', -1: 's'}


def plot(xy, *args, marks='', **kwargs):
    if isinstance(xy, tuple):
        x, y = xy
    else:
        xy = np.asarray(xy)
        if xy.ndim == 1:
            x = np.arange(len(xy))
            y = xy
        else:
            assert xy.ndim == 2
            x, y = xy.T
    p = plt.plot(x, y, *args, **kwargs)
    color = p[0].get_color()
    if marks:
        for mark in marks.split(','):
            i = int(mark)
            plt.plot(x[i], y[i], color=color, marker=_marks_markers[i])


def get_nice_colors() -> list[str]:
    return ['DodgerBlue', 'HotPink', 'LimeGreen', 'Orange', 'Crimson', 'SlateBlue']


def set_axis_equal(ax='gca'):
    ax = get_ax(ax)
    ax.set_aspect('equal', adjustable='box')


def remove_by_label(axs, label):
    for obj in get_objs(axs, label=label):
        obj.remove()


def get_objs(axs, *, kind: str = None, label: str | list[str] = None, one_of: list = None):
    assert kind in (None, 'line', 'marker')

    if not isinstance(axs, list):
        axs = [axs]

    if isinstance(label, str):
        label = [label]

    def _is_match(line) -> bool:
        if kind and (kind != ('marker' if len(line.get_xdata()) == 1 else 'line')):
            return False
        if label is not None and line.get_label() not in label:
            return False
        if one_of is not None and line not in one_of:
            return False
        return True

    ret = []
    for ax in axs:
        ret += [line for line in ax.get_lines() if _is_match(line)]
    return ret


class MplParams:

    RELEVANT_FIELDS = ['legend.fontsize', 'figure.figsize', 'axes.labelsize',
                       'axes.titlesize', 'xtick.labelsize', 'ytick.labelsize']

    SIZES = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']

    @staticmethod
    def _get_smaller_size(size: str):
        ix = MplParams.SIZES.index(size)
        return MplParams.SIZES[max(0, ix)]

    @staticmethod
    def set(size: str = None, figsize=(15, 5)):

        params = {}

        if figsize is not None:
            params['figure.figsize'] = figsize

        if size is not None:
            smaller_size = MplParams._get_smaller_size(size)
            for name in MplParams.RELEVANT_FIELDS:
                if not name.startswith('figure'):
                    params[name] = size if 'title' in name else smaller_size

        pylab.rcParams.update(params)

    @staticmethod
    def reset():
        matplotlib.rcdefaults()


def remove_inner_labels(axs, keep_legend: tuple | str = (0, 0)):
    if axs.ndim == 1:
        axs = axs.reshape(1, -1)
    if not isinstance(keep_legend, str):
        keep_legend = tuple([k if k >= 0 else axs.shape[i] + k for i, k in enumerate(keep_legend)])
    for i, j in product(range(axs.shape[0]), range(axs.shape[1])):
        if i != axs.shape[0] - 1:
            axs[i, j].set_xlabel(None)
        if j != 0:
            axs[i, j].set_ylabel(None)
        if keep_legend != 'all' and keep_legend != (i, j):
            axs[i, j].get_legend().remove()


def set_outter_labels(axs, x: str | list = None, y: str | list = None, t=None):
    if isinstance(axs, dict):
        i0 = y[0]
        j0 = x[0] if x is not None else t[0]
        for i in y:
            axs[(i, j0)].set_ylabel(i)
        if x is not None:
            for j in x:
                axs[(i0, j)].set_xlabel(j)
        else:
            for j in t:
                axs[(i0, j)].set_title(j)
        return
    if x is not None:
        for ax, label in zip(axs[-1, :].flatten(), x):
            ax.set_xlabel(label)
    if y is not None:
        for ax, label in zip(axs[:, 0].flatten(), y):
            ax.set_ylabel(label)
    if t is not None:
        for ax, label in zip(axs[0, :].flatten(), t):
            ax.set_title(label)


def get_fig_bbox(fig):
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    return bbox


def get_primary_monitor() -> Monitor:
    for m in get_monitors():
        if m.is_primary:
            return m


def set_figsize(fig, w, h, units='px'):
    match units:
        case 'px':
            scale_x = scale_y = 1 / fig.dpi
        case 'inch':
            scale_x = scale_y = 1.0
        case 'cm':
            scale_x = scale_y = .3937
        case 'screen':
            monitor = get_primary_monitor()
            scale_x, scale_y = monitor.width, monitor.height
        case _:
            raise ValueError('Unknown units')
    w_inches = w * scale_x / fig.dpi
    h_inches = h * scale_y / fig.dpi
    fig.set_size_inches(w_inches, h_inches, forward=True)


def set_relative_figsize(fig, scale, aspect: str = 'screen'):

    """ Resize the figure based on the screen size and figure's aspect ratio. """

    monitor = get_primary_monitor()
    if aspect == 'screen':
        if isinstance(scale, float):
            w = h = scale
        else:
            assert len(scale) == 2, "Scale must be either scalar or (w,h) pair"
            w, h = scale
    elif aspect == 'keep':
        assert isinstance(scale, float), "Only one scale parameter can be specified for aspect mode 'keep'"
        bbox = get_fig_bbox(fig)
        w = h = scale * min(monitor.width / bbox.width, monitor.height / bbox.height) / fig.dpi
    else:
        raise ValueError("Unknown aspect type")

    set_figsize(fig, w, h, units='screen')


class SyncedIndexMarker:
    """ sync marker across axes & figures """

    def __init__(self, **marker_spec):
        self.marker_spec = {
            'marker': 'D', 'color': 'r', 'markeredgecolor': 'k',
            'markersize': 8, 'label': 'IndexMarker'}
        self.marker_spec.update(marker_spec)
        assert self.marker_spec['label'], "Must have a valid label"
        self.ix = None
        self._attached = []
        self._objs = []

    def _axs(self) -> list[plt.Axes]:
        axs = []
        for obj in self._attached:
            axs += (obj.axes if isinstance(obj, plt.Figure) else [obj])
        return axs

    def _figs(self) -> list[plt.Figure]:
        return list(set(ax.figure for ax in self._axs()))

    def attach(self, fig_or_axs, line_objs: list = None):
        self._objs = line_objs
        self._attached += list(fig_or_axs) if hasattr(fig_or_axs, '__len__') else [fig_or_axs]
        for fig in self._figs():
            fig._markpoint = self
            fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
            fig.canvas.mpl_connect('key_press_event', self.key_press_callback)

    def key_press_callback(self, event):
        if self.ix is None:
            self.ix = .5
        elif event.key == 'left':
            self.ix -= 1
        elif event.key == 'right':
            self.ix += 1
        else:
            return
        self.refresh()

    def button_press_callback(self, event):
        if not event.inaxes:
            return
        for fig in self._figs():
            if fig.canvas.widgetlock.locked():
                return
        min_dist = np.inf
        for line in get_objs(event.inaxes, kind='line', one_of=self._objs):
            dists = (line.get_xdata() - event.xdata) ** 2 + (line.get_ydata() - event.ydata) ** 2
            if np.min(dists) < min_dist:
                self.ix = np.argmin(dists)
                min_dist = dists[self.ix]
        self.refresh()

    def refresh(self):
        remove_by_label(self._axs(), label=self.marker_spec['label'])
        for line in get_objs(self._axs(), kind='line', one_of=self._objs):
            x, y = line.get_data()
            if isinstance(self.ix, float):
                self.ix = int(self.ix * len(x))
            line.axes.plot(x[self.ix], y[self.ix], **self.marker_spec)
        for fig in self._figs():
            fig.canvas.draw()


def make_split_grid(nrows: int):
    """ make subplot where the left side has nrows axis, and the right side is one large axis """
    fig, axs = plt.subplots(nrows, 2, figsize=(16, 8))
    gs = axs[0, 1].get_gridspec()
    for ax in axs[:, 1]:
        ax.remove()
    small_axs = list(axs[:, 0].flatten())
    big_ax = fig.add_subplot(gs[:, 1])
    return small_axs, big_ax


def subplots(nrows: int = 1, ncols: int = 1, ndim: int = 2, figsize=(10, 6), **kwargs):
    assert ndim in (2, 3)
    sns.set_style('darkgrid')
    if ndim == 3:
        kwargs['subplot_kw'] = kwargs.get('subplot_kw', {})
        kwargs['subplot_kw']['projection'] = '3d'
    fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,  **kwargs)
    if not hasattr(axs, '__len__'):
        axs = np.array([axs])
    return axs


#
# def new_subplots(figsize=(10, 6), nrows=1, ncols=1, **kwargs):
#     sns.set_style("darkgrid")
#     fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,  **kwargs)
#     if not hasattr(axs, '__len__'):
#         axs = np.array([axs])
#     return fig, axs


def named_subplots(cols: Sequence = None, rows: Sequence = None, **kwargs) -> dict:

    if cols is None and rows is None:
        raise ValueError('At least one of cols and rows must be provided')

    if isinstance(rows, dict): rows = rows.keys()
    if isinstance(cols, dict): cols = cols.keys()

    if cols is not None and rows is not None:
        keys = product(rows, cols)
    else:
        keys = cols if cols is not None else rows

    axs = subplots(nrows=1 if rows is None else len(rows),
                   ncols=1 if cols is None else len(cols), **kwargs)
    named_axs = dict(zip(keys, axs.flatten()))
    return named_axs


def get_all_figures():
    figures = [plt.figure(i) for i in plt.get_fignums()]
    return figures


def get_ax(ax):
    if isinstance(ax, str):
        if ax == 'gca':
            return plt.gca()
        elif ax == 'new':
            return subplots()[0]
        else:
            raise ValueError('Unknown axis type')
    assert ax is not None
    return ax


def plot_2d_gaussian_ellipse(pts, n_std=2.0, ax='gca', support_fraction: float = .9, **kwargs):
    gauss, inliers_mask = gaussians.gaussian_fit(pts, support_fraction=support_fraction)
    center, (width, height), angle = gaussians.gaussian2d_to_ellipse(gauss, n_std=n_std)
    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle, **kwargs)
    get_ax(ax).add_patch(ellipse)
    return gauss, inliers_mask


def merge_figures():
    figures = get_all_figures()
    combined_fig, axs = plt.subplots(1, len(figures), figsize=(15, 5))

    # Merge all figures into the new figure with subplots
    for i, fig in enumerate(figures):
        for ax in fig.get_axes():
            # Copy each axis from the original figures
            ax.get_shared_x_axes().remove(ax)
            ax.get_shared_y_axes().remove(ax)
            combined_fig._axstack.add(combined_fig._make_key(ax), ax)
            ax.change_geometry(1, len(figures), i + 1)

    # Close the original figures to avoid redundancy
    for fig in figures:
        plt.close(fig)

    # Display the combined figure
    plt.show()


def prep_roc_axis(ax=None):
    if ax is None:
        ax = subplots()[0]
    ax.plot([0, 1], [0, 1], 'k:')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_aspect('equal')

def plot_binned_stats(x, y, bins: stats.BinSpec, loc: str = 'avg', band: str = 'error', idline: bool = True,
                      ax=None, counts: bool = True, **plot_kwargs) -> tuple[plt.Axes, list]:

    if band in ('scale', 'error'):
        scale_name, error_name = stats.get_scale_and_error_names(loc)
        band = scale_name if band == 'scale' else error_name

    stats_ = stats.calc_binned_stats(x, y, stats=['n', loc, band], bins=bins)
    if ax is None:
        ax = subplots()[0]

    stats_['x'] -= np.nanmin(stats_['x'])
    stats_['x'] /= np.nanmax(stats_['x'])

    stats_[loc] -= np.nanmin(stats_[loc])
    stats_[loc] /= np.nanmax(stats_[loc])

    if idline:
        ax.plot(stats_['x'], stats_['x'], ':', lw=.5, color='gray', label=None)


    p = bandplot(stats_['x'], stats_[loc], stats_[band], **plot_kwargs, label=loc.capitalize(),
                 band_kws={'label': f'± {band.upper()}'}, ax=ax)
    if counts:
        for x, y, count in zip(stats_['x'], stats_[loc], stats_['n']):
            plt.text(x, y, str(count), color=p[0].get_color())
    return ax, p


def bandplot(x, y, er, ax=None, band_kws: dict = None, **line_kwargs):
    if ax is None: ax = plt.gca()
    p = ax.plot(x, y, **line_kwargs)
    default_band_kws = {'alpha': .2, 'color': p[0].get_color()}
    if not isinstance(er, tuple):
        er = (er,)
    assert len(er) in (1, 2)
    p_band = ax.fill_between(x, y - er[0], y + er[-1], **(default_band_kws | band_kws))
    p.append(p_band)
    return p



def fill(pts1, pts2, *args, **kwargs):
    x = np.r_[pts1[:, 0], pts2[:, 0][::-1]]
    y = np.r_[pts1[:, 1], pts2[:, 1][::-1]]
    plt.fill(x, y, *args, **kwargs)


def vlines(xs, ax=None, *args, **kwargs):
    ax = plt.gca() if ax is None else ax
    lines = [ax.axvline(x, *args, **kwargs) for x in xs]
    return lines


def hlines(ys, ax=None, *args, **kwargs):
    ax = plt.gca() if ax is None else ax
    lines = [ax.axhline(y, *args, **kwargs) for y in ys]
    return lines


def adjust_line_widths(ax_or_lines, min_width: float = 1, width_step: float = 1):
    try:
        lines = ax_or_lines.get_lines()
    except AttributeError:
        lines = ax_or_lines
    widths = min_width + np.arange(len(lines)) * width_step
    for line, width in zip(lines, widths):
        line.set_linewidth(width)


def get_grid_for_points(pts: np.ndarray, n: int = 500, s: float = 1.1):
    """
    x, y of grid that covers a set of points
    Args:
        pts: points array
        n: number of grid points in each direction
        s: scale factor for the span
    """
    xspan = pts[:, 0].max() - pts[:, 0].min()
    yspan = pts[:, 1].max() - pts[:, 1].min()
    x_mid = (pts[:, 0].max() + pts[:, 0].min()) / 2
    y_mid = (pts[:, 1].max() + pts[:, 1].min()) / 2
    xmin, xmax = x_mid - s * xspan, x_mid + s * xspan
    ymin, ymax = y_mid - s * yspan, y_mid + s * yspan
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    return x, y

