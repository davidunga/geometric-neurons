from screeninfo import get_monitors, Monitor
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pylab
from itertools import product
from dataclasses import dataclass


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


def set_labels(axs, x=None, y=None):
    """
    set x & y labels of multiple axes
    """
    assert axs.ndim == 1, "Currently only supports 1d axis array"
    for x_or_y, labels in (('x', x), ('y', y)):
        if isinstance(labels, str):
            labels = [labels] * len(axs)
        assert len(axs) == len(labels)
        for ax, label in zip(axs, labels):
            if x_or_y == 'x':
                ax.set_xlabel(label)
            else:
                ax.set_ylabel(label)


def set_outter_labels(axs, x: str | list = None, y: str | list = None):
    if axs.ndim == 1:
        axs = axs.reshape(1, -1)
    for ax in axs:
        ax.set_xlabel(None)
        ax.set_ylabel(None)
    set_labels(axs[-1, :], x=x)
    set_labels(axs[:, 0], y=y)


def get_fig_boox(fig):
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
        bbox = get_fig_boox(fig)
        w = h = scale * min(monitor.width / bbox.width, monitor.height / bbox.height) / fig.dpi
    else:
        raise ValueError("Unknown aspect type")

    set_figsize(fig, w, h, units='screen')
