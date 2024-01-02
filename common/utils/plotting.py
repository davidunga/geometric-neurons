from screeninfo import get_monitors, Monitor
import matplotlib


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
