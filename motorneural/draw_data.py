import matplotlib.pyplot as plt
import numpy as np
from . typetools import *
from . data import DataSlice, Trial


def draw_neural_rates(s: DataSlice, trace: bool = True):

    rng = np.random.default_rng(1)
    plt.subplots(figsize=(12, 8))

    num_neurons = s.neural.num_neurons
    rates = s.neural[:].T  # rows = neuron, cols = time bin
    nrow, ncol = 6, 5
    h, w = nrow - 2, ncol - 1

    # Raster plot
    ax1 = plt.subplot2grid((nrow, ncol), (0, 0), colspan=w, rowspan=h)
    ax1.imshow(rates, aspect='auto', origin='lower', cmap='gray')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Neuron')
    plt.title(str(s))

    # event lines
    for i, (event_name, event_ix) in enumerate(s.events.items()):
        p = ax1.plot([event_ix, event_ix], [0, num_neurons - 1])
        x = max(1, min(event_ix, .98 * rates.shape[1]))
        y = 5 + i * num_neurons / len(s.events)
        plt.text(x, y, event_name, color=p[0].get_color())

    # Average per neuron
    ax2 = plt.subplot2grid((nrow, ncol), (0, w), rowspan=h, sharey=ax1)
    ax2.plot([0, 0], [0, num_neurons - 1], color='gray')
    ax2.plot(np.mean(rates, axis=1), range(num_neurons), 'b-')
    ax2.set_title('Avg over time')
    ax2.invert_yaxis()  # Invert y-axis to align with the left plot

    # Population average vs time
    ax3 = plt.subplot2grid((nrow, ncol), (h, 0), colspan=h, sharex=ax1)
    ax3.plot(np.mean(rates, axis=0), 'b-')
    ax3.set_title('Avg over population')
    ax3.set_xlabel('Time')

    # Speed profile
    ax4 = plt.subplot2grid((nrow, ncol), (nrow - 1, 0), colspan=ncol - 1, sharex=ax1)
    ax4.plot(s.kin['EuSpd'], 'r-')
    ax4.set_title('Speed profile')
    ax4.set_xlabel('Time')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.92)


def calc_event_triggered_response(data: list[Trial], event="max_spd", tradius=Pair[float],
                                  group_by=None, shuff=False) -> tuple[NpMatrix, NpVec, list]:
    """
    Calc event-triggered neural response relative to [event], conditioned over [group_by] property
    Args:
        data:
        event: event name
        tradius: time radius [before, after] event
        group_by: property to group by, i.e., response is computed per group
        shuff: shuffle spike counts
    Returns:
        spkcounts - average spike counts matrix: neruons x time-bins x groups
        tms - vector, tms[i] is the time-to-event at bin i
    """

    bin_sz = data[0].bin_size
    num_neurons = data[0].neural.num_neurons

    bins_before = int(.5 + tradius[0] / bin_sz)
    bins_after = int(.5 + tradius[1] / bin_sz)
    bin_span = bins_before + bins_after + 1

    trial_group = [tr[group_by] for tr in data] if group_by is not None else [0 for _ in data]
    groups = list(set(trial_group))
    visitcounts = np.zeros((num_neurons, bin_span, len(groups)), int)
    spkcounts = np.zeros((num_neurons, bin_span, len(groups)), float)
    for ix, tr in enumerate(data):
        bin_start = max(0, tr[event] - bins_before)
        bin_stop = min(tr[event] + bins_after + 1, len(tr))
        s = tr.neural.spkcounts[:, bin_start: bin_stop]
        if shuff:
            s = s[:, np.random.permutation(s.shape[1])]
        ifm = bins_before - (tr[event] - bin_start)
        ito = ifm + s.shape[1]
        spkcounts[:, ifm: ito, groups.index(trial_group[ix])] += s
        visitcounts[:, ifm: ito, groups.index(trial_group[ix])] += 1

    spkcounts /= np.maximum(visitcounts, 1)
    tm_to_event = np.linspace(-bins_before, bins_after, spkcounts.shape[1]) * bin_sz
    return spkcounts, tm_to_event, groups if group_by is not None else [None]


def plot_event_triggered_response(data: list[Trial], **kwargs):
    spkcounts, tm_to_event, groups = calc_event_triggered_response(data, **kwargs)

    tm_ticks_lbls = [tm_to_event[0], 0, tm_to_event[-1]]
    tm_tick_ixs = np.searchsorted(tm_to_event, tm_ticks_lbls)
    tm_ticks_lbls = [f"{lbl:2.2f}" for lbl in tm_ticks_lbls]
    if groups[0] is not None:
        groups.append(None)
    for group_ix, group in enumerate(groups):
        plt.figure()
        if group_ix < spkcounts.shape[2]:
            img = spkcounts[:, :, group_ix]
        else:
            assert group is None
            img = spkcounts.mean(axis=2)
        plt.imshow(img, cmap="hot")
        plt.xticks(tm_tick_ixs, tm_ticks_lbls)
        plt.plot(np.array([1, 1]) * np.searchsorted(tm_to_event, 0), [0, img.shape[0] - 1], "w")
        plt.title(str(data) + f" [{group if group is not None else 'total average'}]")


def plot_trajectories(data: list[Trial], group_by=None, event_markers=None):
    if event_markers is None:
        event_markers = []
    elif isinstance(event_markers, str):
        event_markers = [event_markers]

    trial_group = [tr[group_by] for tr in data] if group_by is not None else [0 for _ in data]
    groups = list(set(trial_group))
    colors = plt.get_cmap('rainbow', len(groups))
    markers = ["o", "s", "^", "*", "D", "p", "8"]
    lgd = {}
    plt.figure()
    for ix, tr in enumerate(data):
        color = colors(groups.index(trial_group[ix]))
        X = tr.kin.X
        plt.plot(X[:, 0], X[:, 1], color=color)
        for event in event_markers:
            marker = markers[event_markers.index(event) % len(markers)]
            plt.plot(X[tr[event], 0], X[tr[event], 1], color=color, marker=marker)
            lgd[event] = marker

    if group_by is not None:
        plt.title(f"Trajectories" + (f" by {group_by}" if group_by is not None else ""))

    plt.legend([plt.Line2D([0], [0], color="k", marker=marker) for marker in lgd.values()], list(lgd.keys()))


def draw_kin_events(data_slice: DataSlice, ax=None, palette: dict = None, traj: bool = False):

    if ax is None:
        ax = plt.gca()

    if not palette:
        palette = {'hit': 'rX'}

    if traj:
        ax.plot(*data_slice['X'].T, 'k')

    for event, ind in data_slice.kin.events.items():
        event_type = event.split('.')[0]
        if event_type in palette:
            xy = data_slice['X'][ind]
            ax.plot(*xy, palette[event_type])
            ax.annotate(event, xy)

