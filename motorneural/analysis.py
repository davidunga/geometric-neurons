import matplotlib.pyplot as plt
from motorneural.data import Data
from motorneural.typetools import *


def calc_event_triggered_response(data: Data, event="max_spd", tradius=Pair[float],
                                  group_by=None, shuff=False) -> Tuple[NpMatrix, NpVec, list]:
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

    bins_before = int(.5 + tradius[0] / data.bin_sz)
    bins_after = int(.5 + tradius[1] / data.bin_sz)
    bin_span = bins_before + bins_after + 1

    trial_group = [tr[group_by] for tr in data] if group_by is not None else [0 for _ in data]
    groups = list(set(trial_group))
    visitcounts = np.zeros((data.num_neurons, bin_span, len(groups)), int)
    spkcounts = np.zeros((data.num_neurons, bin_span, len(groups)), float)
    for ix, tr in enumerate(data):
        bin_start = max(0, tr[event] - bins_before)
        bin_stop = min(tr[event] + bins_after + 1, tr.num_samples)
        s = tr.neural.spkcounts[:, bin_start: bin_stop]
        if shuff:
            s = s[:, np.random.permutation(s.shape[1])]
        ifm = bins_before - (tr[event] - bin_start)
        ito = ifm + s.shape[1]
        spkcounts[:, ifm: ito, groups.index(trial_group[ix])] += s
        visitcounts[:, ifm: ito, groups.index(trial_group[ix])] += 1

    spkcounts /= np.maximum(visitcounts, 1)
    tm_to_event = np.linspace(-bins_before, bins_after, spkcounts.shape[1]) * data.bin_sz
    return spkcounts, tm_to_event, groups if group_by is not None else [None]


def plot_event_triggered_response(data: Data, **kwargs):

    spkcounts, tm_to_event, groups = calc_event_triggered_response(data, **kwargs)
    assert spkcounts.shape == (data.num_neurons, len(tm_to_event), len(groups))

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


def plot_trajectories(data: Data, group_by=None, event_markers=None):

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
