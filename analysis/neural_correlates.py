import pandas as pd
from analysis import cv_results_mgr
from analysis.data_manager import DataMgr
from analysis.neural_population import NeuralPopulation, NEURAL_POP
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.font_manager import FontProperties
from common.utils import sigproc
from itertools import product
from sklearn.metrics import balanced_accuracy_score
from common.utils import plotting
from common.utils.stats import Inliers
from motorneural import draw_data
from scipy.stats import pearsonr, spearmanr


def corrscore(x, y, kind: str = 'p', alpha: float = .05):
    if kind == 'p':
        crr = pearsonr(x, y)
    elif kind == 's':
        crr = spearmanr(x, y)
    else:
        raise ValueError("Unknown kind")
    score = np.abs(crr.statistic) if crr.pvalue < alpha else .0
    return {'score': score, 'pval': crr.pvalue, 'stat': crr.statistic}


def correlation_scores(model_file, smooth_dur: float = .02):

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    cfg = cfg.get_as_eval_config()
    data_mgr = DataMgr(cfg.data, persist=True)

    neural_pop = NeuralPopulation.from_model(model_file)
    neurons = neural_pop.neurons(NEURAL_POP.MINORITY)
    kin_vars = get_kin_vars(kins='Spd,Crv')
    trials, meta = data_mgr.load_trials()

    scores = np.zeros((len(neurons), len(kin_vars), len(trials)), float)
    kin_traces = None
    neural_traces = None
    for tr in trials:
        print(tr)

        if kin_traces is None:
            kin_traces = {kin: tr.get_processed(kin, smooth_dur) for kin in kin_vars}
            neural_traces = {neuron: tr.get_processed(neuron, smooth_dur) for neuron in neurons}
        else:
            kin_traces = {kin: np.r_[tr.get_processed(kin, smooth_dur), v]
                          for kin, v in kin_traces.items()}
            neural_traces = {neuron: np.r_[tr.get_processed(neuron, smooth_dur), v]
                             for neuron, v in neural_traces.items()}

        #
        # for i, (neuron, neural_trace) in enumerate(neural_traces.items()):
        #     for j, (kin, kin_trace) in enumerate(kin_traces.items()):
        #         scores[i, j, tr.ix] = corrscore(neural_trace, kin_trace, kind='s')['score']

    df = []
    from common.utils import stats
    for i, neuron in enumerate(neurons):
        for j, kin in enumerate(kin_vars):
            df.append({'neuron': neural_pop.dispname(neuron), 'kin': kin,
                       'corr': corrscore(np.random.default_rng(1).permutation(neural_traces[neuron]), kin_traces[kin], kind='p')['score']})

    # for i, neuron in enumerate(neurons):
    #     for j, kin in enumerate(kin_vars):
    #         df.append({'neuron': neural_pop.dispname(neuron), 'kin': kin, **stats.calc_stats(scores[i, j])})
    df = pd.DataFrame(df)

    for val in ['avg', 'med']:
        print(val + ":")
        print(df.pivot(index='neuron', columns='kin', values=val))

    print("..")


def neural_segments(model_file):

    # -----
    extrema_r_dur = .1
    smooth_dur = .05
    extrema_width_rheight = .95
    nmax = 2
    # -----

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    cfg = cfg.get_as_eval_config()
    data_mgr = DataMgr(cfg.data, persist=True)

    neural_pop = NeuralPopulation.from_model(model_file)
    neurons = neural_pop.neurons(NEURAL_POP.MINORITY) + neural_pop.neurons(NEURAL_POP.MAJORITY, 2, 'm')

    GEOMS = ['Af', 'Eu', 'Sa']
    kin_vars = []
    kin_vars += [f'{g}Spd' for g in GEOMS]
    kin_vars += [f'{g}Arc' for g in GEOMS]
    #kin_vars += [f'{g}Crv' for g in GEOMS]
    #kin_vars = ['EuArc', 'SaArc', 'AfArc']
    #kin_vars += [f'lg.{k}' for k in kin_vars]

    trials, meta = data_mgr.load_trials()
    ext_det = sigproc.ExtremaDetector(r=extrema_r_dur / trials[0].bin_size,
                                      width_rheight=extrema_width_rheight, nmax=nmax)

    segs = []
    for tr in trials:
        print(tr)
        neural_traces = {neuron: tr.get_processed(neuron, smooth_dur) for neuron in neurons}
        kin_traces = {kin: tr.get_processed(kin, smooth_dur=.0) for kin in kin_vars}
        for neuron, neural_trace in neural_traces.items():
            for (_, l, r) in ext_det.peaks(neural_trace, show=False):
                seg = {'neuron': neuron, 'dur': (r - l) * tr.bin_size}
                seg.update({kin: (kin_trace[r] - kin_trace[l]) if kin.endswith('Arc') else kin_trace[l:r].mean()
                            for kin, kin_trace in kin_traces.items()})
                segs.append(seg)

    from scipy.stats import pearsonr, spearmanr
    segs = pd.DataFrame(segs)
    snrs = np.zeros((len(neurons), len(kin_vars)), float)
    corr_stats = np.zeros((len(neurons), len(kin_vars)), float)
    corr_pvals = np.zeros((len(neurons), len(kin_vars)), float)

    inliers = Inliers('iqr')
    #inliers = None

    metrics = []
    for neuron in neurons:
        dur_ = segs.loc[segs['neuron'] == neuron, 'dur'].to_numpy()
        for kin in kin_vars:
            kin_prop = segs.loc[segs['neuron'] == neuron, kin].to_numpy()

            dur = dur_
            if inliers is not None:
                mask = inliers.is_inlier(kin_prop)
                kin_prop = kin_prop[mask]
                dur = dur_[mask]

            corr = pearsonr(dur, kin_prop)

            snr = (np.mean(kin_prop) / np.std(kin_prop)) ** 2
            s = f'corr:{corr.statistic:2.2f}({corr.pvalue:2.2f}), snr:{snr:2.2f}'
            metrics.append({'neuron': neural_pop.dispname(neuron), 'kin': kin, 'corr_stat': corr.statistic,
                            'corr_pval': corr.pvalue, 'snr': snr, 'summary': s})

    df = pd.DataFrame(metrics).pivot(index='neuron', columns='kin', values='summary')
    print(df.to_string())

    #
    # snrs = pd.DataFrame(snrs, columns=kin_vars, index=neurons)
    # corr_stats = pd.DataFrame(corr_stats, columns=kin_vars, index=neurons)
    # corr_pvals = pd.DataFrame(corr_pvals, columns=kin_vars, index=neurons)
    #
    # _, axs = plotting.new_subplots(ncols=len(kin_vars), nrows=len(neurons))
    # for i, neuron in enumerate(neurons):
    #     for j, kin in enumerate(kin_vars):
    #         v = segs.loc[segs['neuron'] == neuron, kin].to_numpy()
    #         v = inliers.filter(v)
    #         axs[i, j].hist(v)
    # plotting.set_outter_labels(axs, t=kin_vars, y=neurons)
    # plt.show()






def extrema_match_scores(extremas1: dict[str, np.ndarray[int]],
                         extremas2: dict[str, np.ndarray[int]],
                         names=None):
    names = ['attr1', 'attr2'] if names is None else names
    assert len(names) == 2
    records = []
    for attr1, extr1 in extremas1.items():
        n_peaks1 = extr1.sum()
        for attr2, extr2 in extremas2.items():
            n_peaks2 = extr2.sum()
            if n_peaks1 == 0 or n_peaks2 == 0:
                score = float('-inf')
            else:
                score = balanced_accuracy_score(extr1, extr2)
            records.append({'score': score, names[0]: attr1, names[1]: attr2})
    return pd.DataFrame(records)


def neuron_vs_kin_extrema_match(model_file, smooth_dur: float = .02):

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    cfg = cfg.get_as_eval_config()
    data_mgr = DataMgr(cfg.data, persist=True)

    neural_pop = NeuralPopulation.from_model(model_file)
    neurons = neural_pop.neurons(NEURAL_POP.MINORITY)
    kin_vars = ['EuSpd', 'EuAcc', 'lg.EuCrv', 'lg.SaCrv', 'lg.AfCrv', 'lg.AfSpd', 'lg.SaSpd']
    kin_vars = [k.replace('lg.', '') for k in kin_vars]
    trials, meta = data_mgr.load_trials()

    r_dur = .5
    ext_det = sigproc.ExtremaDetector(r=r_dur / trials[0].bin_size)

    bin_dur = .5
    win_sz = round(int(bin_dur / trials[0].bin_size))

    dfs = []
    neural_extremas = []
    kin_extremas = []
    for tr in trials:
        print(tr)

        kin_traces = {kin: tr.get_processed(kin, smooth_dur) for kin in kin_vars}
        neural_traces = {neuron: tr.get_processed(neuron, smooth_dur) for neuron in neurons}

        neural_extremas.append(np.array([ext_det.maximas(v, asmask=True) for k, v in neural_traces.items()]))
        kin_extremas.append(np.array([ext_det.extremas(v) == 1 for k, v in kin_traces.items()]))

        neural_extremas[-1] = sigproc.nonoverlap_reduce(neural_extremas[-1], win_sz=win_sz, axis=1, reduce='any')[0]
        kin_extremas[-1] = sigproc.nonoverlap_reduce(kin_extremas[-1], win_sz=win_sz, axis=1, reduce='any')[0]

        #
        # for attr, ext in neural_extremas.items():
        #     if len(set(ext)) == 1:
        #         #ext_det.maximas(neural_traces[attr], show=True)
        #         plt.plot(neural_traces[attr])
        #         plt.title(attr)
        #         plt.show()

        #scores = extrema_match_scores(neural_extremas, kin_extremas, names=['neural', 'kin'])
        #scores['trial'] = tr.ix
        #dfs.append(scores)

    scores = np.zeros((len(neurons), len(kin_vars)), float)
    neural_extremas = np.concatenate(neural_extremas, axis=1)
    kin_extremas = np.concatenate(kin_extremas, axis=1)
    for i, neuron in enumerate(neurons):
        for j, kin in enumerate(kin_vars):
            scores[i, j] = balanced_accuracy_score(neural_extremas[i], kin_extremas[j])
    scores = pd.DataFrame(scores, columns=kin_vars, index=neurons)
    print(scores.mean(axis=0))


def get_kin_vars(geoms: str | list[str] = 'Eu,Sa,Af', kins: str | list[str] = 'Spd,Acc',
                 aslog=('AfSpd', 'EuCrv', 'AfCrv', 'SaCrv')):

    def _parse(lst):
        if isinstance(lst, str): lst = lst.split(',')
        return [s.capitalize() for s in lst]
    kin_vars = []
    for geom, kin in product(_parse(geoms), _parse(kins)):
        if kin == 'Acc' and geom != 'Eu':
            continue
        kin_var = f'{geom}{kin}'
        if kin_var in aslog:
            kin_var = f'lg.{kin_var}'
        kin_vars.append(kin_var)
    return kin_vars


def show_correlates_for_neuron(model_file):

    # -----
    markers_list = ['1', '2', '3', '4', '5', '6']
    markers_list = [f'${m}$' for m in markers_list]
    smooth_dur = .01

    extrema_r_dur = .1
    extrema_width_rheight = .99
    # -----

    rng = np.random.default_rng(1)

    def _draw_event_markers(ax_or_lines, events):
        def neuron_line_width(neuron): return len(neurons) - neurons.index(neuron)
        try:
            lines = ax_or_lines.get_lines()
        except AttributeError:
            lines = ax_or_lines

        lines = [line for line in lines if len(line.get_xdata()) > 1]
        for i, line in enumerate(lines):
            for (neuron, (ix, l, r), color, marker) in events:
                x, y = line.get_data()
                line.axes.plot(x[l:r], y[l:r], color=color, lw=neuron_line_width(neuron))
                line.axes.plot(x[ix], y[ix], color=color, marker=marker, markeredgecolor='w',
                               markersize=8, markeredgewidth=.5)

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    cfg = cfg.get_as_eval_config()
    data_mgr = DataMgr(cfg.data, persist=True)

    neural_pop = NeuralPopulation.from_model(model_file)
    #neural_pop.draw()
    neurons = neural_pop.neurons(NEURAL_POP.MINORITY, n=2, ranks='t')
    kin_vars = get_kin_vars(kins='Spd,Crv')

    trials, _ = data_mgr.load_trials()
    ext_det = sigproc.ExtremaDetector(r=extrema_r_dur / trials[0].bin_size,
                                      width_rheight=extrema_width_rheight, nmax=len(markers_list))

    for tr in trials:

        kin_traces = {kin: tr.get_processed(kin, smooth_dur) for kin in kin_vars}
        neural_traces = {neuron: tr.get_processed(neuron, smooth_dur) for neuron in neurons}

        trace_axs_, traj_ax = plotting.make_split_grid(len(kin_vars) + 1)
        neuron_ax, kin_axs = trace_axs_[0], trace_axs_[1:]
        plt.suptitle(str(tr))

        line_objs = []
        events = []
        for neuron in neurons:
            label = f'{neuron} {neural_pop.normalized_weight[neuron]:2.1f}'
            v = tr.get_processed(neuron, smooth_dur=smooth_dur)
            p = neuron_ax.plot(v, label=label, lw=1)
            markers = cycle(markers_list)
            events_ = [(neuron, (ix, l, r), p[0].get_color(), next(markers))
                       for (ix, l, r) in ext_det.peaks(v)]
            events += events_
            line_objs += p
            _draw_event_markers(p, events_)
        fontp = FontProperties()
        fontp.set_size(8)
        neuron_ax.legend(loc="center right", bbox_to_anchor=(.01, .5), prop=fontp, title='Neurons')

        tm_tick = .5
        tick_labels = np.arange(0, tr.duration, tm_tick)
        if tr.duration - tick_labels[-1] > tm_tick / 4:
            tick_labels = np.append(tick_labels, tr.duration)
        tick_inds = tick_labels / tr.bin_size
        tick_labels = [f'{round(tk, 2)}' for tk in tick_labels]
        neuron_ax.set_xticks(ticks=tick_inds, labels=tick_labels)

        for ax, kin in zip(kin_axs, kin_vars):
            line_objs += ax.plot(tr.get_processed(kin, smooth_dur=smooth_dur), label=kin, lw=1, color='k')
            _draw_event_markers(ax, events)
            ax.set_ylabel(kin)
            ax.set_xticks(ticks=tick_inds, labels=tick_labels)

        line_objs += traj_ax.plot(*tr.kin.X.T, lw=2, color='k')
        draw_data.draw_kin_events(tr, ax=traj_ax)
        traj_ax.plot(*tr.kin.X[0], marker='+', color='k')
        _draw_event_markers(traj_ax, events)

        plotting.vlines([ind for evnt, ind in tr.kin.events.items() if evnt.startswith('hit')],
                        ax=neuron_ax, color='r', lw=1)

        plotting.SyncedIndexMarker(markersize=10, marker='*').attach(plt.gcf(), line_objs=line_objs)
        plt.show()


if __name__ == "__main__":
    file = "/Users/davidu/geometric-neurons/outputs/models/TP_RS bin10 lag100 dur200 affine-kinX-nmahal f70c5c.Fold0.pth"
    #draw_trajectories_grouped_by_embedded_dist(file)
    #neuron_vs_kin_extrema_match(file)
    show_correlates_for_neuron(file)
    #correlation_scores(file)
