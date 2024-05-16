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


def intra_vs_inter_segment_variance(model_file):


    # -----
    extrema_r_dur = .1
    smooth_dur = .05
    extrema_width_rheight = .95
    nmax = 5
    # -----

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    cfg = cfg.get_as_eval_config()
    data_mgr = DataMgr(cfg.data, persist=True)

    neural_pop = NeuralPopulation.from_model(model_file)
    neurons = neural_pop.neurons(NEURAL_POP.MINORITY) + neural_pop.neurons(NEURAL_POP.MAJORITY, 5, 'm')

    inliers = Inliers('none', .05)

    GEOMS = ['Af', 'Eu', 'Sa']
    kin_vars = []
    kin_vars += [f'{g}Spd' for g in GEOMS]
    kin_vars += [f'{g}Arc' for g in GEOMS]

    trials, meta = data_mgr.load_trials()
    ext_det = sigproc.ExtremaDetector(r=extrema_r_dur / trials[0].bin_size,
                                      width_rheight=extrema_width_rheight, nmax=nmax)

    segs = {f'{neuron}-{kin}': [] for neuron, kin in product(neurons, kin_vars)}
    for tr in trials:
        print(tr)
        neural_traces = {neuron: tr.get_processed(neuron, smooth_dur) for neuron in neurons}
        kin_traces = {kin: tr.get_processed(kin, smooth_dur=.0) for kin in kin_vars}
        for neuron, neural_trace in neural_traces.items():
            for (_, l, r) in ext_det.peaks(neural_trace, show=False):
                #seg = {'neuron': neuron, 'dur': (r - l) * tr.bin_size}
                for kin, kin_trace in kin_traces.items():
                    v = (kin_trace - kin_trace[l]) if kin.endswith('Arc') else kin_trace
                    #v = inliers.filter(v)
                    # seg[f'{kin}.Avg'] = np.mean(v)
                    # seg[f'{kin}.Var'] = np.var(v)
                    # seg[f'{kin}.Sum'] = v.sum()
                    segs[f'{neuron}-{kin}'].append(v)

    metrics = []
    for neuron in neurons:
        for kin in kin_vars:
            vs = segs[f'{neuron}-{kin}']
            vv = np.array([u for v in vs for u in v])
            inliers.fit(vv)
            total_var = np.std(inliers.filter(vv))
            intra_var = np.mean([np.std(inliers.filter(v)) for v in vs])
            metrics.append({'neuron': neural_pop.dispname(neuron),
                            'kin': kin, 'total': total_var, 'intra': intra_var,
                            'score': intra_var / total_var})
    #
    # segs = pd.DataFrame(segs)
    #
    # metrics = []
    # for neuron in neurons:
    #     for kin in kin_vars:
    #         avgs = segs.loc[segs['neuron'] == neuron, f'{kin}.Avg'].to_numpy()
    #         intra_var = segs.loc[segs['neuron'] == neuron, f'{kin}.Var'].to_numpy().mean()
    #         inter_var = np.var(avgs)
    #         metrics.append({'neuron': neural_pop.dispname(neuron),
    #                         'kin': kin, 'inter': inter_var, 'intra': intra_var,
    #                         'score': inter_var / intra_var})

    df = pd.DataFrame(metrics).pivot(index='neuron', columns='kin', values='score')
    print(df.to_string())


if __name__ == "__main__":
    file = "/Users/davidu/geometric-neurons/outputs/models/TP_RS bin10 lag100 dur200 affine-kinX-nmahal f70c5c.Fold0.pth"
    #draw_trajectories_grouped_by_embedded_dist(file)
    #neuron_vs_kin_extrema_match(file)
    intra_vs_inter_segment_variance(file)
    #show_correlates_for_neuron(file)
