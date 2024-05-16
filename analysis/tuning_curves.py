from analysis import cv_results_mgr
from analysis.data_manager import DataMgr
from analysis.neural_population import NeuralPopulation, NEURAL_POP
import numpy as np
from analysis.show_embeddings import plot_binned_stats, _new_subplots
from common.utils import stats
import matplotlib.pyplot as plt


def neural_tuning_curve(model_file):

    # -----
    kin_vars = ['AfSpd', 'EuSpd', 'EuAcc']
    stats_plot_kws = {'bins': stats.BinSpec(8, 'p'), 'loc': 'avg', 'band': 'error'}
    inliers = stats.Inliers('percentiles', [0, .75])
    neural_reduce_func = np.mean
    # -----

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    cfg = cfg.get_as_eval_config()
    data_mgr = DataMgr(cfg.data, persist=True)

    # get reduced kinematics per segment
    reduced_kinematics = data_mgr.get_reduced_kinematics(names=kin_vars)

    # get flat neural activity per segment, and the population label of each neuron
    neural_vecs, _ = data_mgr.get_inputs()
    neural_pop = NeuralPopulation.from_model(model_file)

    reduced_activations = {neuron: neural_reduce_func(neural_vecs[:, neural_pop.inputs_mask(neuron)], axis=1)
                           for neuron in neural_pop.neurons(NEURAL_POP.MINORITY)}

    _, axs = _new_subplots(nrows=len(reduced_activations), ncols=len(reduced_kinematics))
    for i, neuron in enumerate(reduced_activations):
        for j, kin in enumerate(reduced_kinematics):
            mask = inliers.is_inlier(reduced_kinematics[kin])
            plot_binned_stats(x=reduced_kinematics[kin][mask], y=reduced_activations[neuron][mask],
                              **stats_plot_kws, idline=False, ax=axs[i, j], counts=False)

    _set_labels_in_grid(axs, t=reduced_kinematics.keys(), y=reduced_activations.keys())

    plt.show()


def _set_labels_in_grid(axs, x=None, y=None, t=None):
    def _set_labels(axs_, func, labels):
        if labels is None: return
        for ax, label in zip(axs_, labels):
            getattr(ax, func)(label)

    _set_labels(axs[-1], 'set_xlabel', x)
    _set_labels(axs[:, 0], 'set_ylabel', y)
    _set_labels(axs[0], 'set_title', t)



if __name__ == "__main__":
    file = "/Users/davidu/geometric-neurons/outputs/models/TP_RS bin10 lag100 dur200 affine-kinX-nmahal f70c5c.Fold0.pth"
    #draw_trajectories_grouped_by_embedded_dist(file)
    neural_tuning_curve(file)
