import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from common.utils import dlutils
from config import Config
from data_manager import DataMgr
from common.utils.typings import *
from pathlib import Path
from common.metric_learning.embedding_models import LinearEmbedder
from common.metric_learning import embedding_eval
import cv_results_mgr
from common.utils.inlier_detection import InlierDetector


def extract_model_weights(model: LinearEmbedder) -> NpMatrix:
    return model.embedder.get_submodule('1').weight.detach().numpy()


def identify_subpopulation_by_model_weights(model_file, draw = True):

    # ----
    neuron_weight_agg = 'median'
    inlier_detector = InlierDetector(method='iqr')
    # ----

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    weights = extract_model_weights(model)
    data_mgr = DataMgr(cfg.data)
    inputs, input_names = data_mgr.get_inputs()
    assert len(input_names) == weights.shape[1]
    neuron_name_per_weight_col = np.array([nm.split('.')[0] for nm in input_names])
    neuron_names = sorted(set(neuron_name_per_weight_col))

    neuron_weight_agg_func = getattr(np, neuron_weight_agg)
    neuron_weight = np.zeros(len(neuron_names), float)
    for i, neuron_name in enumerate(neuron_names):
        w = weights[:, neuron_name_per_weight_col == neuron_name]
        neuron_weight[i] = neuron_weight_agg_func(np.abs(w))

    inliers_mask = ~inlier_detector.fit_predict(neuron_weight)
    df = pd.DataFrame({'Neuron Weight': neuron_weight, 'Neuron Name': neuron_names,
                       'Neuron Index': np.arange(len(neuron_names)), 'Population': inliers_mask.astype(int)})

    if draw:
        jp = sns.jointplot(data=df, x='Neuron Index', y='Neuron Weight', hue='Population', kind='scatter',
                           palette={0: 'black', 1: 'red'}, alpha=.75)
        jp.ax_marg_x.remove()
        xlm_margin = 2
        jp.ax_joint.set_xlim([-xlm_margin, len(neuron_names) + xlm_margin - 1])
        plt.show()

    return df



if __name__ == "__main__":
    file = "/Users/davidu/geometric-neurons/outputs/models/TP_RS bin10 lag100 dur200 affine-kinX-nmahal f70c5c.Fold0.pth"
    identify_subpopulation_by_model_weights(file)
