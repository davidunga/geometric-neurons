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


def extract_model_weights(model: LinearEmbedder) -> NpMatrix:
    return model.embedder.get_submodule('1').weight.detach().numpy()


def cluster_neurons_by_model_weight(model_file):
    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    weights = extract_model_weights(model)
    data_mgr = DataMgr(cfg.data)
    inputs, input_names = data_mgr.get_inputs()
    assert len(input_names) == weights.shape[1]
    neuron_name_per_weight_col = np.array([nm.split('.')[0] for nm in input_names])
    neuron_names = sorted(set(neuron_name_per_weight_col))

    for metric in ('norm', 'meanAbs', 'medAbs', 'maxAbs'):
        neuron_weight = np.zeros(len(neuron_names), float)
        for i, neuron_name in enumerate(neuron_names):
            w = weights[:, neuron_name_per_weight_col == neuron_name]
            if metric == 'norm':
                ww = np.linalg.norm(w)
            elif metric == 'meanAbs':
                ww = np.mean(np.abs(w))
            elif metric == 'medAbs':
                ww = np.median(np.abs(w))
            elif metric == 'maxAbs':
                ww = np.min(np.abs(w))
            neuron_weight[i] = ww
        plt.figure()
        data = pd.DataFrame({'weight': neuron_weight, 'neuron': neuron_names})
        ax = sns.scatterplot(data=data, x='neuron', y='weight')
        ax.tick_params(axis='x', labelrotation=90)
        plt.title(metric)
    plt.show()





if __name__ == "__main__":
    file = "/Users/davidu/geometric-neurons/outputs/models/TP_RS bin10 lag100 dur200 affine-kinX-nmahal f70c5c.Fold0.pth"
    cluster_neurons_by_model_weight(file)
