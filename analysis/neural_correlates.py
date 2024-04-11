import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch.nn
from data_manager import DataMgr
from common.utils.typings import *
import cv_results_mgr
from common.utils.inlier_detection import InlierDetector


def extract_model_weights(model: torch.nn.Module) -> NpMatrix:
    names = [name for name in model.state_dict() if 'weight' in name]
    assert len(names) == 1
    weights = model.state_dict()[names[0]].detach().numpy()
    return weights


def identify_subpopulation_by_model_weights(model_file, draw=True):
    """
    Estimate the contribution of each input neuron to the embedding model,
    and split the neural population into subpopulations according to their contribution scores.
    optionally draw the result.
    """

    # ----
    neuron_weight_agg = 'median'  # aggregation of weight scores per neuron
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

    outlrs_mask = ~inlier_detector.fit_predict(neuron_weight)

    assert neuron_weight[outlrs_mask].min() > neuron_weight[~outlrs_mask].max(), \
        "Minority population score is not strictly greater than majority's"

    df = pd.DataFrame({'Neuron Weight': neuron_weight, 'Neuron Name': neuron_names,
                       'Neuron Index': np.arange(len(neuron_names)), 'Population': outlrs_mask.astype(int)})

    if draw:
        df['Population'] = ['Main' if p == 0 else 'Sub' for p in df['Population']]
        jp = sns.jointplot(data=df, x='Neuron Index', y='Neuron Weight', hue='Population', kind='scatter',
                           palette={'Main': 'black', 'Sub': 'red'}, alpha=.75)
        jp.ax_marg_x.remove()
        xlm_margin = 2
        jp.ax_joint.set_xlim([-xlm_margin, len(neuron_names) + xlm_margin - 1])
        plt.title("Contribution of Neurons to Embedded Representation" + "\n")
        plt.show()

    return df



if __name__ == "__main__":
    file = "/Users/davidu/geometric-neurons/outputs/models/TP_RS bin10 lag100 dur200 affine-kinX-nmahal f70c5c.Fold0.pth"
    identify_subpopulation_by_model_weights(file)
