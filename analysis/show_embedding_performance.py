import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from common.utils import dlutils
from config import Config
from data_manager import DataMgr
from common.utils.typings import *
from pathlib import Path
from common.utils import plotting
from common.utils import stats
import torch
from common.metric_learning import embedding_eval
import cv_results_mgr
from scipy.spatial.distance import pdist, squareform
from common.utils.planar_align import PlanarAligner
from analysis.neural_population import NeuralPopulation, NEURAL_POP
from collections import Counter
from motorneural.data import Segment
from common.utils import strtools
from scipy.stats import ks_2samp, anderson_ksamp
from common.utils.randtool import Rnd
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, MiniBatchKMeans
import dataslice_properties
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from common.utils import conics
from common.utils import polytools
from analysis.costume_dataframes import pairs_df_funcs
from sklearn.metrics import roc_curve, roc_auc_score
import embedtools


def draw_embedding_vs_sameness_ROC(model_file, pop_names: list[NEURAL_POP],
                                   shuff: bool = False, pop_weight_bys: list[str] = None, seed: int = 1):
    max_n_pairs = 100_000
    if pop_weight_bys is None:
        pop_weight_bys = ['svd_avg']
    elif isinstance(pop_weight_bys, str):
        pop_weight_bys = [pop_weight_bys]

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data)
    pairs_df = data_mgr.load_pairing(n_pairs=max_n_pairs)

    pairs_df_funcs.report_segment_uniformity(pairs_df)
    pairs_df_funcs.report_sameness_part(pairs_df, raise_unbalanced=True)

    vecs, _ = data_mgr.get_inputs()
    embed_types = ['YES']
    axs = plotting.named_subplots(rows=pop_weight_bys, cols=pop_names)
    for pop_weight_by in pop_weight_bys:
        neural_pop = NeuralPopulation.from_model(model_file, weight_by=pop_weight_by)
        for pop_name in pop_names:
            pop_vecs = vecs.copy()
            if pop_name != NEURAL_POP.FULL:
                pop_vecs[:, ~neural_pop.inputs_mask(pop=pop_name)] = .0
            embeddings = embedtools.prep_embeddings(model, pop_vecs, shuff=shuff, seed=seed)
            ax = axs[(pop_weight_by, pop_name)]
            plotting.prep_roc_axis(ax=ax)
            for embed_type, embedded_vecs in embeddings.items():
                embedded_dists = -embedding_eval.pairs_dists(embedded_vecs, pairs=pairs_df[['seg1', 'seg2']].to_numpy())
                fpr, tpr, _ = roc_curve(y_true=is_same, y_score=embedded_dists)
                auc = roc_auc_score(y_true=is_same, y_score=embedded_dists)
                ax.plot(fpr, tpr, label=f'Embed={embed_type}: auc={auc:2.2f}')
            ax.legend()
    plotting.set_outter_labels(axs, t=pop_names, y=pop_weight_bys)
    plt.suptitle(cfg.str() + "\n" + f"{len(pairs_df)} pairs")


def draw_embedded_vs_metric_dists(model_file, shuff: bool = False,
                                  pop_name: NEURAL_POP = NEURAL_POP.FULL, seed: int = 1):
    # ----
    max_n_pairs = 100_000
    binned_plot_kws = {'bins': stats.BinSpec(10, 'u'),
                       'loc': 'med', 'band': 'scale', 'color': 'limeGreen'}
    # ----

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    cfg = cfg.get_as_eval_config()
    data_mgr = DataMgr(cfg.data)

    pairs_df = data_mgr.load_pairing()
    pairs_df.set_index('rank', drop=True, inplace=True)
    sampled_ranks = np.round(np.linspace(0, len(pairs_df) - 1, min(max_n_pairs, len(pairs_df)))).astype(int)
    pairs_df = pairs_df.loc[sampled_ranks, :]

    metric_dists = pairs_df['dist'].to_numpy()

    vecs, _ = data_mgr.get_inputs()

    if pop_name != NEURAL_POP.FULL:
        neural_pop = NeuralPopulation.from_model(model_file)
        vecs[:, ~neural_pop.inputs_mask(pop=pop_name)] = .0

    embeddings = embedtools.prep_embeddings(model, vecs, shuff=shuff, seed=seed)
    for embed_type, embedded_vecs in embeddings.items():
        embedded_dists = embedding_eval.pairs_dists(embedded_vecs, pairs=pairs_df[['seg1', 'seg2']].to_numpy())
        plotting.plot_binned_stats(x=metric_dists, y=embedded_dists, **binned_plot_kws)
        plt.title(cfg.str() + f"\nEmbed={embed_type}, {pop_name}")
        plt.ylabel('Affine Distance (Kinematic)')
        plt.xlabel('SubPopulation Distance (Neural)')


if __name__ == "__main__":
    for monkey, model_file in cv_results_mgr.get_chosen_model_per_monkey().items():
        #compute_neuron_importance(model_file)
        #draw_embedded_vs_metric_dists(model_file, shuff=False, pop_name=pop_name)
        draw_embedding_vs_sameness_ROC(
            model_file, shuff=False,
            pop_names=[NEURAL_POP.MINORITY, NEURAL_POP.MAJORITY, NEURAL_POP.FULL],
            pop_weight_bys=['svd_avg1', 'svd_max1'])
    plt.show()