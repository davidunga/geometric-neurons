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
from sklearn.metrics import roc_curve, roc_auc_score
import embedtools


def draw_embedding_vs_sameness_ROC(model_file, shuff: bool = False, nullify_pop: NEURAL_POP = None, seed: int = 1):
    max_n_pairs = 100_000
    rnd = Rnd(seed)

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data)
    pairs_df = data_mgr.load_pairing()

    same_ixs = np.nonzero(pairs_df['isSame'].to_numpy())[0]
    notSame_ixs = np.nonzero(~pairs_df['isSame'].to_numpy())[0]
    n = min([max_n_pairs // 2, len(same_ixs), len(notSame_ixs)])

    if len(same_ixs) > n:
        same_ixs = rnd.subset(same_ixs, n)
    if len(notSame_ixs) > n:
        notSame_ixs = rnd.subset(notSame_ixs, n)

    pairs_df = pairs_df.iloc[np.r_[same_ixs, notSame_ixs]]
    is_same = pairs_df['isSame'].to_numpy(dtype=int)
    assert len(pairs_df) <= max_n_pairs
    assert abs(is_same.sum() - len(is_same) / 2) <= 1

    vecs, _ = data_mgr.get_inputs()

    if nullify_pop is not None:
        neural_pop = NeuralPopulation.from_model(model_file)
        vecs[:, neural_pop.inputs_mask(pop=nullify_pop)] = .0

    embeddings = embedtools.prep_embeddings(model, vecs, shuff=shuff, seed=seed)
    plotting.prep_roc_axis()
    for embed_type, embedded_vecs in embeddings.items():
        embedded_dists = -embedding_eval.pairs_dists(embedded_vecs, pairs=pairs_df[['seg1', 'seg2']].to_numpy())
        fpr, tpr, _ = roc_curve(y_true=is_same, y_score=embedded_dists)
        auc = roc_auc_score(y_true=is_same, y_score=embedded_dists)
        plt.plot(fpr, tpr, label=f'Embed={embed_type}: auc={auc:2.2f}')
    plt.legend()
    plt.title(cfg.str() + "\n" + f"{n*2} pairs")


def draw_embedded_vs_metric_dists(model_file, shuff: bool = False, nullify_pop: NEURAL_POP = None, seed: int = 1):
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

    if nullify_pop is not None:
        neural_pop = NeuralPopulation.from_model(model_file)
        vecs[:, neural_pop.inputs_mask(pop=nullify_pop)] = .0

    embeddings = embedtools.prep_embeddings(model, vecs, shuff=shuff, seed=seed)
    for embed_type, embedded_vecs in embeddings.items():
        embedded_dists = embedding_eval.pairs_dists(embedded_vecs, pairs=pairs_df[['seg1', 'seg2']].to_numpy())
        plotting.plot_binned_stats(x=metric_dists, y=embedded_dists, **binned_plot_kws)
        plt.title(cfg.str() + f"\nEmbed={embed_type}")
        plt.ylabel('Affine Distance (Kinematic)')
        plt.xlabel('SubPopulation Distance (Neural)')


if __name__ == "__main__":
    for monkey, model_file in cv_results_mgr.get_chosen_model_per_monkey().items():
        draw_embedded_vs_metric_dists(model_file, shuff=False)
        #draw_embedding_vs_sameness_ROC(model_file, shuff=False)
    plt.show()