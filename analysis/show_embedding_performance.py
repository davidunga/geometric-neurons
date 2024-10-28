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
from analyze_embedding_matrix import concentrate_embedder_model


def seek_model():
        max_n_pairs = 100_000

        model, cfg = cv_results_mgr.get_model_and_config(model_file)


        data_mgr = DataMgr(cfg.data)
        pairs_df = data_mgr.load_pairing(n_pairs=max_n_pairs)

        pairs_df_funcs.report_segment_uniformity(pairs_df)
        pairs_df_funcs.report_sameness_part(pairs_df, raise_unbalanced=True)
        is_same = pairs_df['isSame'].to_numpy()

        vecs, _ = data_mgr.get_inputs()


def draw_embedding_vs_sameness_ROC(
        model_file, pop_labels: list[NEURAL_POP], shuff: bool = False,
        pop_spec: str = 'chosen', seed: int = 1):

    embed_types = ['YES']
    max_n_pairs = 100_000
    pop_colors = plotting.get_nice_colors(pop_labels)

    if pop_spec == 'chosen':
        pop_spec = cv_results_mgr.get_chosen_pop_specs()

    model, cfg = cv_results_mgr.get_model_and_config(model_file)

    data_mgr = DataMgr(cfg.data)
    pairs_df = data_mgr.load_pairing(n_pairs=max_n_pairs)

    pairs_df_funcs.report_segment_uniformity(pairs_df)
    pairs_df_funcs.report_sameness_part(pairs_df, raise_unbalanced=True)
    is_same = pairs_df['isSame'].to_numpy()

    input_vecs, _ = data_mgr.get_inputs()
    neural_pop = NeuralPopulation.from_model(model_file, spec=pop_spec)
    embeddings = {}
    for pop_label in pop_labels:
        embeddings[pop_label] = embedtools.prep_embeddings(
            model, neural_pop.filter_inputs(input_vecs, include=pop_label),
            shuff=shuff, seed=seed, embed_types=embed_types)

    axs = plotting.named_subplots(cols=embed_types)
    for ax in axs.values():
        plotting.prep_roc_axis(ax=ax)

    for embed_type in embed_types:
        ax = axs[embed_type]
        for pop_label in pop_labels:
            color = pop_colors[pop_label]
            embedded_vecs = embeddings[pop_label][embed_type]
            embedded_dists2 = -embedding_eval.pairs_dists2(embedded_vecs, pairs=pairs_df[['seg1', 'seg2']].to_numpy())
            fpr, tpr, _ = roc_curve(y_true=is_same, y_score=embedded_dists2)
            auc = roc_auc_score(y_true=is_same, y_score=embedded_dists2)
            print(str(pop_label), "auc=", auc)
            ax.plot(fpr, tpr, color=color, label=f'{pop_label}: auc={auc:2.2f}')

        #ax.legend()
    if len(embed_types) > 1:
        plotting.set_outter_labels(axs, t=embed_types)
    plt.suptitle(f"Affine-Equivalent Prediction - {data_mgr.monkey}\nAUC={auc:2.3f}")
    #plt.suptitle(cfg.str() + "\n" + str(neural_pop))


def draw_embedded_vs_metric_dists(model_file, shuff: bool = False,
                                  pop_name: NEURAL_POP = NEURAL_POP.FULL, seed: int = 1):
    # ----
    max_n_pairs = .005
    binned_plot_kws = {'bins': stats.BinSpec(10, 'u'),
                       'loc': 'med', 'band': 'scale', 'color': 'limeGreen'}
    # ----

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    cfg = cfg.get_as_eval_config()
    data_mgr = DataMgr(cfg.data)

    # pairs_df = data_mgr.load_pairing()
    # pairs_df.set_index('rank', drop=True, inplace=True)
    # sampled_ranks = np.round(np.linspace(0, len(pairs_df) - 1, min(max_n_pairs, len(pairs_df)))).astype(int)
    # pairs_df = pairs_df.loc[sampled_ranks, :]

    pairs_df = data_mgr.load_pairing(n_pairs=max_n_pairs)
    metric_dists = pairs_df['dist'].to_numpy()

    vecs, _ = data_mgr.get_inputs()

    if pop_name != NEURAL_POP.FULL:
        neural_pop = NeuralPopulation.from_model(model_file)
        vecs[:, ~neural_pop.inputs_mask(pop=pop_name)] = .0

    embeddings = embedtools.prep_embeddings(model, vecs, shuff=shuff, seed=seed)
    for embed_type, embedded_vecs in embeddings.items():
        if embed_type != 'YES':
            continue
        embedded_dists = np.sqrt(embedding_eval.pairs_dists2(embedded_vecs, pairs=pairs_df[['seg1', 'seg2']].to_numpy()))
        from scipy.stats import spearmanr
        print(data_mgr.monkey, spearmanr(embedded_dists, metric_dists))
        print(data_mgr.monkey + " SHUFF", spearmanr(np.random.permutation(embedded_dists), metric_dists))
        plotting.plot_binned_stats(x=metric_dists, y=embedded_dists, counts=False, **binned_plot_kws)
        #plt.title(cfg.str() + f"\nEmbed={embed_type}, {pop_name}")
        plt.title("Affine Distance vs Neural-Embedding Distance - " + data_mgr.monkey)
        plt.ylabel('Affine Distance (Kinematic)')
        plt.xlabel('SubPopulation Distance (Neural)')


if __name__ == "__main__":
    for monkey, model_file in cv_results_mgr.get_chosen_model_per_monkey().items():
        #compute_neuron_importance(model_file)
        draw_embedded_vs_metric_dists(model_file, shuff=False, pop_name=NEURAL_POP.FULL)
        # draw_embedding_vs_sameness_ROC(model_file, shuff=False,
        #                                pop_labels=[NEURAL_POP.FULL])
    plt.show()
