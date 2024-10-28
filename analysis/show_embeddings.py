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


def fit_conic_to_traj_raster(paths_raster, raster_value_thresh: float = .1,
                             n_samples_per_col: int = 3, kind: str = 'p', ransac_tol: float = 2,
                             allow_rotation: bool = False, ransac_max_iters: int = 5000):

    x = np.tile(np.arange(paths_raster.shape[1]), n_samples_per_col)
    y = np.argsort(paths_raster, axis=0)[-n_samples_per_col:].flatten()
    v = paths_raster[y, x]
    ii = v > raster_value_thresh
    v = v[ii]
    pts = np.c_[x, y][ii]
    pts = pts[np.argsort(pts[:, 0])]

    conic = conics.fit_conic_ransac(pts, weights=v, kind=kind, allow_rotation=allow_rotation,
                                    max_iters=ransac_max_iters, tol=ransac_tol)

    return conic, pts


def _draw_group_trajs(trajs, aligner, scale, draw_conic: bool = False):
    # ----
    raster_size = 100, 100
    raster_lw = 2
    mode = 'img'
    linewidth = 1
    alpha_by_normalized_zscore = False
    uniform_color = True
    draw_stats = True

    lms = 2 * scale
    color = 'dodgerBlue' if uniform_color else None
    # ----

    ref_traj = scale * np.c_[np.linspace(-1, 1, len(trajs[0])), np.linspace(-1, 1, len(trajs[0])) ** 2]
    ref_traj -= ref_traj.mean()

    aligned_trajs = [aligner(ref_traj, traj)[0] for traj in trajs]
    paths_raster, raster_scale, raster_offset = polytools.rasterize_paths(aligned_trajs, raster_size, width=raster_lw)

    if draw_conic:
        conic, pts = fit_conic_to_traj_raster(paths_raster, kind='p')
        if mode != 'img':
            pts = (pts - raster_offset) / raster_scale
            conic = conic.get_transformed(offset=-raster_offset, sx=1/raster_scale[0], sy=1/raster_scale[1])
    else:
        conic = None

    if mode == 'img':
        plt.imshow(paths_raster, cmap='gray')
    else:
        if alpha_by_normalized_zscore:
            min_alpha, max_alpha = .2, .4
            mu = np.mean(aligned_trajs, axis=0)
            dist_from_mu = np.array([np.linalg.norm(trj - mu) for trj in aligned_trajs])
            dist_from_mu = (dist_from_mu - dist_from_mu.min()) / (dist_from_mu.max() - dist_from_mu.min())
            alpha = min_alpha + (1 - dist_from_mu) * (max_alpha - min_alpha)
        else:
            alpha = np.ones(len(trajs)) * .2

        for i, aligned_traj in enumerate(aligned_trajs):
            plt.plot(*aligned_traj.T, lw=linewidth, alpha=alpha[i], color=color)
            plt.xlim([-lms, lms])
            plt.ylim([-lms, lms])
            plt.gca().set_aspect('equal', adjustable='box')

    if conic is not None:
        #plt.plot(*pts.T,'r.')
        conic.draw(x=pts[:, 0], y=pts[:, 1], details=False, color='c')
        plt.title(str(conic))


def draw_trajectories_grouped_by_embedded_dist(
        model_file, shuff: bool = False, n_groups_to_draw: int = 3,
        n_segs_in_group: int = 50, seed: int = 1):
    """
    group segments by their embedding distance, and draw trajectories of a random subset of
    the groups.
    """

    # -----------------------
    n_raw_clusters = 1 / n_segs_in_group
    # -----------------------

    aligners = [PlanarAligner('offset'), PlanarAligner('rigid'), PlanarAligner('ortho')]
    rnd = Rnd(seed)

    # --------

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data, persist=True)
    trajs = data_mgr.get_pairing_trajectories()
    vecs, _ = data_mgr.get_inputs()

    traj_scale = 2 * np.mean([np.std(traj, axis=0).mean() for traj in trajs])

    # normalized embedded distances
    neural_pop = NeuralPopulation.from_model(model_file)
    vecs[:, ~neural_pop.inputs_mask(NEURAL_POP.MAJORITY)] = .0
    vecs[:, neural_pop.inputs_mask(NEURAL_POP.MAJORITY)] = .0
    embedded_vecs = dlutils.safe_predict(model, vecs)

    # raw clusters:
    n_clusters = int(len(embedded_vecs) * n_raw_clusters) if n_raw_clusters < 1 else n_raw_clusters
    km = MiniBatchKMeans(n_clusters=n_clusters, n_init='auto', random_state=seed).fit(embedded_vecs)
    labels = km.labels_
    if shuff:
        labels = Rnd(1).shuffle(labels)
    valid_labels = [label for label, cluster_size in Counter(labels).items() if cluster_size > n_segs_in_group]

    # decide which raw clusters to draw:
    best_score = 0
    groups_to_draw = None
    for _ in range(10_000):
        candidate_labels = rnd.subset(valid_labels, n_groups_to_draw)
        score = np.median(pdist(km.cluster_centers_[candidate_labels]))
        if score > best_score:
            print(score)
            groups_to_draw = candidate_labels
            best_score = score

    # decide which segments to draw within each cluster:
    segment_groups = []
    for label in groups_to_draw:
        seg_ixs = np.nonzero(labels == label)[0]
        dists_to_centroid = np.linalg.norm(embedded_vecs[seg_ixs] - embedded_vecs[seg_ixs].mean(axis=0), axis=1)
        seg_ixs = seg_ixs[np.argsort(dists_to_centroid)[:n_segs_in_group]]
        segment_groups.append(seg_ixs)

    # draw:
    _, axs = plt.subplots(ncols=len(groups_to_draw), nrows=len(aligners))
    for j, seg_ixs in enumerate(segment_groups):
        for i, aligner in enumerate(aligners):
            plt.sca(axs[i, j])
            _draw_group_trajs([trajs[i] for i in seg_ixs], aligner, traj_scale,
                              draw_conic=aligner.kind=='ortho1')

    plotting.set_outter_labels(axs, y=[aligner.kind for aligner in aligners],
                               t=[f'cluster{i+1}' for i in range(len(segment_groups))])

    plt.suptitle(f"Shuff={shuff}")


def draw_embedded_vs_metric_dists(model_file):

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    cfg = cfg.get_as_eval_config()
    data_mgr = DataMgr(cfg.data)
    pairs_df = data_mgr.load_pairing()

    pairs_df.set_index('rank', drop=True, inplace=True)

    n_samples = min(50_000, len(pairs_df))
    sampled_ranks = np.round(np.linspace(0, len(pairs_df) - 1, n_samples)).astype(int)
    pairs_df = pairs_df.loc[sampled_ranks, :]

    from common.utils.polytools import total_arclen
    trajs = data_mgr.get_pairing_trajectories()
    arclens = np.array([total_arclen(traj) for traj in trajs])
    arclens1 = arclens[pairs_df['seg1'].to_numpy()]
    arclens2 = arclens[pairs_df['seg2'].to_numpy()]
    arclen_diff = np.abs(arclens2 - arclens1)
    arclen_rdiff = 2. * arclen_diff / (arclens2 + arclens1)

    metric_dists = pairs_df['dist'].to_numpy()

    vecs, _ = data_mgr.get_inputs()
    vecs = torch.as_tensor(vecs, dtype=torch.float32)
    binned_plot_kws = {'bins': stats.BinSpec(10, 'u'), 'loc': 'med', 'color': 'limeGreen', 'band': 'scale'}

    neural_pop = NeuralPopulation.from_model(model_file)
    vecs = Rnd(1).shuffle(vecs)

    for embed in [False, True, 'Rand']:
        if embed == 'Rand':
            embedded_vecs = dlutils.safe_predict(dlutils.randomize_weights(model), vecs)
        else:
            embedded_vecs = dlutils.safe_predict(model, vecs) if embed else vecs
        embedded_dists = np.sqrt(embedding_eval.pairs_dists2(embedded_vecs, pairs=pairs_df[['seg1', 'seg2']].to_numpy()))
        plotting.plot_binned_stats(x=metric_dists, y=embedded_dists, **binned_plot_kws)
        plt.title(f"Embed={embed}")

    # embedded_vecs = model(vecs)
    # embedded_vecs = embedded_vecs.detach().cpu().numpy()
    # for dff in ['df', 'rdf']:
    #     xx = arclen_diff if dff == 'df' else arclen_rdiff
    #     embedded_dists = embedding_eval.pairs_dists(embedded_vecs, pairs=pairs_df[['seg1', 'seg2']].to_numpy())
    #     plotting.plot_binned_stats(x=xx, y=embedded_dists, **binned_plot_kws)
    #     plt.title(dff)




if __name__ == "__main__":
    file = cv_results_mgr.get_chosen_model_file('RJ')
    draw_embedded_vs_metric_dists(file)
    #file = "/Users/davidu/geometric-neurons/outputs/models/TP_RS bin10 lag100 dur200 affine-kinX-nmahal f70c5c.Fold0.pth"
    #draw_trajectories_grouped_by_embedded_dist(file, shuff=False, n_segs_in_group=20, n_groups_to_draw=3)
    #draw_trajectories_grouped_by_embedded_dist(file, shuff=True, n_segs_in_group=20, n_groups_to_draw=3)
    plt.show()


