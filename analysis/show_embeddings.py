import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from common.utils import dlutils
from config import Config
from data_manager import DataMgr
from common.utils.typings import *
from pathlib import Path
import torch
from common.metric_learning import embedding_eval
import cv_results_mgr
from scipy.spatial.distance import pdist, squareform
from common.utils.planar_align import PlanarAligner


def _convert_to_test_config(cfg: Config | dict) -> Config:
    if isinstance(cfg, dict):
        cfg = Config(cfg)
    cfg = cfg.copy()
    cfg.data.pairing.notSame_pctl = cfg.data.pairing.same_pctl
    cfg.data.pairing.exclude_pctl = .5
    cfg.data.pairing.balance = False
    return cfg


def _get_random_subset(x, n, rng):
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    return rng.permutation(x)[:n]


def _fill(pts1, pts2, *args, **kwargs):
    x = np.r_[pts1[:, 0], pts2[:, 0][::-1]]
    y = np.r_[pts1[:, 1], pts2[:, 1][::-1]]
    plt.fill(x, y, *args, **kwargs)


def draw_trajectories_grouped_by_embedded_dist(model_file):
    """
    group segments by their embedding distance, and draw trajectories of a random subset of
    the groups.
    """
    # -----------------------
    n_groups_to_draw = 8
    n_segs_in_group = 10
    near_thresh_p = 1
    far_thresh_p = 99
    seed = 1
    # -----------------------

    aligners = [PlanarAligner('offset'), PlanarAligner('rigid'), PlanarAligner('affine')]
    rng = np.random.default_rng(seed)

    def _draw_group_trajs(seg_ixs, aligner):
        # ----
        linewidth = 1
        alpha_by_normalized_zscore = False
        uniform_color = True
        # ----

        aligned_trajs = [aligner(parabola, trajs[ix])[0] for ix in seg_ixs]

        if alpha_by_normalized_zscore:
            min_alpha, max_alpha = .2, .4
            mu = np.mean(aligned_trajs, axis=0)
            dist_from_mu = np.array([np.linalg.norm(trj - mu) for trj in aligned_trajs])
            dist_from_mu = (dist_from_mu - dist_from_mu.min()) / (dist_from_mu.max() - dist_from_mu.min())
            alpha = min_alpha + (1 - dist_from_mu) * (max_alpha - min_alpha)
        else:
            alpha = np.ones(len(seg_ixs)) * .2

        lms = 2 * traj_scale
        color = 'dodgerBlue' if uniform_color else None
        for i, aligned_traj in enumerate(aligned_trajs):
            plt.plot(*aligned_traj.T, lw=linewidth, alpha=alpha[i], color=color)
            plt.xlim([-lms, lms])
            plt.ylim([-lms, lms])
            plt.gca().set_aspect('equal', adjustable='box')

    # --------

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data)
    trajs = data_mgr.get_pairing_trajectories()
    vecs, _ = data_mgr.get_inputs()
    
    traj_scale = 2 * np.mean([np.std(traj, axis=0).mean() for traj in trajs])

    # build shape to align trajectories to
    seg_sz = len(trajs[0])
    parabola = traj_scale * np.c_[np.linspace(-1, 1, seg_sz), np.linspace(-1, 1, seg_sz) ** 2]
    parabola -= parabola.mean()
    
    # normalized embedded distances
    embedded_vecs = dlutils.safe_predict(model, vecs)
    embedded_dists = pdist(embedded_vecs)
    near_thresh, far_thresh = np.percentile(embedded_dists, [near_thresh_p, far_thresh_p])
    embedded_dists = squareform(embedded_dists)

    # near/far in embedding space
    is_near = embedded_dists < near_thresh
    is_far = embedded_dists > far_thresh

    # chose segments to draw along with their near/far groups:
    valids_segs = np.nonzero(np.minimum(np.sum(is_near, axis=1), np.sum(is_far, axis=1)) > n_segs_in_group)[0]
    print("n valids:", len(valids_segs))
    segs_to_draw = _get_random_subset(valids_segs, n_groups_to_draw, rng)

    # near/far segments group for each chosen segment:
    groups = {}
    for seg_ix in segs_to_draw:
        groups[seg_ix] = {
            'near': _get_random_subset(np.nonzero(is_near[seg_ix])[0], n_segs_in_group, rng),
            'far': _get_random_subset(np.nonzero(is_far[seg_ix])[0], n_segs_in_group, rng)
        }

    for group_type in ['near', 'far']:
        _, axs = plt.subplots(ncols=n_groups_to_draw, nrows=len(aligners))
        plt.suptitle(group_type)
        for j, seg_ix in enumerate(segs_to_draw):
            seg_ixs = groups[seg_ix][group_type]
            for i, aligner in enumerate(aligners):
                plt.sca(axs[i, j])
                _draw_group_trajs(seg_ixs, aligner)
                if j == 0: plt.ylabel(aligner.kind)

    plt.show()


def draw_embedded_vs_metric_dists(model_file):

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    cfg = _convert_to_test_config(cfg)
    data_mgr = DataMgr(cfg.data)
    pairs_df = data_mgr.load_pairing()

    pairs_df.set_index('rank', drop=True, inplace=True)

    n_samples = min(50_000, len(pairs_df))
    sampled_ranks = np.round(np.linspace(0, len(pairs_df) - 1, n_samples)).astype(int)
    pairs_df = pairs_df.loc[sampled_ranks, :]

    metric_dists = pairs_df['dist'].to_numpy()

    vecs, _ = data_mgr.get_inputs()
    vecs = torch.as_tensor(vecs, dtype=torch.float32)

    for embed in [False, True]:
        if embed:
            embedded_vecs = model(vecs)
        else:
            embedded_vecs = vecs
        embedded_vecs = embedded_vecs.detach().cpu().numpy()
        embedded_dists = embedding_eval.pairs_dists(embedded_vecs, pairs=pairs_df[['seg1', 'seg2']].to_numpy())
        plot_binned_stats(x=metric_dists, y=embedded_dists, n_bins=10, kind='p', stat='avg', err='se')
        plt.title(f"Embed={embed}")
    plt.show()



# ------ helpers

import numpy as np
import scipy.stats as stats


def calc_stats(a):

    n = len(a)
    if not n:
        a = np.zeros(1)

    std = np.std(a)
    med = np.median(a)
    se = std / np.sqrt(max(1, n))

    ret = {
        'n': n,
        'sum': np.sum(a),
        'avg': np.mean(a),
        'std': std,
        'med': med,
        'mad': 1.4826 * np.median(np.abs(a - med)),
        'se': se,
        'sm': 1.2533 * se
    }

    if not n:
        ret = {k: np.nan if k != 'n' else 0 for k in ret}

    return ret


def calc_binned_stats(x, y, n_bins: int = 10, kind: str = 'u'):
    """
    Computes statistics for y within bins defined on x.
    Parameters:
    - x: Numeric vector.
    - y: Numeric vector, same length as x.
    - n_bins: Number of bins to split x into.
    - kind: 'u' for uniform spacing, 'p' for uniform percentiles.
    """

    if kind == 'u':
        x_bin_edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    elif kind == 'p':
        x_bin_edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError("Unknown binning kind")

    x_bin_edges[-1] += 1e-10
    x_bin_inds = np.digitize(x, x_bin_edges) - 1
    assert x_bin_inds.min() >= 0 and x_bin_inds.max() <= n_bins

    stats_per_bin = [calc_stats(y[x_bin_inds == i]) for i in range(n_bins)]

    stat_names = stats_per_bin[0].keys()
    ret = {stat_name: np.array([ys[stat_name] for ys in stats_per_bin])
           for stat_name in stat_names}

    ret['x'] = (x_bin_edges[1:] + x_bin_edges[:-1]) / 2

    return ret


def plot_binned_stats(x, y, n_bins, kind, stat: str = 'avg', err: str = 'auto'):
    stats = calc_binned_stats(x, y, n_bins, kind)
    if err == 'auto':
        assert stat in ('avg', 'med')
        err = 'se' if stat == 'avg' else 'sm'
    mu = stats[stat]
    er = stats[err]
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    plt.plot(stats['x'], mu, marker='o', linestyle='-', color='b', label=stat.capitalize())
    plt.fill_between(stats['x'], mu - er, mu + er, color='blue', alpha=0.2, label=f'± {err.capitalize()}')
    plt.legend()


if __name__ == "__main__":
    file = "/Users/davidu/geometric-neurons/outputs/models/TP_RS bin10 lag100 dur200 affine-kinX-nmahal f70c5c.Fold0.pth"
    draw_trajectories_grouped_by_embedded_dist(file)
    #draw_embedded_vs_metric_dists(file)
