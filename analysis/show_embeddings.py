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


def _convert_to_test_config(cfg: Config | dict) -> Config:
    if isinstance(cfg, dict):
        cfg = Config(cfg)
    cfg = cfg.copy()
    cfg.data.pairing.notSame_pctl = cfg.data.pairing.same_pctl
    cfg.data.pairing.exclude_pctl = .5
    cfg.data.pairing.balance = False
    return cfg



def draw_embedded_vs_metric_dists(model_file):

    embedder, cfg = cv_results_mgr.get_model_and_config(model_file)
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
            embedded_vecs = embedder(vecs)
        else:
            embedded_vecs = vecs

        embedded_vecs = embedded_vecs.detach().cpu().numpy()
        embedded_dists = embedding_eval.pairs_dists(embedded_vecs, pairs=pairs_df[['seg1', 'seg2']].to_numpy())
        plot_binned_stats(x=metric_dists, y=embedded_dists, n_bins=10, kind='p', stat='avg', err='se')
        plt.title(f"Embed={embed}")
    plt.show()

    print(".")


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
    draw_embedded_vs_metric_dists(file)
