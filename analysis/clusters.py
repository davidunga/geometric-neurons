import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from common.utils import dlutils
from data_manager import DataMgr
from common.utils.typings import *
from pathlib import Path
from common.utils import plotting
from common.utils import stats
import cv_results_mgr
from scipy.spatial.distance import pdist, squareform
import dataslice_properties
from scipy.stats import ttest_ind


def embedded_distances_within_kinematic_clusters(model_file):
    """
    Tests that embedded neural activity is:
     1. Similar for segments of similar shapes.
     2. Different for segments of similar speed / direction.
    """

    # -----
    kin_props = ('ang', 'arclen', 'shape.c', 'shape.p', 'sanity.MustFail', 'sanity.MustPass')
    kin_bins = stats.BinSpec(5, 'u')
    conic_tols = {'circ_tol': .1, 'parab_tol': .1}
    invalid_label = -1
    rng = np.random.default_rng(1)
    # -----

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data, persist=True)
    segments = data_mgr.load_segments()
    n = len(segments)

    vecs, _ = data_mgr.get_inputs()
    embedded_vecs = dlutils.safe_predict(model, vecs)
    embedded_dists = squareform(pdist(embedded_vecs))

    print("Mean embedded pairwise distances -- Within kinematic cluster vs Full population:")

    shape_types = dataslice_properties.shape_types(segments, **conic_tols)

    for kin_prop in kin_props:

        print(f"{kin_prop}:", end=" ")

        if kin_prop == 'ang':
            labels = dataslice_properties.ang_bins(segments, kin_bins)
        elif kin_prop == 'arclen':
            labels = dataslice_properties.arclen_bins(segments, kin_bins)
        elif kin_prop.startswith('shape'):
            shape_type = kin_prop.split('.')[-1][0]
            labels = (shape_types == shape_type).astype(int)
            labels[labels == 0] = invalid_label
        elif kin_prop in ('sanity.MustFail', 'sanity.MustPass'):
            ii = np.arange(n)
            jj = rng.integers(3, size=n)
            labels = stats.safe_digitize(embedded_vecs[ii, jj], kin_bins)[0]
            if kin_prop == 'sanity.MustFail':
                labels = rng.permutation(labels)
        else:
            raise ValueError('Unknown kin prop')

        mask = np.triu(np.ones((n, n)), k=1).astype(bool) & (labels == labels.reshape(-1, 1))
        mask[labels == invalid_label, :] = False
        mask[:, labels == invalid_label] = False
        intra_dists = embedded_dists[mask]

        intra_mu = np.mean(intra_dists)
        all_mu = np.mean(squareform(embedded_dists))
        rel_mu = (intra_mu - all_mu) / all_mu
        result = ttest_ind(intra_dists, squareform(embedded_dists), equal_var=False, alternative='less')
        print(f"Within/Full dists = {intra_mu:2.3f}/{all_mu:2.3f} = {rel_mu:2.3%}, pval={result.pvalue:2.3f}")



if __name__ == "__main__":
    file = "/Users/davidu/geometric-neurons/outputs/models/TP_RS bin10 lag100 dur200 affine-kinX-nmahal f70c5c.Fold0.pth"
    embedded_distances_within_kinematic_clusters(file)
