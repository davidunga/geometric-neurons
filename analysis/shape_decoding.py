import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.utils import stats
from common.utils.procrustes import PlanarAlign
from common.utils.distance_metrics import normalized_mahalanobis
from common.utils.conics import make_conic_points
from motorneural.data import Segment
from data_manager import DataMgr
import cv_results_mgr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from common.utils import plotting
from common.utils import dlutils
from common.utils.randtool import Rnd
from common.utils import polytools
import seaborn as sns
from common.utils.conics import Conic, fit_conic_arc_lsqr
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score
from geometrik.utils import is_convex
from sklearn.model_selection import StratifiedKFold
from common.utils import stats


def get_segment_shape_labels(segments: list[Segment], n: int = 500):
    from common.utils.conics import draw_arc_properties

    errors = np.zeros(len(segments), float) + np.inf
    conics = [Conic() for _ in segments]
    for i, seg in enumerate(segments):
        if is_convex(seg.kin.X) or True:
            conic = fit_conic_arc_lsqr(seg.kin.X)
            if conic is None:
                continue
            xx, yy = plotting.get_grid_for_points(seg.kin.X)
            plt.plot(*seg.kin.X.T, 'k.')
            conic.draw(xx, yy, 'c')
            #draw_arc_properties(seg.kin.X)
            plt.show()
            #conics[i], errors[i] = fit_conic_arc(seg.kin.X)

    n_valids = np.isfinite(errors).sum()
    labels = np.zeros(len(segments), int)
    counts = {1: 0, 2: 0}
    for i in np.argsort(errors)[:n_valids]:
        label = 1 if conics[i].is_parabola else 2
        if counts[label] < n:
            labels[i] = label
            counts[label] += 1
            if min(counts.values()) == n:
                break

    print(counts)
    return labels, errors, conics


def classify_shape(model_file):
    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data, persist=True)
    segments = data_mgr.load_segments()

    shape_labels, shape_errors, conics = get_segment_shape_labels(segments, n=500)

    input_vecs, _ = data_mgr.get_inputs()
    embedded_vecs = dlutils.safe_predict(model, input_vecs)

    classifier = LogisticRegression()
    #classifier = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=1)

    valid_ixs = np.nonzero(shape_labels > 0)[0]
    X = embedded_vecs[valid_ixs]
    is_parab = shape_labels[valid_ixs] == 1
    probas = cross_val_predict(classifier, X=X, y=is_parab, method='predict_proba',
                               cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1))
    ftr, tpr, _ = roc_curve(y_true=is_parab, y_score=probas[:, 1])
    cls_success = (probas[:, 1] > probas[:, 0]) == is_parab

    plotting.prep_roc_axis()
    plt.plot(ftr, tpr)
    plt.title(f"{cfg.str()}\nauc={roc_auc_score(y_true=is_parab, y_score=probas[:, 1])}")

    eccens = np.asarray([conics[i].params.e.real for i in valid_ixs], float)

    plt.figure()
    ii = (eccens < .98) & stats.Inliers('iqr').is_inlier(eccens)
    plt.scatter(eccens[ii], probas[ii, 1])

    #
    # nbins = 10
    # inds, e_bin_edges = stats.safe_digitize(eccens, stats.BinSpec(nbins, 'u'))
    # success_rate = np.zeros(nbins, float) + np.nan
    # success_rate_sd = np.zeros(nbins, float) + np.nan
    # eccen_bins = (e_bin_edges[:-1] + e_bin_edges[1:]) / 2
    # for ind in range(10):
    #     ii = inds == ind
    #     if np.sum(ii) > 5:
    #         v = cls_success[ii]
    #         success_rate[ind] = np.mean(v)
    # plt.figure()
    # plt.plot(eccen_bins, success_rate)



if __name__ == "__main__":
    for monkey, model_file in cv_results_mgr.get_chosen_model_per_monkey().items():
        if monkey in ['RS']:
            classify_shape(model_file)
    plt.show()
