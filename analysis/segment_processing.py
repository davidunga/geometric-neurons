import numpy as np
from typing import Callable
from common.utils import stats
from motorneural.data import Segment
from neural_population import NeuralPopulation, NEURAL_POP
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import embedtools
from sklearn.metrics import silhouette_score, balanced_accuracy_score
from sklearn.mixture import GaussianMixture
from itertools import permutations
import geometrik as gk
from common.utils.conics import Conic


def balanced_accuracy_perms(y_true, y_pred) -> float:
    pred_labels = set(y_pred)
    score = 0
    for mapping in permutations(pred_labels):
        score = max(balanced_accuracy_score(y_true=y_true, y_pred=[mapping[v] for v in y_pred]), score)
    return score



def digitize_segments(segments: list[Segment], n: int, by: str = 'EuSpd',
                      reduce_fun: Callable = np.mean, include_ixs=None) -> np.ndarray[int]:
    if include_ixs is None:
        include_ixs = np.arange(len(segments))
    else:
        include_ixs = np.sort(include_ixs)

    labels = np.zeros(len(segments), int)
    if ',' in by:
        bys = by.split(',')
        assert n % len(bys) == 0
        n = n // len(bys)
        rng = np.random.default_rng(1)
        include_ixs = rng.permutation(include_ixs)
        i_stop = 0
        for by in bys:
            i_start = i_stop
            i_stop = min(i_start + 1 + int(len(include_ixs) / len(bys)), len(include_ixs))
            include_ixs_ = include_ixs[i_start: i_stop]
            values = [reduce_fun(segments[i][by]) for i in include_ixs_]
            labels[include_ixs_] = stats.safe_digitize(values, stats.BinSpec(n, 'p'))[0] + 1 + labels.max()
    else:
        values = np.array([reduce_fun(segments[i][by]) for i in include_ixs])
        p90 = np.percentile(values, 90)
        include_ixs = include_ixs[values < p90]
        values = values[values<p90]

        labels[include_ixs] = stats.safe_digitize(values, stats.BinSpec(n, 'p'))[0] + 1
    return labels


def pop_disp_label(pop_label):
    lookup = {'MINORITY': 'Affine Population', 'MAJORITY': 'NonAffine Population',
              'FULL': 'Full Population', 'MIDMAJ': 'Non-Affine Population'}
    return lookup[str(pop_label)]


def compute_projections(model, input_vecs, neural_pop: NeuralPopulation, groups: dict | np.ndarray,
                        pop_names: list[NEURAL_POP], n_pcs: int = 2, embed_type: str = 'NO',
                        methods: str = 'lda'):

    if isinstance(groups, dict):
        groups = stats.convert_group_dict_to_labels(groups, n=len(input_vecs))
    assert len(groups) == len(input_vecs)

    methods = methods.split(',')
    mask = groups > 0
    n_groups = len(set(groups[mask]))

    results = []
    for pop_name in pop_names:

        vecs = input_vecs.copy()
        vecs[:, ~neural_pop.inputs_mask(pop_name)] = .0
        vecs = embedtools.prep_embeddings(model, vecs, embed_types=[embed_type])[embed_type]

        for method in methods:

            if method == 'lda':
                projector = LinearDiscriminantAnalysis(n_components=n_pcs).fit(vecs[mask], groups[mask])
            elif method == 'pca1':
                projector = PCA(n_components=n_pcs).fit(vecs[mask])
            elif method == 'pca2':
                label_centroids = np.array([np.mean(vecs[groups == lbl], axis=0)
                                            for lbl in set(groups[mask])])
                projector = PCA(n_components=n_pcs).fit(label_centroids)
            else:
                raise ValueError()

            pc_vecs = projector.transform(vecs)
            silhouette = silhouette_score(pc_vecs[mask], groups[mask])
            gmm = GaussianMixture(n_components=n_groups).fit(pc_vecs[mask], groups[mask] - 1)
            gmm_score = balanced_accuracy_perms(y_true=groups[mask] - 1, y_pred=gmm.predict(pc_vecs[mask]))
            score = (silhouette, gmm_score)
            results.append((pop_disp_label(pop_name), method, pc_vecs, score))

    return results



def segment_crv(segment: Segment, t='index', conic: Conic = None, crv_kws: dict = None):
    if not crv_kws: crv_kws = {}
    if conic is not None:
        X = conic.parametric_pts(n=500)
    else:
        X = segment.kin.X
    return gk.spcurves.NDSpline(X, k=4, t=t, **crv_kws)


def stats_over_segments(segments: list, neurons: list = None, kin: str = None):
    if neurons:
        x = [s.nerual[neurons] for s in segments]
    else:
        x = [s.kin[kin] for s in segments]
    stts = stats.calc_stats(x, axis=0)
    return stts
