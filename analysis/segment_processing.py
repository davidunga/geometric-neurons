import numpy as np
from typing import Callable
from common.utils import stats
from motorneural.data import Segment
from neural_population import NeuralPopulation, NEURAL_POP
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import embedtools
from sklearn.metrics import silhouette_score


def digitize_segments(segments: list[Segment], n: int, by: str = 'EuSpd',
                      reduce_fun: Callable = np.mean, include_ixs=None) -> np.ndarray[int]:
    if include_ixs is None:
        include_ixs = np.arange(len(segments))
    else:
        include_ixs = np.sort(include_ixs)
    values = [reduce_fun(segments[i][by]) for i in include_ixs]
    labels = np.zeros(len(segments), int)
    labels[include_ixs] = stats.safe_digitize(values, stats.BinSpec(n, 'p'))[0]
    return labels


def compute_projections(model, input_vecs, neural_pop: NeuralPopulation, groups: dict | np.ndarray,
                        pop_names: list[NEURAL_POP], n_pcs: int = 2, embed_type: str = 'NO',
                        methods: str = 'lda'):

    if isinstance(groups, dict):
        groups = stats.convert_group_dict_to_labels(groups, n=len(input_vecs))
    assert len(groups) == len(input_vecs)

    methods = methods.split(',')
    mask = groups > 0

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
            score = silhouette_score(pc_vecs[mask], groups[mask])
            results.append((pop_name, method, pc_vecs, score))

    return results
