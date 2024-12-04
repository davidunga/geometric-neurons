from collections import defaultdict
from typing import Callable, NamedTuple
import numpy as np
from sklearn import metrics as skmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from common.utils import stats
from sklearn.manifold import MDS
from sklearn_extra.cluster import KMedoids
from scipy.sparse import csr_matrix
from sklearn.manifold import TSNE
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D


class ClusterEvaluator:

    unsupervised_methods = [
        'silhouette_score',
        'calinski_harabasz_score',
        'davies_bouldin_score'
    ]

    supervised_methods = [
        'adjusted_rand_score',
        'normalized_mutual_info_score',
        'fowlkes_mallows_score',
        'adjusted_mutual_info_score',
        'homogeneity_score'
    ][-2:]

    supports_precomp = ['silhouette_score']

    method_funcs = {method: getattr(skmetrics, method)
                    for method in unsupervised_methods + supervised_methods}

    def __init__(self, ignore_neg_labels: bool = True, bootstrap_seed: int = 1,
                 bootstrap_itrs: int = 50, bootstrap_robust: bool = True):
        self.ignore_neg_labels = ignore_neg_labels
        self.bootstrap_seed = bootstrap_seed
        self.bootstrap_itrs = bootstrap_itrs
        self.bootstrap_robust = bootstrap_robust

    def evaluate(self, labels_pred, labels_true=None, x=None, d=None) -> dict[str, stats.ZScoreRes]:

        if labels_true is not None:
            assert x is None and d is None
            methods = self.supervised_methods
        elif x is not None:
            assert d is None
            methods = self.unsupervised_methods
        else:
            assert d is not None
            methods = [m for m in self.unsupervised_methods if m in self.supports_precomp]

        if self.ignore_neg_labels:
            mask = labels_pred >= 0
            if labels_true is not None:
                mask &= labels_true >= 0
                labels_true = labels_true[mask]
            labels_pred = labels_pred[mask]
            if x is not None:
                x = x[mask]
            elif d is not None:
                d = d[mask][:, mask]

        return self._eval_bootstrap(labels_pred=labels_pred,
                                    labels_true=labels_true,
                                    methods=methods, x=x, d=d)

    @staticmethod
    def calc_metric(method: str, labels_pred, labels_true=None, x=None, d=None):
        func = ClusterEvaluator.method_funcs[method]
        if labels_true is not None:
            return func(labels_pred=labels_pred, labels_true=labels_true)
        elif d is not None:
            return func(d, labels=labels_pred, metric='precomputed')
        else:
            return func(x, labels=labels_pred)

    def _eval_bootstrap(self, labels_pred, methods, labels_true=None, x=None, d=None) -> dict[str, stats.ZScoreRes]:
        assert sum(arg is not None for arg in (labels_true, x, d)) == 1

        rng = np.random.default_rng(seed=self.bootstrap_seed)
        values = {method: np.zeros(self.bootstrap_itrs + 1, float) for method in methods}
        lbls = labels_pred.copy()
        for itr in range(self.bootstrap_itrs + 1):
            for method in methods:
                values[method][itr] = ClusterEvaluator.calc_metric(method, labels_pred=lbls,
                                                                   x=x, d=d, labels_true=labels_true)
            lbls = rng.permutation(lbls)
        result = {}
        for method, values in values.items():
            result[method] = stats.calc_zscore(values[0], values[1:], robust=self.bootstrap_robust)
        return result


def labels_to_ind_groups(labels, ignore_neg: bool = True, label_to_key: Callable = None):
    ind_groups = defaultdict(list)
    for ind, label in enumerate(labels):
        if label >= 0 or not ignore_neg:
            ind_groups[label].append(ind)

    if label_to_key is None:
        label_to_key = lambda lbl: lbl

    ind_groups = {label_to_key(label): ind_groups[label] for label in sorted(ind_groups)}
    return ind_groups


def k_medoids(x, k: int, cluster_sz: int = None, precomp: bool = False, seed: int = 42) -> np.ndarray[int]:

    kmedoids = KMedoids(n_clusters=k, metric="precomputed" if precomp else "euclidean",
                        init="k-medoids++",
                        random_state=seed, method='pam')

    labels = kmedoids.fit_predict(x)

    if cluster_sz is not None:
        assert precomp, "to use this functionality x should be a precomputed distance matrix"
        new_labels = np.zeros_like(labels) - 1
        for lbl in range(kmedoids.n_clusters):
            dists_to_mediod = x[kmedoids.medoid_indices_[lbl]]
            core_inds = np.argsort(dists_to_mediod)[:cluster_sz]
            assert np.all(labels[core_inds] == lbl)
            new_labels[core_inds] = lbl
        labels = new_labels

    return labels


def plot_mds(S_dists, A_dists, S_lbls, A_lbls):
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
    S_2D = mds.fit_transform(S_dists)
    A_2D = mds.fit_transform(A_dists)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    sns.scatterplot(x=S_2D[:, 0], y=S_2D[:, 1], hue=S_lbls, palette="viridis", ax=ax[0])
    ax[0].set_title("MDS Visualization (S)")

    sns.scatterplot(x=A_2D[:, 0], y=A_2D[:, 1], hue=A_lbls, palette="viridis", ax=ax[1])
    ax[1].set_title("MDS Visualization (A)")

    plt.show()


def visualize_with_projection(x, pred_labels, true_labels, method='tsne',
                              ignore_neg: bool = True, precomp: bool = False, seed: int = 1, ndims: int = 3):

    kws = {'n_components': ndims, 'random_state': seed}
    if precomp:
        kws['metric'] = 'precomputed'

    if method == 'tsne':
        reducer = TSNE(**kws)
    elif method == 'umap':
        reducer = UMAP(**kws)
    else:
        raise ValueError("Invalid method. Choose 'tsne' or 'umap'.")

    if ignore_neg:
        mask = (pred_labels >= 0) & (true_labels >= 0)
        pred_labels = pred_labels[mask]
        true_labels = true_labels[mask]
        x = x[mask]
        if precomp:
            x = x[:, mask]

    projected = reducer.fit_transform(x)

    fig = plt.figure(figsize=(10, 8))
    if ndims == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        assert ndims == 2
        ax = fig.add_subplot(111)

    scatter = ax.scatter(
        *projected.T,
        c=true_labels, cmap='viridis', s=50, alpha=0.8
    )

   # scatter = sns.scatterplot(x=xx, y=yy, hue=true_labels, style=pred_labels, palette='viridis', s=70)
    # scatter.legend_.set_title("A_lbls (True)")
    # plt.title(f"{method.upper()} Projection of S (Color: A_lbls, Style: S_lbls)")
    # plt.xlabel("Component 1")
    # plt.ylabel("Component 2")
    #plt.show()

