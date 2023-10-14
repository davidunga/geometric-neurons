"""
Manage triplet data & loss
"""
import torch
import numpy as np
from collections import Counter
from common import symmetric_pairs
from common.utils.typings import *
from sklearn import metrics
from scipy import stats
from copy import deepcopy
from common.utils.devtools import verbolize


class SamenessData(symmetric_pairs.SymmetricPairsData):

    _SAME = 'same'
    _NOTSAME = 'notSame'

    def __init__(self,
                 X: ArrayLike,
                 triplet_min_prevalence: int = 5,
                 same_counts: Counter = None,
                 notSame_counts: Counter = None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.X = np.asarray(X)
        self._triplet_min_prevalence = triplet_min_prevalence
        self._same_counts = same_counts
        self._notSame_counts = notSame_counts
        self._triplet_sampling_proba = None
        self._triplet_participating_items = None

    def copy(self, include_items: Container = None):

        pairs = self.pairs.copy()

        if include_items is not None:
            pairs.loc[~pairs[['item1', 'item2']].isin(include_items).all(axis=1), 'group'] = None
            same_counts = None
            notSame_counts = None
        else:
            same_counts = deepcopy(self._same_counts)
            notSame_counts = deepcopy(self._notSame_counts)

        return SamenessData(X=self.X.copy(),
                            n=self.n,
                            data=self.data.copy() if self.data is not None else None,
                            pairs=pairs,
                            triplet_min_prevalence=self._triplet_min_prevalence,
                            same_counts=same_counts,
                            notSame_counts=notSame_counts)

    def init_triplet_sampling(self):
        counts = self.prevalence_counts()
        self._triplet_sampling_proba = np.divide(1, counts, out=np.zeros_like(counts, float), where=counts != 0)
        self._triplet_sampling_proba /= self._triplet_sampling_proba.sum()
        assert len(counts) == len(self._triplet_sampling_proba) == self.n
        self._triplet_participating_items = np.nonzero(counts)[0]

    @property
    def triplet_participating_items(self) -> NpVec[int]:
        return self._triplet_participating_items

    @property
    def triplet_participating_n(self) -> int:
        return len(self._triplet_participating_items)

    @classmethod
    @verbolize()
    def from_sameness_sign(cls, sameness: Sequence, X: ArrayLike, *args, **kwargs):
        n = symmetric_pairs.num_items(len(sameness))
        sameness = sameness[sameness != 0]
        sameness[sameness == -1] = SamenessData._NOTSAME
        sameness[sameness == 1] = SamenessData._SAME
        return cls(X=X, n=n, group_by=sameness, *args, **kwargs)

    def to(self, *args, **kwargs):
        if not torch.is_tensor(self.X):
            self.X = torch.as_tensor(np.asarray(self.X), *args, **kwargs)
        else:
            self.X = self.X.to(*args, **kwargs)

    @property
    def same_counts(self) -> Counter:
        if self._same_counts is None:
            self._same_counts = Counter(self.item_pairs(label=SamenessData._SAME).flatten().tolist())
        return self._same_counts

    @property
    def notSame_counts(self) -> Counter:
        if self._notSame_counts is None:
            self._notSame_counts = Counter(self.item_pairs(label=SamenessData._NOTSAME).flatten().tolist())
        return self._notSame_counts

    def prevalence_counts(self) -> NpVec[int]:
        both_counts = np.zeros((self.n, 2), int)
        for item in range(self.n):
            both_counts[item] = self.same_counts.get(item, 0), self.notSame_counts.get(item, 0)
        counts = both_counts.max(axis=1)
        counts[both_counts.min(axis=1) < self._triplet_min_prevalence] = 0
        return counts

    def sample_triplet_items(self, anchors: int | Sequence[int], rand_seed: int = None):

        assert self._triplet_sampling_proba is not None, "Triplet sampling not initialized"
        assert self.triplet_participating_items is not None, "Triplet sampling not initialized"

        rng = np.random.default_rng(rand_seed)

        if isinstance(anchors, int):
            anchors = self.triplet_participating_items[rng.integers(self.triplet_participating_n, size=anchors)]

        def _sample(item_partners):
            assert len(item_partners)
            cumulative_proba = np.cumsum(self._triplet_sampling_proba[item_partners])
            cumulative_proba /= cumulative_proba[-1]
            i = np.searchsorted(cumulative_proba, rng.random())
            return item_partners[i]

        positives = []
        negatives = []
        for anchor in anchors:
            partners_of_anchor = self.partners_of_item(anchor)
            positives.append(_sample(partners_of_anchor[SamenessData._SAME]))
            negatives.append(_sample(partners_of_anchor[SamenessData._NOTSAME]))

        return anchors, positives, negatives

    def sample_triplets(self, anchors: int | Sequence[int], rand_seed: int = None):
        anchors, positives, negatives = self.sample_triplet_items(anchors=anchors, rand_seed=rand_seed)
        A = self.X[anchors]
        P = self.X[positives]
        N = self.X[negatives]
        return A, P, N


def calc_triplet_loss(p_dists: NDArray, n_dists: NDArray, margin: float = 1.) -> float:
    d = p_dists - n_dists + margin
    d[d < 0] = 0
    return float(np.mean(d))


class SamenessEval:

    fpr: NpVec[float]
    tpr: NpVec[float]
    ths: NpVec[float]
    auc: float
    loss: float | None

    @property
    def tscore(self):
        # returns the t-test statistic if it's statistically significant, else 0
        return 0 if self.ttest.pvalue > .05 else self.ttest.statistic

    def __init__(self,
                 sameness: SamenessData,
                 n_samples: int = 1000,
                 kind: str = "triplet",
                 triplet_margin: float = 1.):

        assert n_samples % 2 == 0

        self.sameness = sameness
        self.triplet_margin = triplet_margin
        self.kind = kind
        self.ttest = None
        self.items: Sequence[tuple[int, int]]
        self.is_same: NpVec[bool]

        if self.kind == "pair":
            self.items, self.is_same = zip(*sameness.labeled_pairs())
        elif self.kind == "triplet":
            a, p, n = sameness.sample_triplet_items(anchors=n_samples // 2, rand_seed=0)
            self.items = [(aa, pp) for aa, pp in zip(a, p)] + [(aa, nn) for aa, nn in zip(a, n)]
            self.is_same = np.arange(len(self.items)) < len(p)

        assert len(self.items) == len(self.is_same) == n_samples

    def evaluate(self, embedder: torch.nn.Module):

        is_training = embedder.training
        embedder.train(False)
        eX = embedder(self.sameness.X).detach().numpy()
        embedder.train(is_training)

        dists = np.linalg.norm([eX[i] - eX[j] for (i, j) in self.items], axis=1)
        self.fpr, self.tpr, self.ths = metrics.roc_curve(y_true=~self.is_same, y_score=dists)
        self.auc = metrics.auc(self.fpr, self.tpr)
        self.ttest = stats.ttest_ind(dists[self.is_same], dists[~self.is_same], alternative='less')
        self.loss = None
        if self.kind == "triplet":
            self.loss = calc_triplet_loss(dists[self.is_same], dists[~self.is_same], margin=self.triplet_margin)

    def __str__(self):
        s = f"auc={self.auc * 100:2.2f}, tscore={self.tscore:2.1f}"
        if self.loss is not None:
            s += f" loss={self.loss:2.4f}"
        return s

    def results_dict(self):
        return {"auc": self.auc,
                "tscore": self.tscore,
                "loss": self.loss,
                "ttest-t": self.ttest.statistic,
                "ttest-p": self.ttest.pvalue,
                "ttest-ci": self.ttest.confidence_interval()}
