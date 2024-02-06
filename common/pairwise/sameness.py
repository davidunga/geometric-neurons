"""
Manage triplet data & loss
"""
import pandas as pd
import torch
import numpy as np
from collections import Counter
from common import symmetric_pairs
from common.utils.typings import *
from sklearn import metrics
from scipy import stats
from copy import deepcopy
from common.utils.devtools import verbolize
from common.utils import strtools
from common.pairwise.embedding_eval import EmbeddingEvaluator, make_triplet_pairs


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
        self._triplet_sampling_rate = None
        self._triplet_anchors = None
        self.triplet_sampled_counts = None

    def modcopy(self, include_items: Container[int] = None, index_mask: Sequence[bool] = None):
        """ make modified copy """

        if include_items is not None:
            assert index_mask is None
            index_mask = self.mutex_indexes(split_items=include_items) == 1

        pairs = self.pairs.copy()
        if index_mask is not None:
            pairs.loc[~index_mask, 'group'] = None
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

    def init_triplet_sampling(self, anchors: Sequence[int] = None):

        self.reset_triplet_sampled_counts()

        min_counts = np.zeros(self.n, int)
        max_counts = np.zeros(self.n, int)
        for item in range(self.n):
            min_counts[item], max_counts[item] = self.same_counts.get(item, 0), self.notSame_counts.get(item, 0)
            if min_counts[item] > max_counts[item]:
                min_counts[item], max_counts[item] = max_counts[item], min_counts[item]

        can_be_anchor = min_counts >= self._triplet_min_prevalence
        if anchors is not None:
            assert can_be_anchor[anchors].all(), "Some of the specified anchors do not have sufficient counts"
        else:
            anchors = np.nonzero(can_be_anchor)[0]

        # index of pairs that are reachable through achors, i.e., are anchors or pairs of an anchor
        is_reachable = self.pairs.loc[self.pairs['group'].notna(), ['item1', 'item2']].isin(anchors).any(axis=1)
        reachable_indexes = is_reachable[is_reachable].index

        # count how many times each item can be reached (=potentially sampled)
        reachable_counts = np.zeros(self.n, int)
        for item, count in Counter(self.pairs.loc[reachable_indexes, ['item1', 'item2']].to_numpy().flatten()).items():
            reachable_counts[item] = count

        # set the sampling rate inversely-proportional to the count
        self._triplet_sampling_rate = np.zeros(self.n, float)
        self._triplet_sampling_rate[reachable_counts > 0] = 1 / reachable_counts[reachable_counts > 0]
        self._triplet_sampling_rate /= self._triplet_sampling_rate.sum()

        self._triplet_anchors = anchors

        assert np.all(self._triplet_sampling_rate[self._triplet_anchors] > 0)
        assert len(min_counts) == len(self._triplet_sampling_rate) == self.n

    @property
    def triplet_anchors(self) -> NpVec[int]:
        return self._triplet_anchors

    @property
    def triplet_samplable_items(self) -> NpVec[int]:
        return np.nonzero(self._triplet_sampling_rate)[0]

    @property
    def triplet_samplable_indexes_mask(self):
        return self.pairs.loc[:, ['item1', 'item2']].isin(self.triplet_samplable_items).all(axis=1)

    @classmethod
    @verbolize()
    def from_sameness_sign(cls, sameness: Sequence, X: ArrayLike, *args, **kwargs):
        n = len(X)
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

    @property
    def _triplet_sampling_is_initialized(self):
        return self._triplet_sampling_rate is not None and self._triplet_anchors is not None

    def sample_triplet_items(self, anchors: int | Sequence[int], rand_seed: int = None):

        assert self._triplet_sampling_is_initialized, "Triplet sampling not initialized"

        rng = np.random.default_rng(rand_seed)

        if isinstance(anchors, int):
            anchors = self._triplet_anchors[rng.integers(len(self._triplet_anchors), size=anchors)]

        def _sample(item_partners):
            assert len(item_partners)
            cumulative_proba = np.cumsum(self._triplet_sampling_rate[item_partners])
            assert np.all(cumulative_proba > 0)
            cumulative_proba /= cumulative_proba[-1]
            i = np.searchsorted(cumulative_proba, rng.random())
            return item_partners[i]

        positives = []
        negatives = []
        for anchor in anchors:
            assert anchor in self._triplet_anchors
            partners_of_anchor = self.partners_of_item(anchor)
            positives.append(_sample(partners_of_anchor[SamenessData._SAME]))
            negatives.append(_sample(partners_of_anchor[SamenessData._NOTSAME]))

            # done here (and not out size of the loop) because each item can be sampled multiple times
            self.triplet_sampled_counts.loc[anchor, 'anchor'] += 1
            self.triplet_sampled_counts.loc[positives[-1], 'positive'] += 1
            self.triplet_sampled_counts.loc[negatives[-1], 'negative'] += 1

        self.triplet_sampled_counts['total'] = self.triplet_sampled_counts[['anchor', 'positive', 'negative']].sum(axis=1)

        # validate that we're only sampling samplable items..
        is_non_samplable = np.ones(self.n, bool)
        is_non_samplable[self.triplet_samplable_items] = False
        assert not self.triplet_sampled_counts.loc[is_non_samplable, :].any().any()

        return anchors, positives, negatives

    def reset_triplet_sampled_counts(self):
        self.triplet_sampled_counts = pd.DataFrame(data=np.zeros((self.n, 4), int),
                                                   columns=['anchor', 'positive', 'negative', 'total'])

    def sample_triplets(self, anchors: int | Sequence[int], rand_seed: int = None):
        anchors, positives, negatives = self.sample_triplet_items(anchors=anchors, rand_seed=rand_seed)
        A = self.X[anchors]
        P = self.X[positives]
        N = self.X[negatives]
        return A, P, N

    def triplet_summary_string(self) -> str:
        s = "Samplable items: " + strtools.part(len(self.triplet_samplable_items), self.n)
        groups = self.pairs.loc[self.triplet_samplable_indexes_mask, 'group']
        s += "Participating pairs: " + strtools.parts(Same=np.sum(groups == SamenessData._SAME),
                                                      notSame=np.sum(groups == SamenessData._NOTSAME))
        return s

    def make_evaluator(self, n_samples: int = 1000, rand_seed: int = 1, **kwargs) -> EmbeddingEvaluator:
        assert n_samples % 2 == 0
        a, p, n = self.sample_triplet_items(anchors=n_samples // 2, rand_seed=rand_seed)
        return EmbeddingEvaluator.from_triplets(self.X, a, p, n, **kwargs)
