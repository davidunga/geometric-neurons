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
from scipy import sparse
from copy import deepcopy
from common.utils.devtools import verbolize
from common.utils import strtools
from common.pairwise.embedding_eval import EmbeddingEvaluator, make_triplet_pairs
from torch import Tensor


class TripletLossWithIntermediates(torch.nn.Module):

    def __init__(self, margin: float, detach_intermediates: bool = True):
        super().__init__()
        self.margin = margin
        self.detach_intermediates = detach_intermediates

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor
                ) -> tuple[Tensor, Tensor | NDArray, Tensor | NDArray, Tensor | NDArray]:
        return triplet_loss_with_intermediates(anchor, positive, negative, self.margin, self.detach_intermediates)


def triplet_loss_with_intermediates(
        anchor: Tensor, positive: Tensor, negative: Tensor, margin: float,
        detach_intermediates: bool = True) -> tuple[Tensor, Tensor | NDArray, Tensor | NDArray, Tensor | NDArray]:

    pos_dists = torch.pairwise_distance(anchor, positive)
    neg_dists = torch.pairwise_distance(anchor, negative)
    losses = torch.clamp_min(margin + pos_dists - neg_dists, 0)
    loss = torch.mean(losses)
    if detach_intermediates:
        losses = losses.detach().numpy()
        pos_dists = pos_dists.detach().numpy()
        neg_dists = neg_dists.detach().numpy()
    return loss, losses, pos_dists, neg_dists


def validate_pairs_dataframe(df: pd.DataFrame):
    assert 'seg1' in df.columns
    assert 'seg2' in df.columns
    assert 'isSame' in df.columns
    assert 'dist' in df.columns
    assert 'num_segments' in df.attrs
    assert 'num_pairs' in df.attrs


def pairs_to_matrix(pairs_df: pd.DataFrame, value_col: str, map_zero_value=None, dtype=None) -> sparse.csr_array:
    validate_pairs_dataframe(pairs_df)
    values = pairs_df[value_col].to_numpy(dtype=dtype)
    if map_zero_value:
        values[values == 0] = map_zero_value
    n = pairs_df.attrs['num_segments']
    i, j = pairs_df[['seg1', 'seg2']].to_numpy().T
    mtx = sparse.coo_array((values, (i, j)), (n, n)).tocsr()
    return mtx


class TripletSampler:

    def __init__(self, sameness_mtx: sparse.csr_array, dist_mtx: sparse.csr_array, p_hard: float = 0):
        self.p_hard = p_hard
        self._sameness_mtx = sameness_mtx
        self._uniform_probas = TripletSampler._calc_sampling_probas(sameness_mtx, None)
        self._dist_mtx = None
        self._hard_probas = None
        self.included_items = sorted(list(set(np.r_[sameness_mtx.nonzero()].flatten())))
        self.update_dist_mtx(dist_mtx)

    @classmethod
    def from_pairs_df(cls, pairs: pd.DataFrame, **kwargs):
        sameness_mtx = pairs_to_matrix(pairs, 'isSame', map_zero_value=-1, dtype=int)
        dist_mtx = pairs_to_matrix(pairs, 'dist')
        return cls(sameness_mtx=sameness_mtx, dist_mtx=dist_mtx, **kwargs)

    def get_dist_matrix(self) -> sparse.csr_array:
        return self._dist_mtx

    def update_dist_mtx(self, dist_mtx: sparse.csr_array):
        self._dist_mtx = dist_mtx
        self._hard_probas = TripletSampler._calc_sampling_probas(self._sameness_mtx, self._dist_mtx)

    @staticmethod
    def _calc_sampling_probas(sameness_mtx, dist_mtx):

        same_p = (sameness_mtx == 1).astype(float)
        notSame_p = (sameness_mtx == -1).astype(float)

        if dist_mtx is not None:
            normalized_dist_mtx = dist_mtx - dist_mtx.min()
            normalized_dist_mtx /= normalized_dist_mtx.max()
            same_p = same_p.multiply(normalized_dist_mtx)
            notSame_p = notSame_p.multiply(normalized_dist_mtx)
            notSame_p.data = 1 - notSame_p.data

        same_p.eliminate_zeros()
        notSame_p.eliminate_zeros()

        same_p_marginal = np.asarray(same_p.sum(axis=1)).squeeze()
        notSame_p_marginal = np.asarray(notSame_p.sum(axis=1)).squeeze()

        anchor_p = same_p_marginal * notSame_p_marginal
        anchor_p /= anchor_p.sum()

        row_indices, _ = same_p.nonzero()
        same_p.data /= same_p_marginal[row_indices]

        row_indices, _ = notSame_p.nonzero()
        notSame_p.data /= notSame_p_marginal[row_indices]

        return anchor_p, same_p, notSame_p

    def sample_uniform(self, n: int, rand_state: RandState) -> NDArray[int]:
        return _sample_by_probas(*self._uniform_probas, n, rand_state)

    def sample_hard(self, n: int, rand_state: RandState) -> NDArray[int]:
        return _sample_by_probas(*self._hard_probas, n, rand_state)

    def sample(self, n: int, rand_state: RandState) -> NDArray[int]:
        rng = np.random.default_rng(rand_state)
        n_hard = int(round(self.p_hard * n))
        is_hard = (np.arange(n) < (n - n_hard)).astype(int)
        triplets = np.r_[self.sample_uniform(n - n_hard, rng), self.sample_hard(n_hard, rng)]
        triplets = np.c_[triplets, is_hard]
        triplets = rng.permutation(triplets, axis=0)
        return triplets

    @property
    def n_anchors(self) -> int:
        anchor_p, _, _ = self._uniform_probas
        return int(np.sum(anchor_p > 0))

    @property
    def n_total_items(self) -> int:
        return len(self._uniform_probas[0])

    def summary_string(self) -> str:
        anchor_p, same_p, notSame_p = self._uniform_probas
        s = "Included items: " + strtools.part(len(self.included_items), self.n_total_items)
        s += ", Anchors: " + strtools.part(self.n_anchors, len(self.included_items))
        s += ", Pairs: " + strtools.parts(Same=same_p.nnz, notSame=notSame_p.nnz)
        return s


def _sample_by_probas(anchor_p, same_p, notSame_p, n, rand_state) -> NDArray[int]:
    n_total_items = len(anchor_p)
    rng = np.random.default_rng(rand_state)
    triplets = np.zeros((n, 3), int)
    for i, a in enumerate(rng.choice(n_total_items, size=n, p=anchor_p)):
        p = rng.choice(n_total_items, size=1, p=same_p.getrow(a).todense().flatten())[0]
        n = rng.choice(n_total_items, size=1, p=notSame_p.getrow(a).todense().flatten())[0]
        triplets[i] = a, p, n
    return triplets


#
# class TripletBatcher:
#
#     def __init__(self, isSame_matrix: sparse.coo_matrix, dist_matrix: sparse.coo_matrix = None,
#                  p_hard: float = 0.5, batch_size: int = 64, rand_state: RandState = None):
#         self.batch_size = batch_size
#         self.p_hard = p_hard
#         self.regular_sampler = TripletSampler(isSame_matrix=isSame_matrix)
#         self.hard_sampler = TripletSampler(isSame_matrix=isSame_matrix, dist_matrix=dist_matrix)
#         self.rng = None
#         self.set_random_state(rand_state)
#
#     @classmethod
#     def from_pairs_df(cls, pairs: pd.DataFrame, **kwargs):
#         isSame_matrix = pairs_to_matrix(pairs, 'isSame')
#         dist_matrix = pairs_to_matrix(pairs, 'dist')
#         return cls(isSame_matrix=isSame_matrix, dist_matrix=dist_matrix, **kwargs)
#
#     def update_dist_matrix(self, dist_matrix: sparse.coo_matrix):
#         self._dist_matrix = dist_matrix
#         self._init_sampling_probas()
#
#     def set_random_state(self, rand_state: RandState):
#         self.rng.set_random_state(rand_state)
#
#     def sample(self):
#         n_triplets = self.batch_size
#         n_hard = int(round(self.p_hard * n_triplets))
#         n_regular = n_triplets - n_hard
#         regular_triplets = self.regular_sampler.sample(n_regular, rand_state=self.rng)
#         hard_triplets = self.regular_sampler.sample(n_hard, rand_state=self.rng)
#         triplets = regular_triplets + hard_triplets
#         triplets = [triplets[i] for i in self.rng.permutation(n_triplets)]
#         return triplets

    # def update_distances(self, distance_matrix):
    #     normalized_dist_mtx = 1e-6 - distance_matrix.min()
    #     normalized_dist_mtx /= normalized_dist_mtx.max()
    #     self.normalized_dist_mtx = normalized_dist_mtx

    #
    # def sample(self, distance_matrix: NDArray, n_triplets: int, seed: int):
    #     n_hard = int(round(self.p_hard * n_triplets))
    #     n_regular = n_triplets - n_hard
    #
    #     rng = np.random.default_rng(seed)
    #
    #     same_p = distance_matrix
    #     notSame_p = 1 - distance_matrix
    #
    #     regular_triplets = self._sample_by_probas(same_p=same_p, notSame_p=notSame_p, n_triplets=n_regular, rng=rng)
    #
    #     same_p = self.normalized_dist_mtx * self.similarity_matrix
    #     notSame_p = (1 - self.normalized_dist_mtx) * (1 - self.similarity_matrix)
    #     hard_triplets = self._sample_by_probas(same_p=same_p, notSame_p=notSame_p, n_triplets=n_hard, rng=rng)
    #
    #     triplets = regular_triplets + hard_triplets
    #     triplets = [triplets[i] for i in rng.permutation(n_triplets)]
    #     return triplets
    #
    # @staticmethod
    # def _sample_by_probas(same_p, notSame_p, n_triplets, rng):
    #
    #     anchor_p = same_p.sum(axis=1) * notSame_p.sum(axis=1)
    #     anchor_p /= anchor_p.sum()
    #     same_p /= same_p.sum(axis=1, keepdims=True)
    #     notSame_p /= notSame_p.sum(axis=1, keepdims=True)
    #
    #     triplets = []
    #     for _ in range(n_triplets):
    #         anchor_ix = rng.choice(anchor_p)
    #         positive_ix = rng.choice(same_p[anchor_ix])
    #         negative_ix = rng.choice(notSame_p[anchor_ix])
    #         triplets.append((anchor_ix, positive_ix, negative_ix))
    #
    #     return triplets
    #

#
# class TripletSampler:
#
#     def __init__(self, pairs_df: pd.DataFrame, n_samples: int, seed: int = 1):
#         n_samples = 100
#         import time
#         t = time.time()
#
#         rng = np.random.default_rng(seed)
#
#         n_items = symmetric_pairs.num_items(len(pairs_df))
#         same_indexes = set(pairs_df.loc[pairs_df['sameness'] == 1].index)
#         notSame_indexes = set(pairs_df.loc[pairs_df['sameness'] == -1].index)
#
#         pairs_df['seg1'] = [s[0] for s in symmetric_pairs.iter_pairs(n_items)]
#         pairs_df['seg2'] = [s[1] for s in symmetric_pairs.iter_pairs(n_items)]
#
#         def _random_pop(a, indexes_subset):
#             candidate_indexes = indexes_subset.intersection(symmetric_pairs.iter_indexes_of_item(a, n_items))
#             candidate_indexes = list(candidate_indexes)
#             index = candidate_indexes[rng.integers(len(candidate_indexes))]
#             s = int(set(pairs_df.loc[index, ['seg1', 'seg2']]).difference([a]).pop())
#             indexes_subset.difference_update(symmetric_pairs.iter_indexes_of_item(a, n_items))
#             indexes_subset.difference_update(symmetric_pairs.iter_indexes_of_item(s, n_items))
#             return s
#
#         self.triplets = []
#         for _ in range(n_samples):
#             segments_with_same = set(pairs_df.loc[list(same_indexes), ['seg1', 'seg2']].values.reshape(-1))
#             segments_with_notSame = set(pairs_df.loc[list(notSame_indexes), ['seg1', 'seg2']].values.reshape(-1))
#             anchor_candidates = list(segments_with_same.intersection(segments_with_notSame))
#             a = anchor_candidates[rng.integers(len(anchor_candidates))]
#             p = _random_pop(a, same_indexes)
#             n = _random_pop(a, notSame_indexes)
#             self.triplets.append((a, p, n))
#
#         print("t:",time.time()-t)
#
#
#
# class SamenessData(symmetric_pairs.SymmetricPairsData):
#
#     _SAME = 'same'
#     _NOTSAME = 'notSame'
#
#     def __init__(self,
#                  X: ArrayLike,
#                  triplet_min_prevalence: int = 5,
#                  same_counts: Counter = None,
#                  notSame_counts: Counter = None,
#                  *args, **kwargs):
#
#         super().__init__(*args, **kwargs)
#         self.X = np.asarray(X)
#         self._triplet_min_prevalence = triplet_min_prevalence
#         self._same_counts = same_counts
#         self._notSame_counts = notSame_counts
#         self._triplet_sampling_rate = None
#         self._triplet_anchors = None
#         self.triplet_sampled_counts = None
#
#     def modcopy(self, include_items: Container[int] = None, index_mask: Sequence[bool] = None):
#         """ make modified copy """
#
#         if include_items is not None:
#             assert index_mask is None
#             index_mask = self.mutex_indexes(split_items=include_items) == 1
#
#         pairs = self.pairs.copy()
#         if index_mask is not None:
#             pairs.loc[~index_mask, 'group'] = None
#             same_counts = None
#             notSame_counts = None
#         else:
#             same_counts = deepcopy(self._same_counts)
#             notSame_counts = deepcopy(self._notSame_counts)
#
#         return SamenessData(X=self.X.copy(),
#                             n=self.n,
#                             data=self.data.copy() if self.data is not None else None,
#                             pairs=pairs,
#                             triplet_min_prevalence=self._triplet_min_prevalence,
#                             same_counts=same_counts,
#                             notSame_counts=notSame_counts)
#
#     def init_triplet_sampling(self, anchors: Sequence[int] = None):
#
#         self.reset_triplet_sampled_counts()
#
#         min_counts = np.zeros(self.n, int)
#         max_counts = np.zeros(self.n, int)
#         for item in range(self.n):
#             min_counts[item], max_counts[item] = self.same_counts.get(item, 0), self.notSame_counts.get(item, 0)
#             if min_counts[item] > max_counts[item]:
#                 min_counts[item], max_counts[item] = max_counts[item], min_counts[item]
#
#         can_be_anchor = min_counts >= self._triplet_min_prevalence
#         if anchors is not None:
#             assert can_be_anchor[anchors].all(), "Some of the specified anchors do not have sufficient counts"
#         else:
#             anchors = np.nonzero(can_be_anchor)[0]
#
#         # index of pairs that are reachable through achors, i.e., are anchors or pairs of an anchor
#         is_reachable = self.pairs.loc[self.pairs['group'].notna(), ['item1', 'item2']].isin(anchors).any(axis=1)
#         reachable_indexes = is_reachable[is_reachable].index
#
#         # count how many times each item can be reached (=potentially sampled)
#         reachable_counts = np.zeros(self.n, int)
#         for item, count in Counter(self.pairs.loc[reachable_indexes, ['item1', 'item2']].to_numpy().flatten()).items():
#             reachable_counts[item] = count
#
#         # set the sampling rate inversely-proportional to the count
#         self._triplet_sampling_rate = np.zeros(self.n, float)
#         self._triplet_sampling_rate[reachable_counts > 0] = 1 / reachable_counts[reachable_counts > 0]
#         self._triplet_sampling_rate /= self._triplet_sampling_rate.sum()
#
#         self._triplet_anchors = anchors
#
#         assert np.all(self._triplet_sampling_rate[self._triplet_anchors] > 0)
#         assert len(min_counts) == len(self._triplet_sampling_rate) == self.n
#
#     @property
#     def triplet_anchors(self) -> NpVec[int]:
#         return self._triplet_anchors
#
#     @property
#     def triplet_samplable_items(self) -> NpVec[int]:
#         return np.nonzero(self._triplet_sampling_rate)[0]
#
#     @property
#     def triplet_samplable_indexes_mask(self):
#         return self.pairs.loc[:, ['item1', 'item2']].isin(self.triplet_samplable_items).all(axis=1)
#
#     @classmethod
#     @verbolize()
#     def from_sameness_sign(cls, sameness: Sequence, X: ArrayLike, *args, **kwargs):
#         n = len(X)
#         sameness = sameness[sameness != 0]
#         sameness[sameness == -1] = SamenessData._NOTSAME
#         sameness[sameness == 1] = SamenessData._SAME
#         return cls(X=X, n=n, group_by=sameness, *args, **kwargs)
#
#     def to(self, *args, **kwargs):
#         if not torch.is_tensor(self.X):
#             self.X = torch.as_tensor(np.asarray(self.X), *args, **kwargs)
#         else:
#             self.X = self.X.to(*args, **kwargs)
#
#     @property
#     def same_counts(self) -> Counter:
#         if self._same_counts is None:
#             self._same_counts = Counter(self.item_pairs(label=SamenessData._SAME).flatten().tolist())
#         return self._same_counts
#
#     @property
#     def notSame_counts(self) -> Counter:
#         if self._notSame_counts is None:
#             self._notSame_counts = Counter(self.item_pairs(label=SamenessData._NOTSAME).flatten().tolist())
#         return self._notSame_counts
#
#     @property
#     def _triplet_sampling_is_initialized(self):
#         return self._triplet_sampling_rate is not None and self._triplet_anchors is not None
#
#     def sample_triplet_items(self, anchors: int | Sequence[int], rand_seed: int = None):
#
#         assert self._triplet_sampling_is_initialized, "Triplet sampling not initialized"
#
#         rng = np.random.default_rng(rand_seed)
#
#         if isinstance(anchors, int):
#             anchors = self._triplet_anchors[rng.integers(len(self._triplet_anchors), size=anchors)]
#
#         def _sample(item_partners):
#             assert len(item_partners)
#             cumulative_proba = np.cumsum(self._triplet_sampling_rate[item_partners])
#             assert np.all(cumulative_proba > 0)
#             cumulative_proba /= cumulative_proba[-1]
#             i = np.searchsorted(cumulative_proba, rng.random())
#             return item_partners[i]
#
#         positives = []
#         negatives = []
#         for anchor in anchors:
#             assert anchor in self._triplet_anchors
#             partners_of_anchor = self.partners_of_item(anchor)
#             positives.append(_sample(partners_of_anchor[SamenessData._SAME]))
#             negatives.append(_sample(partners_of_anchor[SamenessData._NOTSAME]))
#
#             # done here (and not out size of the loop) because each item can be sampled multiple times
#             self.triplet_sampled_counts.loc[anchor, 'anchor'] += 1
#             self.triplet_sampled_counts.loc[positives[-1], 'positive'] += 1
#             self.triplet_sampled_counts.loc[negatives[-1], 'negative'] += 1
#
#         self.triplet_sampled_counts['total'] = self.triplet_sampled_counts[['anchor', 'positive', 'negative']].sum(axis=1)
#
#         # validate that we're only sampling samplable items..
#         is_non_samplable = np.ones(self.n, bool)
#         is_non_samplable[self.triplet_samplable_items] = False
#         assert not self.triplet_sampled_counts.loc[is_non_samplable, :].any().any()
#
#         return anchors, positives, negatives
#
#     def reset_triplet_sampled_counts(self):
#         self.triplet_sampled_counts = pd.DataFrame(data=np.zeros((self.n, 4), int),
#                                                    columns=['anchor', 'positive', 'negative', 'total'])
#
#     def sample_triplets(self, anchors: int | Sequence[int], rand_seed: int = None):
#         anchors, positives, negatives = self.sample_triplet_items(anchors=anchors, rand_seed=rand_seed)
#         A = self.X[anchors]
#         P = self.X[positives]
#         N = self.X[negatives]
#         return A, P, N
#
#     def triplet_summary_string(self) -> str:
#         s = "Samplable items: " + strtools.part(len(self.triplet_samplable_items), self.n)
#         groups = self.pairs.loc[self.triplet_samplable_indexes_mask, 'group']
#         s += "Participating pairs: " + strtools.parts(Same=np.sum(groups == SamenessData._SAME),
#                                                       notSame=np.sum(groups == SamenessData._NOTSAME))
#         return s
#
#     def make_evaluator(self, n_samples: int = 1000, rand_seed: int = 1, **kwargs) -> EmbeddingEvaluator:
#         assert n_samples % 2 == 0
#         a, p, n = self.sample_triplet_items(anchors=n_samples // 2, rand_seed=rand_seed)
#         return EmbeddingEvaluator.from_triplets(self.X, a, p, n, **kwargs)
