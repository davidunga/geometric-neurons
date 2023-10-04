"""
Manage triplet data & loss
"""
import pandas as pd
import torch
from copy import deepcopy
import numpy as np
from collections import Counter
from common import symmetric_pairs
from common.type_utils import *
from itertools import chain
from sklearn import metrics
from geometric_encoding.embedding import SamenessClassifier, Embedder, embdded_dist_fnc
from scipy import stats


def calc_triplet_loss(p_dists: NDArray, n_dists: NDArray, margin: float = 1.) -> float:
    return float(np.mean(np.max(p_dists - n_dists + margin, np.zeros_like(p_dists))))


class SamenessData(symmetric_pairs.SymmetricPairsData):

    def __init__(self, X: ArrayLike, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = X
        self._same_counts = None
        self._notSame_counts = None

    @classmethod
    def from_sameness_sign(cls, sameness: Sequence, X: ArrayLike):
        n = symmetric_pairs.num_items(len(sameness))
        sameness = sameness[sameness != 0]
        sameness[sameness == -1] = 'notSame'
        sameness[sameness == 1] = 'same'
        return cls(X=X, n=n, group_by=sameness)

    def to(self, *args, **kwargs):
        if not torch.is_tensor(self.X):
            self.X = torch.as_tensor(np.array(self.X), *args, **kwargs)
        else:
            self.X = self.X.to(*args, **kwargs)

    @property
    def same_counts(self) -> Counter:
        if self._same_counts is None:
            self._same_counts = Counter(item for item_pair in self.iter_pairs('same') for item in item_pair)
        return self._same_counts

    @property
    def notSame_counts(self) -> Counter:
        if self._notSame_counts is None:
            self._notSame_counts = Counter(item for item_pair in self.iter_pairs('notSame') for item in item_pair)
        return self._notSame_counts

    def prevalence_counts(self, thresh: int = 0) -> NpVec:
        both_counts = np.zeros((self.n, 2), int)
        for i in range(self.n):
            both_counts[i] = self.same_counts.get(i, 0), self.notSame_counts.get(i, 0)
        counts = both_counts.max(axis=1)
        counts[both_counts.min(axis=1) < thresh] = 0
        return counts


class TripletBatcher:

    def __init__(self,
                 sameness_data: SamenessData,
                 batch_size: int = 64,
                 device: str = 'cpu',
                 min_segment_count: int = 5):

        super().__init__()
        self.batch_size = batch_size
        self.sameness_data = sameness_data
        self._total_samples_count = 0
        self._epoch_samples_count = 0
        self._epoch_anchors = None

        counts = self.sameness_data.prevalence_counts(thresh=min_segment_count)
        self._sampling_proba = np.divide(1, counts, out=np.zeros_like(counts, float), where=counts != 0)
        self._sampling_proba /= self._sampling_proba.sum()
        assert len(counts) == len(self._sampling_proba) == self.sameness_data.n
        self._participating_items = np.nonzero(counts)[0]
        if device:
            self.to(device=device)

    @property
    def batches_in_epoch(self) -> int:
        return len(self._participating_items) // self.batch_size

    def to(self, *args, **kwargs):
        self.sameness_data.to(*args, **kwargs)

    def init_epoch(self, epoch: int):
        self._epoch_samples_count = 0
        self._epoch_anchors = np.random.default_rng(epoch).permutation(self._participating_items)

    def next_batch(self):

        assert self._epoch_anchors is not None, "Not initialized"

        rng = np.random.default_rng(self._total_samples_count)

        def _sample(item_partners):
            cumulative_proba = np.cumsum(self._sampling_proba[item_partners])
            cumulative_proba /= cumulative_proba[-1]
            i = np.searchsorted(cumulative_proba, rng.random())
            return item_partners[i]

        i_start = self._epoch_samples_count
        i_stop = self._epoch_samples_count + self.batch_size
        anchors = self._epoch_anchors[i_start: i_stop]
        positives = [_sample(list(self.sameness_data.iter_partners_of_item(a, 'same'))) for a in anchors]
        negatives = [_sample(list(self.sameness_data.iter_partners_of_item(a, 'notSame'))) for a in anchors]

        A = self.sameness_data.X[anchors]
        P = self.sameness_data.X[positives]
        N = self.sameness_data.X[negatives]

        assert len(A) == len(P) == len(N) == self.batch_size
        self._epoch_samples_count += self.batch_size
        self._total_samples_count += self.batch_size

        return A, P, N





class SamenessEval:

    pred_dist: NpVec[float]
    isSame_gt: NpVec[bool]
    fpr: NpVec[float]
    tpr: NpVec[float]
    ths: NpVec[float]
    auc: float
    loss: float
    ttest: stats._result_classes.TtestResult

    @property
    def tscore(self):
        return 0 if self.ttest.pvalue > .05 else self.ttest.statistic

    def __init__(self, embedder: Embedder, sameness: SamenessData, loss_margin: float):
        self.pred_dist, self.isSame_gt = self.get_gt_and_embedded_dist(embedder, sameness)
        self.fpr, self.tpr, self.ths = metrics.roc_curve(y_true=self.isSame_gt, y_score=self.pred_dist)
        self.auc = metrics.auc(self.fpr, self.tpr)
        p_dists = self.pred_dist[self.isSame_gt]
        n_dists = self.pred_dist[~self.isSame_gt]
        self.ttest = stats.ttest_ind(p_dists, n_dists, alternative='less')
        self.loss = calc_triplet_loss(p_dists=p_dists, n_dists=n_dists, margin=loss_margin)

    def __str__(self):
        s = f"auc={self.auc * 100:2.2f}, tscore={self.tscore:2.1f}"
        if self.loss is not None:
            s += f" loss={self.loss}"
        return s

    def results_dict(self):
        return {"auc": self.auc,
                "tscore": self.tscore,
                "loss": self.loss,
                "ttest.statistic": self.ttest.statistic,
                "ttest.pvalue": self.ttest.pvalue,
                "ttest.confidence_interval": self.ttest.confidence_interval}

    @staticmethod
    def get_gt_and_embedded_dist(embedder: Embedder, sameness: SamenessData):
        eX = embedder(sameness.X)
        emdist, gt = np.array([[np.square(eX[i] - eX[j]).sum(), gt_label == 'same']
                               for (i, j), gt_label in sameness.iter_labeled_pairs()]).T
        emdist = np.sqrt(emdist)
        return emdist, gt






# def eval_classification(embedding_dist, gt):
#     fpr, tpr, ths = metrics.roc_curve(y_true=gt, y_score=pred_score)
#     auc = metrics.auc(fpr, tpr)
#     geometric_mean = tpr * (1 - fpr)
#     optimal_thresh = ths[np.argmax(geometric_mean)]
#     thresh = optimal_thresh if thresh is None else thresh
#     pred = pred_score > thresh
#     acc = metrics.balanced_accuracy_score(y_true=gt, y_pred=pred)
#     pass

if __name__ == "__main__":
    # from analysis.data_manager import DataMgr
    # data_mgr = DataMgr.from_default_config()
    # sameness_data, pairs, segmets = data_mgr.load_sameness()
    # t = TripletBatcher(sameness_data)
    # # same_pairs = SymmetricPairs.from_mask(pairs['sameness'] == 1)
    # # notSame_pairs = SymmetricPairs.from_mask(pairs['sameness'] == -1)
    # # t = TripletData(same_pairs, notSame_pairs, X=[s.kin.EuSpd for s in segmets])
    #
    # for e in range(3):
    #     t.init_epoch(e)
    #     for b in range(10):
    #         print(b, t.next_batch())
