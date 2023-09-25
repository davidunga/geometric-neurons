"""
Manage triplet data & loss
"""

import torch
from copy import deepcopy
import numpy as np
from collections import Counter
from common.symmetric_pairs import SymmetricPairsData
from common.type_utils import *

class TripletLoss(torch.nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def __call__(self, p_dists, n_dists):
        return self.forward(p_dists, n_dists)

    def forward(self, p_dists, n_dists):
        loss = torch.mean(torch.max(p_dists - n_dists + self.margin, torch.zeros_like(p_dists)))
        return loss


class TripletData:

    """
    same_pairs: SymmetricPairsData of same pairs, with dataframe: index, seg1_uid, seg2_uid2
    notSame_pairs: SymmetricPairsData of notSame pairs, with dataframe: index, seg1_uid, seg2_uid2
    X
    """

    def __init__(self,
                 pairs: SymmetricPairsData,
                 sameness: str | Vec[int],
                 X: ArrayLike,
                 batch_size: int = 64,
                 device: str = 'cpu'):

        self.device = device
        self.batch_size = batch_size
        self._batch = 0
        self._epoch = 0
        self._epoch_anchors = None

        MIN_SEGMENT_COUNT = 5  # ignore segments with low prevalence

        if isinstance(sameness, str):
            sameness = pairs.data[sameness]

        self.same_pairs = SymmetricPairsData(pairs.data[sameness == 1], pairs.n)
        self.notSame_pairs = SymmetricPairsData(pairs.data[sameness == -1], pairs.n)

        same_counts = Counter([item for item_pair in self.same_pairs.iter_pairs() for item in item_pair])
        notSame_counts = Counter([item for item_pair in self.notSame_pairs.iter_pairs() for item in item_pair])

        counts = np.zeros(self.same_pairs.n, 'uint64')
        for item, count in same_counts.items():
            if count >= MIN_SEGMENT_COUNT and notSame_counts.get(item, 0) >= MIN_SEGMENT_COUNT:
                counts[item] = max(count, notSame_counts[item])

        self._sampling_proba = np.divide(1, counts, out=np.zeros_like(counts, float), where=counts != 0)
        self._sampling_proba /= self._sampling_proba.sum()

        self._participating_items = np.nonzero(counts)[0]
        self._X = torch.as_tensor(np.asarray(X), device=self.device)

    @property
    def batches_in_epoch(self) -> int:
        return len(self._participating_items)

    def set_batch_and_epoch(self, batch: int, epoch: int):
        assert 0 < batch <= self.batches_in_epoch
        self._batch = batch
        if epoch != self._epoch:
            self._epoch = epoch
            self._epoch_anchors = np.random.default_rng(self._epoch).permutation(self._participating_items)

    def get_next_batch(self):

        if self._epoch == 0 or self._batch == self.batches_in_epoch:
            # new/first epoch
            self.set_batch_and_epoch(batch=1, epoch=self._epoch + 1)
        else:
            # increment batch
            self.set_batch_and_epoch(batch=self._batch + 1, epoch=self._epoch)

        total_batch_count = (self._epoch - 1) * self.batches_in_epoch + self._batch
        rng = np.random.default_rng(total_batch_count)

        def _sample(a, pairs_data):
            items = pairs_data.partners_of_item(a)
            cumulative_proba = np.cumsum(self._sampling_proba[items])
            cumulative_proba /= cumulative_proba[-1]
            i = np.searchsorted(cumulative_proba, rng.random())
            return items[i]

        i_start = (self._batch - 1) * self.batch_size
        i_stop = i_start + self.batch_size
        anchors = self._epoch_anchors[i_start: i_stop]
        positives = [_sample(a, self.same_pairs) for a in anchors]
        negatives = [_sample(a, self.notSame_pairs) for a in anchors]

        A = self._X[anchors]
        P = self._X[positives]
        N = self._X[negatives]

        return A, P, N


if __name__ == "__main__":
    from analysis.data_manager import DataMgr
    data_mgr = DataMgr.from_default_config()
    pairs = data_mgr.load_pairing()
    segmets = data_mgr.load_segments()
    data_mgr.assert_pairs_and_segments_compatibility(pairs, segmets)
    t = TripletData(pairs, 'sameness', X=[s.kin.EuSpd for s in segmets])
    for b in range(50):
        print(b, t.get_next_batch())
