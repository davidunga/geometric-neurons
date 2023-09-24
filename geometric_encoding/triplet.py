"""
Manage triplet data & loss
"""

import torch
from copy import deepcopy
import numpy as np
from data import get_samples, DATASPLIT
from utils import utils


class TripletLoss(torch.nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def __call__(self, p_dists, n_dists):
        return self.forward(p_dists, n_dists)

    def forward(self, p_dists, n_dists):
        loss = torch.mean(torch.max(p_dists - n_dists + self.margin, torch.zeros_like(p_dists)))
        return loss


class TripletBatches:
    """
    Batch manager for triplet data
    """

    def __init__(self, X: torch.FloatTensor, y, pair_ixs, batch_size, noise_sig=0, split=None,
                 y_dists=None, ctrl_dists=None):
        self.X = X
        self.batch_size = batch_size
        self.same_pairs, self.notSame_pairs = self._make_pairs(pair_ixs, y)
        self.triplets = []
        self.device = self.X.device
        self.noise_sig = noise_sig
        self.split = split
        self.y_dists = None
        self.ctrl_dists = None
        self._seg_pair_to_ix = utils.make_seg_pair_dict(pair_ixs, symm=True)
        if y_dists is not None:
            self.y_dists = y_dists
        if ctrl_dists is not None:
            self.ctrl_dists = np.stack(ctrl_dists, axis=1)

    def make_epoch(self):
        self.triplets = self._make_triplets(deepcopy(self.same_pairs), deepcopy(self.notSame_pairs))
        self.triplets = np.random.permutation(self.triplets)

        # sanity
        for triplet_ix in range(0, len(self.triplets), int(len(self.triplets) / 100)):
            a, p, n = self.triplets[triplet_ix]
            assert p in self.same_pairs[a]
            assert p not in self.notSame_pairs[a]
            assert n in self.notSame_pairs[a]
            assert n not in self.same_pairs[a]

    @staticmethod
    def _make_pairs(pair_ixs, y):
        num_segs = np.max(pair_ixs) + 1

        same_pairs = [[] for _ in range(num_segs)]
        label_pairs = pair_ixs[y]
        for i_seg in range(num_segs):
            ii = np.any(label_pairs == i_seg, axis=1)
            if np.any(ii):
                pair_segments = label_pairs[ii].flatten()
                same_pairs[i_seg] = pair_segments[pair_segments != i_seg]

        notSame_pairs = [[] for _ in range(num_segs)]
        label_pairs = pair_ixs[~y]
        for i_seg in range(num_segs):
            ii = np.any(label_pairs == i_seg, axis=1)
            if np.any(ii):
                pair_segments = label_pairs[ii].flatten()
                notSame_pairs[i_seg] = pair_segments[pair_segments != i_seg]

        for i_seg in range(num_segs):
            assert len(np.intersect1d(same_pairs[i_seg], notSame_pairs[i_seg])) == 0

        return same_pairs, notSame_pairs

    @staticmethod
    def _make_triplets(same_pairs, notSame_pairs):
        anchor_candidates = [len(same_pairs[i_seg]) and len(notSame_pairs[i_seg])
                             for i_seg in range(len(same_pairs))]
        triplets = []
        while np.any(anchor_candidates):
            for anchor_seg in np.random.permutation(np.nonzero(anchor_candidates)[0]):
                ix = np.random.choice(len(same_pairs[anchor_seg]))
                same_seg = same_pairs[anchor_seg][ix]
                same_pairs[anchor_seg] = np.delete(same_pairs[anchor_seg], ix)
                ix = np.random.choice(len(notSame_pairs[anchor_seg]))
                notSame_seg = notSame_pairs[anchor_seg][ix]
                notSame_pairs[anchor_seg] = np.delete(notSame_pairs[anchor_seg], ix)
                if not (len(same_pairs[anchor_seg]) and len(notSame_pairs[anchor_seg])):
                    anchor_candidates[anchor_seg] = False
                triplets.append((anchor_seg, same_seg, notSame_seg))

        return triplets

    def __len__(self):
        return len(self.triplets) // self.batch_size

    def __str__(self):
        s1 = f"{self.__class__.__name__}: {self.split.name.upper()}."
        s2 = f"{len(self)} batches, batchSize={self.batch_size}, triplets={len(self.triplets)}"
        return s1 + " " + s2

    def __getitem__(self, item):
        assert len(self.triplets)
        A = torch.zeros((self.batch_size, self.X.shape[1]), device=self.device)
        P = torch.zeros_like(A)
        N = torch.zeros_like(A)
        i_from = item * self.batch_size
        i_to = i_from + self.batch_size

        dists = {'P': None, 'N': None, 'ctrl_P': None, 'ctrl_N': None}
        if self.y_dists is not None:
            dists['P'] = np.zeros(len(A))
            dists['N'] = np.zeros(len(A))
        if self.ctrl_dists is not None:
            num_ctrl_vars = len(next(iter(self.ctrl_dists.values())))
            dists['ctrl_P'] = np.zeros((len(A), num_ctrl_vars))
            dists['ctrl_N'] = np.zeros((len(A), num_ctrl_vars))

        for i, triplet in enumerate(self.triplets[i_from:i_to]):
            a, p, n = triplet
            A[i] = self.X[a]
            P[i] = self.X[p]
            N[i] = self.X[n]
            if self.y_dists is not None:
                dists['P'][i] = self.y_dists[self._seg_pair_to_ix[(a, p)]]
                dists['N'][i] = self.y_dists[self._seg_pair_to_ix[(a, n)]]
            if self.ctrl_dists is not None:
                dists['ctrl_P'][i] = self.ctrl_dists[self._seg_pair_to_ix[(a, p)]]
                dists['ctrl_N'][i] = self.ctrl_dists[self._seg_pair_to_ix[(a, n)]]
        if self.noise_sig > 0:
            P += self.noise_sig * torch.randn(P.size(), device=self.device)
            N += self.noise_sig * torch.randn(N.size(), device=self.device)
            A += self.noise_sig * torch.randn(A.size(), device=self.device)

        return A, P, N, dists


def make_triplet_batches(cfg, device, splits, ctrls):

    if isinstance(splits, DATASPLIT):
        splits = (splits,)
    assert all([isinstance(split, DATASPLIT) for split in splits])

    y, split_ixs, samples_meta, data = get_samples(cfg)
    S = torch.from_numpy(data['S']).type(torch.FloatTensor).to(device)
    pair_ixs = data['pair_ixs']

    batches = {}
    ctrl_dists = None
    for split in splits:
        ii = split_ixs == split
        noise_sig = cfg['train']['pepper_sigma'] if split == DATASPLIT.TRAIN else 0
        if ctrls:
            ctrl_dists = list(v[ii] for v in data['ctrl_dists'].values())
        batches[split] = TripletBatches(
            S, y[ii], pair_ixs[ii], batch_size=cfg['train']['batch_size'],
            noise_sig=noise_sig, split=split, y_dists=data['y_dists'][ii],
            ctrl_dists=ctrl_dists)

    return samples_meta, [batches[s] for s in splits]
