import numpy as np
from scipy import sparse
from common.utils import strtools
from common.utils.typings import *


class TripletSampler:
    """ Manages sampling of (anchor, positive, negative) triplets for metric learning
        Supports hard sampling
    """

    def __init__(self, sameness_mtx: sparse.csr_array,
                 dist_mtx: sparse.csr_array = None, p_hard: float = 0):
        """
        Initialize triplet sampling from N items.
        Args:
            sameness_mtx: sparse N*N matrix with values (-1, 1):
                sameness_mtx[i,j] = 1 <-> i & j are the same
                sameness_mtx[i,j] = -1 <-> i & j are not the same
                sameness_mtx[i,j] = 0 <-> i & j is not a valid pair (ignored)
            dist_mtx: sparse N*N matrix of pairwise distances, used for hard sampling.
            p_hard: proportion of hard triplets in each call to sample()
        """

        self.p_hard = p_hard
        self._sameness_mtx = sameness_mtx
        self._dist_mtx = None
        self._uniform_probas = TripletSampler._calc_sampling_probas(sameness_mtx, None)
        self._hard_probas = None
        self.included_items = sorted(list(set(np.r_[sameness_mtx.nonzero()].flatten())))
        self.update_dist_mtx(dist_mtx)

    def get_dist_matrix(self) -> sparse.csr_array:
        return self._dist_mtx

    def update_dist_mtx(self, dist_mtx: sparse.csr_array):
        self._dist_mtx = dist_mtx
        self._hard_probas = TripletSampler._calc_sampling_probas(self._sameness_mtx, self._dist_mtx)

    @staticmethod
    def _calc_sampling_probas(sameness_mtx, dist_mtx):
        """
        returns:
            anchor_p: anchor_p[i] = chance of sampling item i as anchor
            same_p: same_p[i,j] = chance of sampling item j as positive, for anchor i
            notSame_p: notSame_p[i,j] = chance of sampling item j as negative, for anchor i
        """

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
        """ sample triplets, ignore distance information """
        return _sample_triplets_by_probas(*self._uniform_probas, n, rand_state)

    def sample_hard(self, n: int, rand_state: RandState) -> NDArray[int]:
        """ sample hard triplets, based on distances """
        return _sample_triplets_by_probas(*self._hard_probas, n, rand_state)

    def sample(self, n: int, rand_state: RandState) -> NDArray[int]:
        """ sample triplets, number of hard triplets determined by p_hard """
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


def _sample_triplets_by_probas(anchor_p, same_p, notSame_p, n, rand_state) -> NDArray[int]:
    """
    sample n triplets out of N items
    Args:
        anchor_p: 1*N array, chance of sampling each item as anchor
        same_p: N*N matrix, same_p[i,j] chance of sampling j as positive of anchor i
        notSame_p: N*N matrix, notSame_p[i,j] chance of sampling j as negative of anchor i
        n: number of triplets to sample
        rand_state: random seed or generator
    Returns:
        n*3 numpy int array, each row = [anchor, positive, negative]
    """
    n_total_items = len(anchor_p)
    rng = np.random.default_rng(rand_state)
    triplets = np.zeros((n, 3), int)
    for i, a in enumerate(rng.choice(n_total_items, size=n, p=anchor_p)):
        p = rng.choice(n_total_items, size=1, p=same_p.getrow(a).todense().flatten())[0]
        n = rng.choice(n_total_items, size=1, p=notSame_p.getrow(a).todense().flatten())[0]
        triplets[i] = a, p, n
    return triplets

