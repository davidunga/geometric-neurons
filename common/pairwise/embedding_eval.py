import numpy as np
import torch.nn
from common.utils.typings import *
from sklearn import metrics
from scipy import stats
from dataclasses import dataclass
from common.utils import strtools


@dataclass
class EmbeddingEvalResult:

    _tscore_alpha = .01

    fpr: NpVec[float]
    tpr: NpVec[float]
    ths: NpVec[float]
    ttest: Any
    loss: float | None = None

    @property
    def auc(self) -> float:
        return self._auc

    @property
    def tscore(self) -> float:
        return self._tscore

    @property
    def metrics_dict(self) -> dict:
        return {"auc": self.auc,
                "tscore": self.tscore,
                "loss": self.loss,
                "ttest-t": self.ttest.statistic,
                "ttest-p": self.ttest.pvalue,
                "ttest-ci": self.ttest.confidence_interval()}

    def __post_init__(self):
        self._auc = metrics.auc(self.fpr, self.tpr)
        self._tscore = 0.0 if self.ttest.pvalue > self._tscore_alpha else self.ttest.statistic

    def __str__(self) -> str:
        loss_str = f", loss={self.loss:2.4f}" if self.loss is not None else ""
        return f"auc={self.auc * 100:2.2f}, tscore={self.tscore:2.1f}{loss_str}"


class EmbeddingLoss:

    def __init__(self, kind: str, margin: float):
        assert kind in ("triplet", "contrastive")
        self.kind = kind
        self.margin = margin

    def __call__(self, p_dists: NpVec[float], n_dists: NpVec[float]) -> float:
        if self.kind == "triplet":
            d = np.maximum(p_dists - n_dists + self.margin, 0)
            loss = np.mean(d)
        elif self.kind == "contrastive":
            positive_part = np.square(p_dists).sum()
            negative_part = np.square(np.maximum(self.margin - n_dists, 0)).sum()
            loss = (positive_part + negative_part) / (len(p_dists) + len(n_dists))
        else:
            raise ValueError("Unknown loss kind")
        return float(loss)


class EmbeddingEvaluator:
    """ Evaluates embedding of vectors """

    vecs: NpPoints | torch.TensorType
    paired_items: list[tuple[int, int]]
    is_same: NpVec[bool]
    loss_func: EmbeddingLoss | None

    def __init__(self, vecs: NpPoints | torch.TensorType | None,
                 paired_items: list[tuple[int, int]], is_same: NpVec[bool],
                 n: int = None,
                 loss_margin: float = None, loss_kind: str = "triplet", loss_func: Callable = None):

        """
        Args:
            vecs: vectors (not embedded). if None, must be provided each evaluate() call
            paired_items: pairing of vectors
            is_same: indicates if pair is 'same' or 'not same'
            loss_margin: if provided, loss function is constructed from loss kind + this margin
            loss_kind: "triplet" / "contrastive". ignored if loss_margin is None
            loss_func: loss function. if provided loss_margin must be None
        """

        if loss_margin is not None:
            assert loss_func is None, "Provide either loss margin or loss function, not both"
            loss_func = EmbeddingLoss(kind=loss_kind, margin=loss_margin)

        if vecs is None:
            assert n is not None
        elif n is None:
            n = len(vecs)

        self.vecs = vecs
        self.is_same = is_same
        self.paired_items = paired_items
        self.loss_func = loss_func
        self.n = n
        self.included_items = sorted(list(set(item for pair in self.paired_items for item in pair)))

        assert (self.vecs is None) or (self.n == len(self.vecs))
        assert self.included_items[-1] < n
        assert self.included_items[0] >= 0
        assert len(self.paired_items) == len(self.is_same)

    @classmethod
    def from_triplets(cls, vecs: NpPoints | torch.TensorType | None, triplets: NDArray[int], **kwargs):
        paired_items, is_same = merge_pos_neg(triplets[:, [0, 1]], triplets[:, [0, 2]])
        return cls(vecs=vecs, paired_items=paired_items, is_same=is_same, **kwargs)

    def evaluate(self, embedder: torch.nn.Module | Callable = None, inputs: torch.Tensor = None) -> EmbeddingEvalResult:

        assert (self.vecs is None) ^ (inputs is None)

        if inputs is None:
            inputs = self.vecs

        assert len(inputs) == self.n

        if embedder is None:
            embedded_vecs = inputs
        elif isinstance(embedder, torch.nn.Module):
            is_training = embedder.training
            embedder.train(False)
            embedded_vecs = embedder(inputs)
            embedder.train(is_training)
        else:
            embedded_vecs = embedder(inputs)

        try:
            embedded_vecs = embedded_vecs.detach().cpu().numpy()
        except AttributeError:
            pass

        result = evaluate_embedded_vecs(
            embedded_vecs=embedded_vecs,
            pairs=self.paired_items,
            is_same=self.is_same,
            loss_func=self.loss_func)

        return result

    def summary_string(self) -> str:
        s = "Included items: " + strtools.part(len(self.included_items), self.n)
        s += ", Pairs: " + strtools.parts(Same=self.is_same.sum(), notSame=(~self.is_same).sum())
        return s


def evaluate_embedded_dists(
        dists: NpVec[float],
        is_same: NpVec[bool] = None,
        loss_func: EmbeddingLoss = None) -> EmbeddingEvalResult:

    fpr, tpr, ths = metrics.roc_curve(y_true=~is_same, y_score=dists)
    ttest = stats.ttest_ind(dists[is_same], dists[~is_same], alternative='less')
    result = EmbeddingEvalResult(fpr=fpr, tpr=tpr, ths=ths, ttest=ttest)
    if loss_func is not None:
        result.loss = loss_func(p_dists=dists[is_same], n_dists=dists[~is_same])

    return result


def evaluate_embedded_vecs(
        embedded_vecs: NpPoints,
        pairs: Sequence[tuple[int, int]],
        is_same: NpVec[bool] = None,
        loss_func: EmbeddingLoss = None) -> EmbeddingEvalResult:
    dists = pairs_dists(embedded_vecs, pairs)
    assert len(pairs) == len(is_same) == len(dists)
    return evaluate_embedded_dists(dists, is_same, loss_func)


def merge_pos_neg(p: Sequence, n: Sequence) -> [NpVec[float], NpVec[bool]]:
    merged = np.r_[p, n]
    is_same = np.array(([True] * len(p)) + ([False] * len(n)), bool)
    return merged, is_same


def make_triplet_pairs(anchors: list[int], positives: list[int], negatives: list[int]):
    p_pairs = [(a, i) for a, i in zip(anchors, positives)]
    n_pairs = [(a, i) for a, i in zip(anchors, negatives)]
    pairs, is_same = merge_pos_neg(p_pairs, n_pairs)
    return pairs, is_same


def pairs_dists(x: NpPoints, pairs: Sequence[tuple[int, int]]) -> NpVec[float]:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    assert x.ndim == 2
    return np.linalg.norm([x[i] - x[j] for (i, j) in pairs], axis=1)


def _test_loss_func():
    from torch.nn.functional import triplet_margin_loss

    max_rel_error = 1e-3

    n_pairs = 1000
    embeddeding_dim = 5
    nrg = np.random.default_rng(0)
    embedded_vecs = nrg.random(size=(3 * n_pairs, embeddeding_dim)) * .001

    anchor_items = np.arange(n_pairs)
    positive_items = np.arange(n_pairs, 2 * n_pairs)
    negatives_items = np.arange(2 * n_pairs, 3 * n_pairs)

    p_pairs = [(a, i) for a, i in zip(anchor_items, positive_items)]
    n_pairs = [(a, i) for a, i in zip(anchor_items, negatives_items)]

    anchors = torch.Tensor(embedded_vecs[anchor_items])
    positives = torch.Tensor(embedded_vecs[positive_items])
    negatives = torch.Tensor(embedded_vecs[negatives_items])

    for loss_kind in ["triplet"]:
        for loss_margin in np.linspace(.001, 5, 100):
            loss_func = EmbeddingLoss(kind=loss_kind, margin=loss_margin)
            pairs, is_same = merge_pos_neg(p_pairs, n_pairs)
            result = evaluate_embedded_vecs(
                embedded_vecs=embedded_vecs, pairs=pairs, is_same=is_same, loss_func=loss_func)
            torch_loss = triplet_margin_loss(anchor=anchors, positive=positives, negative=negatives, margin=loss_margin)
            torch_loss = float(torch_loss)
            rel_error = (result.loss - torch_loss) / torch_loss
            print(f"Torch={torch_loss:2.5f}, Result={result.loss:2.5f}, Error={rel_error:2.5%}")
            assert abs(rel_error) < max_rel_error


def _test():
    import matplotlib.pyplot as plt

    nrg = np.random.default_rng(0)

    min_dist = 0
    max_dist = 1
    n = 500
    thresh_p = .5
    max_noise_sigma = 3 * max_dist
    n_noise_levels = 100
    loss_margin = .1

    true_dists = np.linspace(min_dist, max_dist, n)
    thresh = min_dist + thresh_p * (max_dist - min_dist)
    true_labels = true_dists < thresh

    sig1_noise = nrg.standard_normal(size=true_labels.shape)
    sigmas = np.linspace(0, max_noise_sigma, n_noise_levels)

    _, axs = plt.subplots(ncols=3)

    for loss_kind in ["triplet", "contrastive"]:
        loss_func = EmbeddingLoss(kind=loss_kind, margin=loss_margin)

        results = []
        for sigma in sigmas:
            dists = true_dists + sigma * sig1_noise
            result = evaluate_embedded_dists(dists, is_same=true_labels, loss_func=loss_func)
            results.append(result)

        loss_per_sigma = [r.loss for r in results]
        auc_per_sigma = [r.auc for r in results]

        plt.sca(axs[0])
        plt.plot(sigmas, loss_per_sigma, label=loss_kind)
        plt.xlabel("Noise Sigma")
        plt.title("Loss vs Noise")

        plt.sca(axs[1])
        plt.plot(auc_per_sigma, loss_per_sigma, label=loss_kind)
        plt.xlabel("AUC")
        plt.title("Loss vs AUC")

        # (should be the same for both loss kinds..)
        plt.sca(axs[2])
        plt.plot(sigmas, auc_per_sigma)
        plt.xlabel("Noise Sigma")
        plt.title("AUC vs Noise")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    #_test()
    _test_loss_func()