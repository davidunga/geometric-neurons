import torch
from torch import Tensor
from common.utils.typings import *


class TripletLossWithIntermediates(torch.nn.Module):
    """
    triplet loss which also returns the loss and distances for each triplet
    """
    def __init__(self, margin: float, detach_intermediates: bool = True):
        """
        Args:
            margin: the triplet loss margin
            detach_intermediates: return the per-triplet values as detached numpy arrays
        """
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
        losses = losses.detach().cpu().numpy()
        pos_dists = pos_dists.detach().cpu().numpy()
        neg_dists = neg_dists.detach().cpu().numpy()
    return loss, losses, pos_dists, neg_dists
