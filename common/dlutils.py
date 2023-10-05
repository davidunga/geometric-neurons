""" Deep-learning utilities """

import torch
from typing import Hashable, Callable, Sequence
from dataclasses import dataclass
import numpy as np
from time import time



def get_optimizer(model_params, opt_params: str | dict):
    if isinstance(opt_params, str): opt_params = {'kind': opt_params}
    return getattr(torch.optim, opt_params['kind'])(
        model_params, **{k: v for k, v in opt_params.items() if k != 'kind'})


class checkpoint:

    @staticmethod
    def load(fname: str):
        items = torch.load(fname)
        model = items['model']
        model.load_state_dict(items['state_dict'])
        optimizer = items['optimizer']
        if optimizer:
            optimizer.load_state_dict(items['optimizer_state'])
        meta = items['meta']
        return model, optimizer, meta

    @staticmethod
    def get_meta(fname: str):
        return torch.load(fname)['meta']

    @staticmethod
    def dump(fname: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, meta: Hashable = None):
        torch.save({'model': model,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer,
                    'optimizer_state': optimizer.state_dict() if optimizer else None,
                    'meta': meta}, fname)


class EarlyStopping:

    @dataclass
    class StopCriterion:
        thresh: float
        check_above: bool
        fn: Callable[[float, float, float], float]
        value: float = .0
        count: int = 0
        _last_val_loss: float = None

        def update(self, val_loss: float, train_loss: float) -> None:
            self.value = self.fn(val_loss, train_loss, self._last_val_loss)
            if (self.value > self.thresh) == self.check_above:
                self.count += 1
            else:
                self.count = 0

    def __init__(self, patience: int | None = 5, converge: float = .001, overfit: float = .2):
        """
        Args:
            patience: number of steps to wait before stopping, once a threshold values is met. None = never.
            converge: convergence threshold
                convergence = (last_val_loss - current_val_loss) / last_val_loss
            overfit: overfit threshold
                overfit = (val_loss - train_loss) / train_loss
        """

        self.patience = patience
        self._stop_reason = ''

        def _converge(val_loss, train_loss, last_val_loss) -> float:
            return 1. - (0 if last_val_loss is None else val_loss / last_val_loss)

        def _overfit(val_loss, train_loss, last_val_loss) -> float:
            return val_loss / max(train_loss, 1e-9) - 1

        self.criteria: dict[str, EarlyStopping.StopCriterion] = {
            'converge': EarlyStopping.StopCriterion(thresh=converge, check_above=False, fn=_converge),
            'overfit': EarlyStopping.StopCriterion(thresh=overfit, check_above=True, fn=_overfit)
        }

    def should_stop(self) -> bool:
        return len(self._stop_reason) > 0

    def stop_reason(self) -> str:
        return self._stop_reason

    def report(self) -> str:
        s = ""
        for criterion in self.criteria:
            s += "{:s}:{:+2.3f} ".format(criterion, self.criteria[criterion].value)
        return s[:-1]

    def process(self, val_loss: float, train_loss: float) -> bool:
        for criterion in self.criteria:
            self.criteria[criterion].update(val_loss, train_loss)
            if self.patience is not None and self.criteria[criterion].count > self.patience:
                self._stop_reason = criterion
        return self.should_stop()


class BatchManager:

    def __init__(self, items: int | Sequence, batch_size: int = 64, shuffle: bool = True):
        self._items = np.arange(items) if isinstance(items, int) else items
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch_items = self._items

    @property
    def batches_in_epoch(self) -> int:
        return len(self._items) // self.batch_size

    def init_epoch(self, epoch: int | None = None) -> None:
        if self.shuffle:
            assert epoch is not None
            self.epoch_items = np.random.default_rng(epoch).permutation(self._items)
        else:
            assert epoch is None

    def get_items(self, batch: int) -> Sequence:
        assert 0 <= batch < self.batches_in_epoch, "Batch out of range"
        start = batch * self.batch_size
        stop = start + self.batch_size
        return self.epoch_items[start: stop]


if __name__ == "__main__":
    pass
