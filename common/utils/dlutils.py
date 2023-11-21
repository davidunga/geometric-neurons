""" Deep-learning utilities """

import torch
from common.utils.typings import *
from dataclasses import dataclass
import numpy as np


def get_optimizer(model_params, kind: str = 'Adam', **optim_kws):
    return getattr(torch.optim, kind)(model_params, **optim_kws)


class checkpoint:

    @staticmethod
    def load(model_file: PathLike):
        items = checkpoint._safe_load_items(model_file)
        model = items['model']
        model.load_state_dict(items['state_dict'])
        optimizer = items['optimizer']
        if optimizer:
            optimizer.load_state_dict(items['optimizer_state'])
        meta = items['meta']
        return model, optimizer, meta

    @staticmethod
    def get_meta(model_file: PathLike):
        return checkpoint._safe_load_items(model_file)['meta']

    @staticmethod
    def set_meta(model_file: PathLike, meta):
        items = checkpoint._safe_load_items(model_file)
        items['meta'] = meta
        checkpoint._safe_save_items(model_file, items)

    @staticmethod
    def update_meta(model_file: PathLike, **kwargs) -> dict:
        items = checkpoint._safe_load_items(model_file)
        items['meta'].update(kwargs)
        checkpoint._safe_save_items(model_file, items)
        return items

    @staticmethod
    def dump(model_file: PathLike, model: torch.nn.Module,
             optimizer: torch.optim.Optimizer = None, meta: dict = None):
        items = {'model': model,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer,
                 'optimizer_state': optimizer.state_dict() if optimizer else None,
                 'meta': meta}
        checkpoint._safe_save_items(model_file, items)

    @staticmethod
    def _validate_items(items: dict):
        assert set(items.keys()) == {'model', 'state_dict', 'optimizer', 'optimizer_state', 'meta'}
        assert isinstance(items['model'], torch.nn.Module)
        assert isinstance(items['state_dict'], dict)
        assert (items['optimizer'] is None) == (items['optimizer_state'] is None)
        assert items['optimizer'] is None or isinstance(items['optimizer'], torch.optim.Optimizer)
        assert items['optimizer_state'] is None or isinstance(items['optimizer_state'], dict)
        assert items['meta'] is None or isinstance(items['meta'], dict)

    @staticmethod
    def _safe_load_items(model_file: PathLike) -> dict:
        items = torch.load(str(model_file))
        checkpoint._validate_items(items)
        return items

    @staticmethod
    def _safe_save_items(model_file: PathLike, items: dict):
        checkpoint._validate_items(items)
        torch.save(items, str(model_file))


class ProgressManager:

    @dataclass
    class StopCriterion:
        thresh: float
        check_above: bool
        fn: Callable[[float, float, float], float]
        value: float = .0
        count: int = 0
        _last_val_loss: float = None
        _last_train_loss: float = None

        def update(self, val_loss: float, train_loss: float, enable_count: bool) -> None:
            self.value = self.fn(val_loss, train_loss, self._last_val_loss, self._last_train_loss)
            self._last_val_loss = val_loss
            self._last_train_loss = train_loss
            if enable_count and ((self.value > self.thresh) == self.check_above):
                self.count += 1
            else:
                self.count = 0

    def __init__(self, patience=5, converge: float = .001, overfit: float = .2, epochs: int = None, grace_period=0):
        """
        Args:
            patience: number of steps to wait before stopping, once a threshold values is met. None = never.
                if patience value is between 0 & 1, patience = round(value * epochs)
            converge: convergence threshold
                convergence = (last_val_loss - val_loss) / last_val_loss
            overfit: overfit threshold
                overfit = (val_loss - train_loss) / train_loss
            grace_period: steps to wait before starting. None = same as patience.
        """

        if patience and 0 < patience < 1:
            assert epochs is not None, "Patience given as fraction of epochs, but epochs was not specified."
            patience = int(round(patience * epochs))

        if grace_period is None:
            grace_period = patience if patience is not None else 0
        elif 0 < grace_period < 1:
            assert epochs is not None, "Grace period given as fraction of epochs, but epochs was not specified."
            grace_period = int(round(grace_period * epochs))

        self.patience = patience
        self.epochs = epochs
        self.grace_period = grace_period
        self._best_val_score = None
        self._epoch = 0

        self._stop_reason = ''
        self._stop_epoch = None
        self._is_new_nest = False

        def _converge(val_loss, train_loss, last_val_loss, last_train_loss) -> float:
            return 1. if last_val_loss is None else (last_val_loss - val_loss) / last_val_loss

        def _overfit(val_loss, train_loss, last_val_loss, last_train_loss) -> float:
            return val_loss / max(train_loss, 1e-9) - 1

        self.criteria: dict[str, ProgressManager.StopCriterion] = {
            'converge': ProgressManager.StopCriterion(thresh=converge, check_above=False, fn=_converge),
            'overfit': ProgressManager.StopCriterion(thresh=overfit, check_above=True, fn=_overfit)
        }

    @property
    def should_stop(self) -> bool:
        return len(self._stop_reason) > 0

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    @property
    def stop_epoch(self) -> int:
        return self._stop_epoch

    @property
    def is_new_best(self) -> bool:
        return self._is_new_nest

    def report(self) -> str:
        s = ""
        for criterion in self.criteria:
            s += "{:s}:{:+2.3f} ".format(criterion, self.criteria[criterion].value)
        s = "[" + s[:-1] + "]"
        if self.is_new_best:
            s += " -- New best"
        return s

    @property
    def status_dict(self) -> dict:
        return {'done': self.should_stop, 'stop_reason': self.stop_reason, 'stop_epoch': self.stop_epoch}

    def process(self, val_loss: float, train_loss: float, val_score: float, epoch: int = None) -> None:

        if epoch is not None:
            self._epoch = epoch
        else:
            self._epoch += 1
        epoch = self._epoch

        self._is_new_nest = False
        self._stop_reason = ''

        for criterion in self.criteria:
            self.criteria[criterion].update(val_loss, train_loss, enable_count=epoch >= self.grace_period)
            if self.patience is not None and self.criteria[criterion].count > self.patience:
                self._stop_reason = criterion
                self._stop_epoch = epoch

        if self._best_val_score is None or val_score > self._best_val_score:
            self._is_new_nest = True
            self._stop_reason = ''
            self._stop_epoch = None
            self._best_val_score = val_score

        if self.epochs is not None and (epoch + 1) >= self.epochs:
            self._stop_reason = 'last_epoch'
            self._stop_epoch = epoch


class BatchManager:

    def __init__(self, items: int | Sequence,
                 batch_size: int = 64,
                 batches_in_epoch: int = None,
                 shuffle: bool = True):

        self._items = np.arange(items) if isinstance(items, int) else items
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch_items = self._items
        self.batches_in_epoch = len(self._items) // self.batch_size if batches_in_epoch is None else batches_in_epoch

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
        if stop > len(self.epoch_items):
            start = 0
            stop = self.batch_size
        return self.epoch_items[start: stop]


if __name__ == "__main__":
    pass
