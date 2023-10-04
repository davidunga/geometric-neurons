""" Deep-learning utilities """

import torch
from typing import Hashable


def get_optimizer(model_params, opt_params: str | dict):
    if isinstance(opt_params, str): opt_params = {'kind': opt_params}
    return getattr(torch.optim, opt_params['kind'])(
        model_params, **{k: v for k, v in opt_params.items() if k != 'kind'})


def load_checkpoint(fname: str):
    items = torch.load(fname)
    model = items['model']
    model.load_state_dict(items['state_dict'])
    optimizer = items['optimizer']
    if optimizer:
        optimizer.load_state_dict(items['optimizer_state'])
    meta = items['meta']
    return model, optimizer, meta


def save_checkpoint(fname: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    meta: Hashable = None):
    torch.save({'model': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer,
                'optimizer_state': optimizer.state_dict() if optimizer else None,
                'meta': meta}, fname)


def get_meta(fname: str):
    return torch.load(fname)['meta']


class EarlyStopping:

    def __init__(self, patience: int | None = 5, min_improve: float = .001, max_overfit: float = .2):
        """
        Args:
            patience: number of steps to wait before stopping, once a threshold values is met. None = never.
            min_improve: minimum acceptable improvement
                improvement = (last_val_loss - current_val_loss) / last_val_loss
            max_overfit: maximum acceptable overfit
                overfit = (val_loss - train_loss) / train_loss
        """

        self.patience = patience
        self.thresholds = {'improve': min_improve, 'overfit': max_overfit}
        self.counters = {'improve': .0, 'overfit': .0}
        self.metrics = {'improve': .0, 'overfit': .0}
        self._last_val_loss = None
        self._stop_reason = ''

    def should_stop(self) -> bool:
        return len(self._stop_reason) > 0

    def stop_reason(self) -> str:
        return self._stop_reason

    def report_scores(self) -> str:
        s = ""
        for criterion in self.metrics:
            s += "{:s}:{:+2.3f} ".format(criterion, self.metrics[criterion])
        return s[:-1]

    def process(self, val_loss: float, train_loss: float) -> bool:

        self.metrics['overfit'] = val_loss / max(train_loss, 1e-9) - 1

        if self._last_val_loss is None:
            self.metrics['improve'] = 1
        else:
            self.metrics['improve'] = 1 - (val_loss / self._last_val_loss)

        self._last_val_loss = val_loss
        self._stop_reason = ''

        if self.patience is None:
            return self.should_stop()

        if self.metrics['improve'] < self.thresholds['improve']:
            self.counters['improve'] += 1
            if self.counters['improve'] >= self.patience:
                self._stop_reason = 'converged'
        else:
            self.counters['improve'] = 0

        if self.metrics['overfit'] > self.thresholds['overfit']:
            self.counters['overfit'] += 1
            if self.counters['overfit'] >= self.patience:
                self._stop_reason = 'overfit'
        else:
            self.counters['overfit'] = 0

        return self.should_stop()


if __name__ == "__main__":
    pass
