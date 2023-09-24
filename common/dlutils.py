"""
Deep-learning related utils
"""

import torch


def load_checkpoint(fname):
    items = torch.load(fname)
    model = items['model']
    model.load_state_dict(items['state_dict'])
    optimizer = items['optimizer']
    if optimizer:
        optimizer.load_state_dict(items['optimizer_state'])
    meta = items['meta']
    return model, optimizer, meta


def save_checkpoint(fname, model, optimizer=None, meta=None):
    optimizer_state = optimizer.state_dict() if optimizer else None
    torch.save({'model': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer,
                'optimizer_state': optimizer_state,
                'meta': meta}, fname)


def get_meta(fname):
    _, _, meta = load_checkpoint(fname)
    return meta


class EarlyStopping:

    """
    Manage stopping criteria
    """

    def __init__(self, patience=5, min_improve=.001, max_overfit=.2):
        """
        :param patience: number of steps to wait before stopping, once a threshold values is met
        :param max_overfit: maximum acceptable overfit
        :param min_improve: minimum acceptable improvement

        Definitions:
        - improvement - the relative difference between the current and last loss:
            (last_val_loss - current_val_loss) / last_val_loss

        - overfit - the relative difference between the validation and training losses:
            (val_loss - train_loss) / train_loss

        """
        self.patience = patience
        self.thresholds = {'improve': min_improve, 'overfit': max_overfit}
        self.counters = {'improve': 0, 'overfit': 0}
        self.metrics = {'improve': 0, 'overfit': 0}
        self._last_val_loss = None
        self._stop_reason = ''

    def should_stop(self):
        return len(self._stop_reason) > 0

    def stop_reason(self):
        return self._stop_reason

    def report_scores(self):
        s = ""
        for criterion in self.metrics:
            s += "{:s}:{:+2.3f} ".format(criterion, self.metrics[criterion])
        return s[:-1]

    def process(self, val_loss, train_loss):

        self.metrics['overfit'] = val_loss / max(train_loss, 1e-9) - 1

        if self._last_val_loss is None:
            self.metrics['improve'] = 1
        else:
            self.metrics['improve'] = 1 - (val_loss / self._last_val_loss)

        self._last_val_loss = val_loss

        self._stop_reason = ''

        if self.patience is not None:

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
