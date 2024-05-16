import numpy as np
from typing import Callable
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, explained_variance_score,
    median_absolute_error, r2_score, balanced_accuracy_score, f1_score, roc_auc_score)

_metrics = {
    'reg': {
        'mae': mean_absolute_error,
        'mse': mean_squared_error,
        'medae': median_absolute_error,
        'mape': mean_absolute_percentage_error,
        'r2': r2_score,
        'evs': explained_variance_score,
    },
    'cls': {
        'acc': balanced_accuracy_score,
        'f1': f1_score,
        'auc': roc_auc_score
    }
}


class BootstrapEvaluator:
    """ evaluate metrics and compare to bootstrapped baseline distribution """

    def __init__(self, n_shuffs: int = 1000, seed: int = 1, metrics='auto'):
        """
        Args:
            n_shuffs: number of bootstrap shuffles
            seed: random seed for bootstrap
            metrics: either a dict of metrics, a list of metric names,
                or 'reg'/'cls' for all regression/classification metrics,
                or 'auto' to automatically decide between 'reg' and 'cls'.
        """
        self.n_shuffs = n_shuffs
        self.seed = seed
        self.metrics = metrics

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
        metrics = self.metrics
        if metrics == 'auto':
            metrics = 'cls' if np.all(np.round(y_true) == y_true) else 'reg'
        metrics_dict = metrics if isinstance(metrics, dict) else get_named_metrics(metrics)

        ret = {}
        for metric, func in metrics_dict.items():
            rng = np.random.default_rng(self.seed)
            value = func(y_true=y_true, y_pred=y_pred, **kwargs)
            shuffs = np.fromiter((func(y_true, rng.permutation(y_pred))
                                  for _ in range(self.n_shuffs)), float)
            zscore = (value - np.mean(shuffs)) / np.std(shuffs)
            ret[metric] = {'value': value, 'zscore': zscore, 'shuffs': shuffs}
        return ret


def get_named_metrics(names: list[str] | str) -> dict[str, Callable]:
    if isinstance(names, str):
        names = _metrics[names]
    all_metrics = {**_metrics['reg'], **_metrics['cls']}
    return {name: all_metrics[name] for name in names}
