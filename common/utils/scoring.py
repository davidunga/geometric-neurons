import numpy as np
from typing import Callable


class BootstrapEvaluator:
    """ evaluate metrics and compare to bootstrapped baseline distribution """

    def __init__(self, n_shuffs: int = 1000, seed: int = 1, metrics=None):
        """
        Args:
            n_shuffs: number of bootstrap shuffles
            seed: random seed for bootstrap
            metrics: either a dict of metrics, or a list of metric names
                None = all relevant metrics (default)
        """
        self.n_shuffs = n_shuffs
        self.seed = seed
        self.metrics_dict = metrics if isinstance(metrics, dict) else get_named_metrics(metrics)

    def evaluate(self, ytrue: np.ndarray, yhat: np.ndarray):
        ret = {}
        for metric, func in self.metrics_dict.items():
            rng = np.random.default_rng(self.seed)
            value = func(ytrue, yhat)
            shuffs = np.fromiter((func(ytrue, rng.permutation(yhat))
                                  for _ in range(self.n_shuffs)), float)
            zscore = (value - np.mean(shuffs)) / np.std(shuffs)
            ret[metric] = {'value': value, 'shuffs': shuffs, 'zscore': zscore}
        return ret


def get_named_metrics(names: list[str] = None):
    _metrics = {
        'mae': lambda a, b: np.mean(np.abs(a - b)),
        'mse': lambda a, b: np.mean((a - b) ** 2),
    }
    if not names:
        return _metrics
    else:
        return {name: _metrics[name] for name in names}
