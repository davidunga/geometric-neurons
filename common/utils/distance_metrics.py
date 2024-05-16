import numpy as np
from typing import Callable


def normalized_mahalanobis(X, Y) -> float:
    """ average mahalanobis between matching X-Y points, relative to total covariance
        empirically the average seems to be bounded around 2, and final result
        is therefore divided by 2 to give normalization [0, 1]
        TODO: Get theoretical understanding of what exactly is the bound and why
    """
    inv_cov = np.linalg.pinv(np.cov(np.r_[X, Y].T))
    delta = X - Y
    dists = np.sqrt(np.sum(np.dot(delta, inv_cov) * delta, axis=1))
    return dists.mean() / 2


def normalized_frobenius(X, Y):
    scale = np.linalg.norm(X, 'fro') + np.linalg.norm(Y, 'fro')
    dist = 2 * np.linalg.norm(X - Y, 'fro') / scale
    return dist


def absolute_average(X, Y) -> float:
    return float(np.abs(np.mean(X - Y)))


_func_names = {
    normalized_mahalanobis: ['normalized_mahalanobis', 'nmahal'],
    normalized_frobenius: ['normalized_frobenius', 'nfrob'],
    absolute_average: ['absolute_average', 'absavg']
}


def get_metric_func(name: str) -> Callable:
    for func, names in _func_names.items():
        if name in names:
            return func
    raise ValueError("Could not find function for metric name")
