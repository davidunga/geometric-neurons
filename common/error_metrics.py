import numpy as np


def error_abs_rel(ytrue, ypred):
    return np.abs(ypred - ytrue) / np.maximum(np.abs(ytrue), 1e-6)


def error_abs(ytrue, ypred):
    return np.abs(ypred - ytrue)


def error_with_shuffles(ytrue, ypred, relative=True, num_shuffs=1000):
    _err_fn = error_abs_rel if relative else error_abs
    err = _err_fn(ytrue, ypred).mean()
    err_shuffles = np.zeros(num_shuffs)
    for i in range(num_shuffs):
        err_shuffles[i] = _err_fn(np.random.permutation(ytrue), ypred).mean()
    return err, err_shuffles
