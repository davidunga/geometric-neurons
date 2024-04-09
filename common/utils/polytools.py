import numpy as np

from common.utils.typings import *


def edge_lengths(X: NpPoints):
    return np.linalg.norm(np.diff(X, axis=0), axis=1)


def cumm_arclen(X: NpPoints) -> NpVec[float]:
    return np.r_[0, np.cumsum(edge_lengths(X))]


def total_arclen(X: NpPoints) -> float:
    return cumm_arclen(X)[-1]



def _test():
    from common.utils.testing_tools import shapesbank
    a = 1
    b = 1.5
    X = shapesbank.ellipse(a=a, b=b, n=50000)
    approx_true_arclen = (a + b) * np.pi
    print("approx true arclen =", approx_true_arclen)
    print("est total arclen=", total_arclen(X))

if __name__ == "__main__":
    _test()