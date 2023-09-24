import numpy as np
from src.common.linalg import tform, linlsqr
from scipy.linalg import orthogonal_procrustes
from src.common.type_utils import *


class Procrustes:

    def __init__(self, kind: str = 'affine'):
        self.kind = kind

    def __call__(self, X: NDArray, Y: NDArray) -> tuple[float, NpMatrix]:

        if self.kind == 'affine':
            A = linlsqr(Y, X)
        elif self.kind == 'ortho':
            t = X.mean(axis=0) - Y.mean(axis=0)
            R, s = orthogonal_procrustes(X, Y + t)
            A = np.eye(3)
            A[:2, :2] = s * R
            A[:, -1] = t
        else:
            raise ValueError('Unknown procrustes kind')

        Z = tform.apply(A, Y)
        d = Procrustes.error(X, Z)

        return d, A

    @staticmethod
    def error(X, Y) -> float:
        dist = np.sum((X - Y) ** 2)
        scale = np.sum((X - X.mean(axis=0)) ** 2)
        return np.sqrt(dist / scale)
