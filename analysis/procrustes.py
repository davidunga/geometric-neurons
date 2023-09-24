import numpy as np
import common.linalg as linalg
from common.type_utils import *


class Procrustes:

    def __init__(self, kind: str = 'affine'):
        assert kind in ('affine', 'ortho')
        self.kind = kind

    def __call__(self, X: NDArray, Y: NDArray) -> tuple[float, NpMatrix]:
        """
        Args:
            X, Y - 2d point arrays, same size
        Returns:
            d - the procrustes distance
            A - homogeneous matrix [R|t] such that X ~ AY.
                R is either affine (general) or orthogonal (uniform scaling), according to 'kind'
        """

        if self.kind == 'affine':
            A = linalg.lsqr(Y, X)

        elif self.kind == 'ortho':
            def _normalize(xx):
                loc = np.mean(xx, axis=0)
                scale = np.linalg.norm(xx - loc)
                return (xx - loc) / scale, loc, scale
            XX, locX, scaleX = _normalize(X)
            YY, locY, scaleY = _normalize(Y)
            U, _, Vt = np.linalg.svd(np.dot(XX.T, YY))
            R = np.dot(U, Vt)
            s = scaleX / scaleY
            t = -s * R @ locY + locX
            A = linalg.planar.to_homogeneous(s * R, t)
        else:
            raise ValueError('Unknown procrustes kind')

        Z = linalg.planar.apply(A, Y)
        d = Procrustes.error(X, Z)

        return d, A

    @staticmethod
    def error(X, Y) -> float:
        dist = np.sum((X - Y) ** 2)
        scale = np.sum((X - X.mean(axis=0)) ** 2)
        return np.sqrt(dist / scale)
