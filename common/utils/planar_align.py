import numpy as np
import common.utils.linalg as linalg
from common.utils.typings import *


class PlanarAligner:

    KINDS = ('affine', 'ortho', 'rigid', 'offset', 'none')

    def __init__(self, kind: str = 'affine'):
        assert kind in self.KINDS, f"Unknown kind: {kind}"
        self.kind = kind

    def __call__(self, X: NDArray, Y: NDArray) -> tuple[NDArray, NpMatrix]:
        """
        Align Y to X, using transform type [self.kind]
        Args:
            X - 2d point array
            Y - 2d point array, same length as X
        Returns:
            AY - A*Y
            A - homogeneous matrix [R|t] such that X ~ AY
        """
        A = self.calc_transform(X, Y)
        AY = linalg.planar.apply(A, Y)
        return AY, A

    def calc_transform(self, X: NDArray, Y: NDArray) -> NpMatrix:
        """
        Find transformation from Y to X, using transform type [self.kind]
        Args:
            X - 2d point array
            Y - 2d point array, same length as X
        Returns:
            A - homogeneous matrix [R|t] such that X ~ AY
        """

        if self.kind == 'affine':
            A = linalg.lsqr(Y, X)
        elif self.kind in ('ortho', 'rigid'):
            XX, locX, scaleX = _normalize(X, rigid=self.kind == 'rigid')
            YY, locY, scaleY = _normalize(Y, rigid=self.kind == 'rigid')
            U, _, Vt = np.linalg.svd(np.dot(XX.T, YY))
            R = np.dot(U, Vt)
            s = scaleX / scaleY
            t = -s * R @ locY + locX
            A = linalg.planar.to_homogeneous(s * R, t)
        elif self.kind == 'offset':
            t = np.mean(X, axis=0) - np.mean(Y, axis=0)
            A = linalg.planar.build(t=t)
        elif self.kind == 'none':
            A = linalg.planar.build()
        else:
            raise ValueError("Unknown alignment kind")
        return A


def _normalize(xx: NDArray, rigid: bool):
    loc = np.mean(xx, axis=0)
    scale = 1.0 if rigid else np.linalg.norm(xx - loc, axis=1).sum()
    return (xx - loc) / scale, loc, scale
