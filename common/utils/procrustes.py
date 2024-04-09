import numpy as np

import common.utils.linalg as linalg
from common.utils.typings import *


class PlanarAlign:

    KINDS = ('affine', 'ortho', 'rigid', 'offset', 'none')

    def __init__(self, kind: str = 'affine'):
        assert kind in self.KINDS, f"Unknown kind: {kind}"
        self.kind = kind

    def calc_transform(self, X: NDArray, Y: NDArray) -> NpMatrix:
        if self.kind == 'affine':
            A = linalg.lsqr(Y, X)
        elif self.kind in ('ortho', 'rigid'):
            XX, locX, scaleX = self._normalize(X)
            YY, locY, scaleY = self._normalize(Y)
            U, _, Vt = np.linalg.svd(np.dot(XX.T, YY))
            R = np.dot(U, Vt)
            s = scaleX / scaleY
            t = -s * R @ locY + locX
            A = linalg.planar.to_homogeneous(s * R, t)
        elif self.kind == 'offset':
            t = np.mean(X, axis=0) - np.mean(Y, axis=0)
            A = linalg.planar.build(t=t)
        elif self.kind == 'none':
            A = linalg.planar.build(t=0)
        else:
            raise ValueError('Unknown procrustes kind')
        return A

    def __call__(self, X: NDArray, Y: NDArray) -> tuple[NDArray, NpMatrix]:
        """
        Args:
            X, Y - 2d point arrays, same size
        Returns:
            AY - A*Y
            A - homogeneous matrix [R|t] such that X ~ AY
        """
        A = self.calc_transform(X, Y)
        AY = linalg.planar.apply(A, Y)
        return AY, A

    def _normalize(self, xx):
        loc = np.mean(xx, axis=0)
        if self.kind in ('affine', 'ortho'):
            scale = np.linalg.norm(xx - loc, axis=1).sum()
        else:
            scale = 1.0
        return (xx - loc) / scale, loc, scale
#
#     def verify_matrix_kind(self, A: NpMatrix):
#
#         b, ang, t, is_reflective, ortho_score = linalg.planar.decompose(A)
#         geom_level = {'offset': 0, 'rigid': 1, 'ortho': 2, 'affine': 3}
#         level = geom_level[self.kind]
#
#         is_allowed_skew = level >= geom_level['affine']
#         is_allowed_scale = level >= geom_level['ortho']
#         is_allowed_rot = level >= geom_level['rigid']
#
#         rtol = 1e-4
#         is_scaled = abs(1 - b) > rtol
#         is_skewed = abs(1 - ortho_score) > rtol
#         is_rot = np.min([abs(ang), abs(360.0 - ang)]) > rtol
#
#         if not is_allowed_skew:
#             assert not is_skewed
#
#         if not is_allowed_scale:
#             assert not is_scaled
#
#         if not is_allowed_rot:
#             assert not is_rot
#
#
# def normalized_mahalanobis(X, Y) -> float:
#     """ average mahalanobis between matching X-Y points, relative to total covariance
#         empirically the average seems to be bounded around 2, and final result
#         is therefore divided by 2 to give normalization [0, 1]
#         TODO: Get theoretical understanding of what exactly is the bound and why
#     """
#     inv_cov = np.linalg.pinv(np.cov(np.r_[X, Y].T))
#     delta = X - Y
#     dists = np.sqrt(np.sum(np.dot(delta, inv_cov) * delta, axis=1))
#     return dists.mean() / 2
#
#
# def absolute_average(X, Y) -> float:
#     return float(np.abs(np.mean(X - Y)))
