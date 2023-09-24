import numpy as np


class tform:

    @staticmethod
    def _check_mtx(A):
        assert A.shape in ((2, 2), (3, 3))
        if A.shape == (3, 3):
            assert np.abs(A[2] - np.array([0, 0, 1])).max() < 1e-8

    @staticmethod
    def apply(A, X):
        tform._check_mtx(A)
        return (A @ _pad_to_3d(X).T).T[:, :2]

    @staticmethod
    def apply_inv(A, X):
        return tform.apply(np.linalg.inv(A), X)

    @staticmethod
    def decompose(A, scale_metric: str = 'det'):
        """
        decompose affine matrix to: scale, rotation angle, translation, and is_reflected
        :param A: 2x2 or 3x3 matrix
        :param scale_metric: how to measure the scale
        """

        tform._check_mtx(A)

        t = A[:2, -1] if A.shape == (3, 3) else np.zeros(2, float)

        U, s, Vt = np.linalg.svd(A[:2, :2])

        if scale_metric == 'avg':
            b = s.mean()
        elif scale_metric == 'norm':
            b = np.linalg.norm(s)
        elif scale_metric == 'det':
            b = np.abs(np.linalg.det(A[:2, :2]))
        else:
            raise ValueError('Unknown scale metric')

        R = U.T @ Vt.T

        ang1 = np.degrees(np.arctan2(-R[0, 1], R[0, 0])) % 360
        ang1 = min(ang1, 360 - ang1)

        ang2 = np.degrees(np.arctan2(R[1, 0], R[1, 1])) % 360
        ang2 = min(ang2, 360 - ang2)

        if ang2 < ang1:
            ang1, ang2 = (ang2, ang1)

        is_reflected = np.linalg.det(R) < 0
        if is_reflected:
            ang2 = 180 - ang2

        assert np.abs(ang1 - ang2) < .5
        ang = (ang1 + ang2) / 2

        return b, ang, t, is_reflected


def linlsqr(x, b):
    """ matrix A s.t. Ax ~ b """
    return _pad_to_3d(x).T @ np.linalg.pinv(_pad_to_3d(b).T)


def _pad_to_3d(X):
    if X.shape[1] == 2:
        return np.c_[X, np.ones(len(X))]
    elif X.shape[1] == 3:
        return X
    else:
        ValueError()
