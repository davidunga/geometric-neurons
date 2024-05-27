import numpy as np
from sklearn.covariance import MinCovDet


def gaussian_fit(pts, support_fraction: float = .9):
    if support_fraction == 1:
        cov = np.cov(pts[:, 0], pts[:, 1])
        inlier_mask = np.ones(len(pts), bool)
    else:
        robust_cov = MinCovDet(support_fraction=support_fraction, random_state=1).fit(pts)
        cov = robust_cov.covariance_
        inlier_mask = robust_cov.support_
    mu = pts[inlier_mask].mean(axis=0)
    return (mu, cov), inlier_mask


def gaussian2d_to_ellipse(gauss, n_std: float = 2):
    validate_gaussian(gauss, ndim=2)
    center, cov = gauss
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    return center, (width, height), angle


def mahalanobis_distance(mu1, mu2, cov):
    """
    Compute the Mahalanobis distance between two means given a shared covariance matrix.

    Parameters:
    - mu1: Mean vector of the first Gaussian distribution.
    - mu2: Mean vector of the second Gaussian distribution.
    - cov: Covariance matrix (assumed to be the same for both distributions).

    Returns:
    - Mahalanobis distance.
    """
    delta_mu = mu1 - mu2
    inv_cov = np.linalg.inv(cov)
    return np.sqrt(np.dot(np.dot(delta_mu.T, inv_cov), delta_mu))


def bhattacharyya_distance(gauss1, gauss2):
    """
    Compute the Bhattacharyya distance between two 2D Gaussian distributions.

    Parameters:
    - mu1: Mean vector of the first Gaussian distribution.
    - mu2: Mean vector of the second Gaussian distribution.
    - cov1: Covariance matrix of the first Gaussian distribution.
    - cov2: Covariance matrix of the second Gaussian distribution.

    Returns:
    - Bhattacharyya distance.
    """

    validate_gaussian(gauss1)
    validate_gaussian(gauss2)

    mu1, cov1 = gauss1
    mu2, cov2 = gauss2

    mean_diff = mu1 - mu2
    cov_avg = (cov1 + cov2) / 2

    # First term
    inv_cov_avg = np.linalg.inv(cov_avg)
    term1 = 0.125 * np.dot(np.dot(mean_diff.T, inv_cov_avg), mean_diff)

    # Second term
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)
    det_cov_avg = np.linalg.det(cov_avg)
    term2 = 0.5 * np.log(det_cov_avg / np.sqrt(det_cov1 * det_cov2))

    return term1 + term2


def validate_gaussian(gauss, ndim: int = None):
    mu, cov = gauss
    assert mu.ndim == 1
    assert cov.ndim == 2
    assert cov.shape[0] == cov.shape[1] == len(mu)
    if ndim:
        assert len(mu) == ndim
