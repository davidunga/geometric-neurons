from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
import numpy as np
from common.utils import stats


class CenterCalibRegressor(BaseEstimator, RegressorMixin):
    """ Calibrates regressor to align with true centers in designated bins.
        Default behavior applies a scale and offset to align prediction bin averages with true bin centers.
    """

    def __init__(self, base_estimator=None, bins: stats.BinSpec = stats.BinSpec(),
                 stat: str = 'avg', calib_model=None):
        self.base_estimator = LinearRegression() if base_estimator is None else base_estimator
        self.bins = bins
        self.stat = stat
        self.calib_model = LinearRegression() if calib_model is None else calib_model

    def fit(self, X, y, *args, **kwargs):

        self.base_estimator = clone(self.base_estimator)
        self.base_estimator.fit(X, y, *args, **kwargs)

        yhat = self.base_estimator.predict(X)

        bin_stats = stats.calc_binned_stats(x=y, y=yhat, bins=self.bins, stats=['n', self.stat])
        pred_locations = bin_stats[self.stat][bin_stats['n'] > 0]
        true_centers = bin_stats['x'][bin_stats['n'] > 0]

        self.calib_model = clone(self.calib_model)
        self.calib_model.fit(pred_locations.reshape(-1, 1), true_centers)

        return self

    def predict(self, X):
        preds = self.base_estimator.predict(X)
        preds = self.calib_model.predict(preds.reshape(-1, 1)).flatten()
        return preds


def calc_balancing_weights(x, bins: stats.BinSpec) -> np.ndarray[float]:
    """ assign weights inversely proportional to bin counts """
    inds = stats.safe_digitize(x, bins)[0]
    weights = np.zeros_like(x, float)
    for ind in range(bins.n):
        bin_inds = inds == ind
        weights[bin_inds] = 1 / np.sum(bin_inds)
    return weights
