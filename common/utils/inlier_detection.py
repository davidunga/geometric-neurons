import numpy as np


class InlierDetector:

    def __init__(self, method: str, thresh: float = None):
        """
        method: 'iqr' = interquartile range
                'sigmas' = robust zscore
        thresh: if method = 'iqr', thresh is in IQR units. default = 1.5
                if method = 'sigmas', thresh is in sigma units. default = 3.
        """
        assert method in ('iqr', 'sigmas')
        self.method = method
        self.thresh = thresh
        self._lb = None
        self._ub = None

    def fit(self, x: np.ndarray, y=None):
        if self.method == 'iqr':
            thresh = 1.5 if self.thresh is None else self.thresh
            q1, q3 = np.percentile(x, [25, 75])
            iqr = q3 - q1
            self._lb = q1 - thresh * iqr
            self._ub = q3 + thresh * iqr
        elif self.method == 'sigmas':
            thresh = 3 if self.thresh is None else self.thresh
            med = np.median(x)
            mad = 1.4826 * np.median(np.abs(x - med))
            self._lb = -thresh * mad + med
            self._ub = thresh * mad + med
        else:
            assert ValueError("Unknown method")
        return self

    def predict(self, x: np.ndarray, y=None) -> np.ndarray[bool]:
        assert self._lb is not None and self._ub is not None, "Detector is not fitted"
        return (x >= self._lb) & (x <= self._ub)

    def fit_predict(self, x: np.ndarray, y=None) -> np.ndarray[bool]:
        return self.fit(x, y).predict(x, y)
