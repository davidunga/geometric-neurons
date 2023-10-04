import torch
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from abc import ABC, abstractmethod, abstractproperty


def embdded_dist_fnc(x1, x2):
    return torch.sum((x1 - x2) ** 2, dim=1)


class Embedder(ABC, torch.nn.Module):

    def __init__(self, input_size, output_size, *args, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    @abstractmethod
    def forward(self, x):
        pass


class LinearEmbedder(Embedder):

    def __init__(self, input_size, output_size, dropout):
        super().__init__(input_size, output_size)
        torch.manual_seed(0)
        self.embedder = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(self.input_size, self.output_size))

    def forward(self, x):
        return self.embedder(x)


class SamenessClassifier(BaseEstimator):

    def __init__(self, embedder: Embedder):
        self.classifier = CalibratedClassifierCV(embedder, cv='prefit')

    @staticmethod
    def _prep_x(x1, x2):
        x1_x2 = torch.concat([x1, x2], dim=1)
        return x1_x2

    def fit(self, x1, x2, is_same):
        self.classifier.fit(self._prep_x(x1, x2), is_same)

    def predict(self, x1, x2):
        return self.classifier.predict(self._prep_x(x1, x2))

    def predict_proba(self, x1, x2):
        return self.classifier.predict_proba(self._prep_x(x1, x2))
