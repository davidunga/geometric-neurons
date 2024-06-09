import json
import os.path
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from common.utils import hashtools
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
import paths
from collections import Counter
from common.utils import stats
from common.utils.procrustes import PlanarAlign
from common.utils.distance_metrics import normalized_mahalanobis
from motorneural.data import Segment
from data_manager import DataMgr
import cv_results_mgr
from neural_population import NeuralPopulation, NEURAL_POP
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from common.utils import plotting
from common.utils import dlutils
from common.utils.randtool import Rnd
from common.utils import polytools
import seaborn as sns
import embedtools
from common.utils import gaussians
from common.utils import dictools
from common.utils import hashtools
from copy import deepcopy
from common.utils import linalg
from analysis.config import DataConfig
from common.utils.conics.conic_fitting import fit_conic_ransac, Conic
from dataclasses import dataclass
from common.utils.conics import ConicEllipse, ConicParabola


@dataclass
class ShapeSpec:
    kind: str
    e: float
    bias: float

    @classmethod
    def from_conic(cls, conic: Conic):
        return cls(kind=conic.kind, e=conic.eccentricity(), bias=conic.bounds_bias())

    @property
    def name(self) -> str:
        if not self.is_valid:
            return 'invalid'
        else:
            return f'{self.kind}{self.bias:+1.2f} {self.e:1.2f}'

    def __str__(self):
        return self.name

    def to_dict(self):
        return {'kind': self.kind, 'e': self.e, 'bias': self.bias}

    @property
    def is_valid(self):
        return self.kind is not None

    def dist2(self, other):
        e_scale = .05
        bias_scale = .2
        if other.kind != self.kind:
            return np.inf
        e_diff = (self.e - other.e) / e_scale
        bias_diff = (self.bias - other.bias) / bias_scale
        return e_diff ** 2 + bias_diff ** 2



