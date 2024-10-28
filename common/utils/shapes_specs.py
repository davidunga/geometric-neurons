import numpy as np
from common.utils.conics.conic_fitting import fit_conic_ransac, Conic
from dataclasses import dataclass


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



