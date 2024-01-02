import json
from copy import deepcopy
from glob import glob
import numpy as np
import os
from collections import namedtuple
from pathlib import Path
from dataclasses import dataclass
import paths
import yaml
from munch import munchify, Munch
from hashlib import sha1
from common.utils import dictools, hashtools

_configs_grid = yaml.safe_load((paths.CONFIG_DIR / 'config.yml').open())

# fields that were added to configs, and their default values
# used for backward compatibility in terms of hash and data structure
_ADDED_FIELDS_AND_DEFAULTS = {"data.sameness.normalize_neural": False, "data.base.kin_as_neural": False}


@dataclass(init=False)
class Config:

    def __init__(self, cfg: dict):

        cfg = dictools.flatten_dict(cfg)
        for added_key, default_value in _ADDED_FIELDS_AND_DEFAULTS.items():
            if added_key not in cfg:
                cfg[added_key] = default_value
        cfg = dictools.unflatten_dict(cfg)

        cfg = munchify(cfg)
        self.data = DataConfig(cfg.data)
        self.model = ModelConfig(cfg.model)
        self.training = TrainingConfig(cfg.training)

    def _dict_for_hashing(self):
        d = dictools.flatten_dict(self.__dict__)
        for added_key, default_value in _ADDED_FIELDS_AND_DEFAULTS.items():
            if d[added_key] == default_value:
                del d[added_key]
        d = dictools.unflatten_dict(d)
        return d

    @classmethod
    def from_default(cls):
        return cls(next(dictools.dict_product_from_grid(_configs_grid)))

    @classmethod
    def yield_from_grid(cls):
        for cfg_dict in dictools.dict_product_from_grid(_configs_grid):
            yield cls(cfg_dict)

    def str(self) -> str:
        hash_size = 6
        return self.data.str(DataConfig.PAIRING) + " " + hashtools.calc_hash(self._dict_for_hashing(), fmt='hex')[:hash_size]

    def __str__(self):
        return self.str()

    def __hash__(self) -> int:
        return hashtools.calc_hash(self._dict_for_hashing(), fmt='int')

    @property
    def __dict__(self) -> dict:
        return {"data": self.data.__dict__,
                "model": self.model.__dict__,
                "training": self.training.__dict__}

    def jsons(self) -> str:
        return json.dumps(self.__dict__)


class DataConfig(Munch):

    class Level(str): pass
    BASE = Level("base")
    SEGMENTS = Level("segments")
    PAIRING = Level("pairing")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert DataConfig.BASE in self

    def str(self, level: Level = None):
        if level is None:
            level = DataConfig.PAIRING
        s = f"{self.base.name} bin{round(1000 * self.base.bin_sz):d} lag{round(1000 * self.base.lag):d}"
        if level == DataConfig.BASE:
            return s
        s += f" dur{round(1000 * self.segments.dur)}"
        if level == DataConfig.SEGMENTS:
            return s
        assert level == DataConfig.PAIRING
        s += " pair" + f"{self.pairing.variable} {self.pairing.metric}".replace(".", "").title().replace(" ", "")
        return s


class ModelConfig(Munch):
    pass


class TrainingConfig(Munch):
    pass
