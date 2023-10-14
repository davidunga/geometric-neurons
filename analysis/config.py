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
from common.utils import dictools

_configs_grid = yaml.safe_load((paths.CONFIG_DIR / 'config.yml').open())


@dataclass(init=False)
class Config:

    def __init__(self, cfg: dict):
        cfg = munchify(cfg)
        self.data = DataConfig(cfg.data)
        self.model = ModelConfig(cfg.model)
        self.training = TrainingConfig(cfg.training)

    @classmethod
    def from_default(cls):
        return cls(next(dictools.dict_product_from_grid(_configs_grid)))

    @classmethod
    def yield_from_grid(cls):
        for cfg_dict in dictools.dict_product_from_grid(_configs_grid):
            yield cls(cfg_dict)

    def str(self):
        hash_str_sz = 6
        hash_str = sha1(str(self.__dict__()).encode('utf-8')).hexdigest()[:hash_str_sz]
        return self.data.str(DataConfig.PAIRING) + " " + hash_str

    def __dict__(self) -> dict:
        return {"data": self.data.__dict__,
                "model": self.model.__dict__,
                "training": self.training.__dict__}

    def jsons(self) -> str:
        return json.dumps(self.__dict__())


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
        s += f" proc{self.pairing.proc_kind.capitalize()}"
        return s


class ModelConfig(Munch):
    pass


class TrainingConfig(Munch):
    pass
