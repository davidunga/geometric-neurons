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

_default_config_dict = yaml.safe_load((paths.PROJECT_CONFIG_DIR / 'default_config.yml').open())


@dataclass(init=False)
class Config:

    def __init__(self, cfg: dict):
        cfg = munchify(cfg)
        self.data = DataConfig(cfg.data)
        self.model_selection = ModelSelectionConfig(cfg.model_selection)
        self.embedding = EmbeddingConfig(cfg.embedding)

    @classmethod
    def from_default(cls):
        return cls(deepcopy(_default_config_dict))


class DataConfig(Munch):

    class Level(str): pass
    BASE = Level("base")
    SEGMENTS = Level("segments")
    PAIRING = Level("pairing")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert DataConfig.BASE in self

    def str(self, level: Level):
        s = f"{self.base.name} bin{round(1000 * self.base.bin_sz):d} lag{round(1000 * self.base.lag):d}"
        if level == DataConfig.BASE:
            return s
        s += f" dur{round(1000 * self.segments.dur)}"
        if level == DataConfig.SEGMENTS:
            return s
        assert level == DataConfig.PAIRING
        s += f" proc{self.pairing.proc_kind.capitalize()}"
        return s


class ModelSelectionConfig(Munch):
    pass


class EmbeddingConfig(Munch):
    pass



