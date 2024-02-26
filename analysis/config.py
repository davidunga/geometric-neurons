import json
from dataclasses import dataclass
import paths
import yaml
from munch import munchify, Munch
from common.utils import dictools, hashtools

_configs_grid = yaml.safe_load((paths.CONFIG_DIR / 'configs_grid.yml').open())


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

    def str(self) -> str:
        hash_size = 6
        return self.data.str(DataConfig.PAIRING) + " " + hashtools.calc_hash(self.__dict__, fmt='hex')[:hash_size]

    def __str__(self):
        return self.str()

    def __hash__(self) -> int:
        return hashtools.calc_hash(self.__dict__, fmt='int')

    @property
    def __dict__(self) -> dict:
        return {"data": self.data.__dict__,
                "model": self.model.__dict__,
                "training": self.training.__dict__}

    def jsons(self) -> str:
        return json.dumps(self.__dict__)

    @property
    def output_name(self) -> str:
        hash_size = 6
        return self.data.str(DataConfig.OUTPUT) + " " + hashtools.calc_hash(self.__dict__, fmt='hex')[:hash_size]


class DataConfig(Munch):

    class Level(str): pass
    BASE = Level("base")
    SEGMENTS = Level("segments")
    PAIRING = Level("pairing")
    OUTPUT = Level("output")

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
        s += " pair" + f"{self.pairing.variable} {self.pairing.metric}".replace(".", "").title().replace(" ", "")
        if level == DataConfig.PAIRING:
            return s
        assert level == DataConfig.OUTPUT
        addition = ""
        if self.pairing.sub_metric != "proc_dist":
            s += self.pairing.sub_metric.split("_")[0].title()
        if self.predictor.variable != "neural":
            addition += self.predictor.variable.replace(".", "").title().replace(" ", "")
        if self.predictor.shuffle:
            addition += "Shuff"
        if addition:
            s += f" pred{addition}"
        return s


class ModelConfig(Munch):
    pass


class TrainingConfig(Munch):
    pass
