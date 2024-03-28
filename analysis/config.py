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
    TRIALS = Level("trials")
    SEGMENTS = Level("segments")
    PAIRING = Level("pairing")
    OUTPUT = Level("output")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert DataConfig.TRIALS in self

    def str(self, level: Level = None):
        if level is None:
            level = DataConfig.PAIRING
        s = f"{self.trials.name} bin{round(1000 * self.trials.bin_sz):d} lag{round(1000 * self.trials.lag):d}"
        if level == DataConfig.TRIALS:
            return s
        s += f" dur{round(1000 * self.segments.dur)}"
        if level == DataConfig.SEGMENTS:
            return s
        s += f" {self.pairing.align_kind}-{self.pairing.variable}".replace(".", "")
        if level == DataConfig.PAIRING:
            return s
        assert level == DataConfig.OUTPUT
        s += "-" + self.pairing.dist_metric
        pred_token = ""
        if self.inputs.variable != "neural":
            pred_token += self.inputs.variable.replace(".", "").title().replace(" ", "")
        if self.pairing.shuffle:
            pred_token += "Shuff"
        if pred_token:
            s += f" pred{pred_token}"
        return s


class ModelConfig(Munch):
    pass


class TrainingConfig(Munch):
    pass
