import json
from dataclasses import dataclass
import yaml
from munch import munchify, Munch
from paths import ANALYSIS_DIR
from common.utils import dictools, hashtools
from copy import deepcopy
from common.utils import dictools

_configs_grid = yaml.safe_load((ANALYSIS_DIR / 'configs_grid.yml').open('r'))
_configs_best_grid = yaml.safe_load((ANALYSIS_DIR / 'configs_grid_best.yml').open('r'))
_config_mods_for_eval = yaml.safe_load((ANALYSIS_DIR / 'config_mods_for_eval.yml').open('r'))
_configs_chosen_per_dataset = {
    'TP_RS': yaml.safe_load((ANALYSIS_DIR / 'configs_chosen.yml').open('r')),
    'TP_RJ': yaml.safe_load((ANALYSIS_DIR / 'configs_chosen.yml').open('r')),
}
for dataset in _configs_chosen_per_dataset:
    _configs_chosen_per_dataset[dataset]['data']['trials']['name'] = dataset


@dataclass(init=False)
class Config:

    def __init__(self, cfg: dict):
        cfg = munchify(dictools.inherit_values(cfg))
        self.data = DataConfig(cfg.data)
        self.model = ModelConfig(cfg.model)
        self.training = TrainingConfig(cfg.training)

    def copy(self):
        return Config(deepcopy(self.__dict__))

    @classmethod
    def from_default(cls):
        return cls(next(dictools.dict_product_from_grid(_configs_grid)))

    @classmethod
    def from_chosen(cls, monkey: str):
        return cls(_configs_chosen_per_dataset['TP_' + monkey])

    @classmethod
    def yield_from_grid(cls, best: bool = False):
        for cfg_dict in dictools.dict_product_from_grid(_configs_best_grid if best else _configs_grid):
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

    @property
    def short_output_name(self) -> str:
        monkey = self.data.trials.name.split('_')[1]
        align = self.data.pairing.align_kind[:3]
        dist = self.data.pairing.dist_metric[:3]
        hash_ = hashtools.calc_hash(self.__dict__, fmt='hex')[:6]
        return f"{monkey} {align} {dist} {hash_}"

    def get_as_eval_config(self):
        return _get_modified_config(self, _config_mods_for_eval)


class DataConfig(Munch):

    class Level(str): pass
    TRIALS = Level("trials")
    SEGMENTS = Level("segments")
    PAIRING = Level("pairing")
    OUTPUT = Level("output")
    LEVELS_ORDER = [TRIALS, SEGMENTS, PAIRING, OUTPUT]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert DataConfig.TRIALS in self

    @classmethod
    def from_dict(cls, cfg):
        return cls(munchify(dictools.inherit_values(cfg)))

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


def _get_modified_config(cfg, mod_dict: dict):
    """ get a copy of config, recursively modified by mod_dict """
    modified_dict = dictools.deep_merge(deepcopy(cfg.__dict__), mod_dict)
    return Config(modified_dict)
