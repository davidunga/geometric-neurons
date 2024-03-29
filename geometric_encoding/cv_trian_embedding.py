import json
from itertools import product
from datetime import datetime
import numpy as np
import pandas as pd
import torch.nn

from analysis.config import Config
from geometric_encoding.embedding import LinearEmbedder
from analysis.data_manager import DataMgr
from common.pairwise.triplet_train import triplet_train
from common.pairwise.sameness import SamenessData
import paths
from glob import glob
from common.utils.typings import *
from common.utils.dlutils import checkpoint
from common.utils import dlutils
from common.utils import ostools
from common.utils import dictools
import os
from time import time
from cv_models_manager import CvModelsManager
from analysis.config import DataConfig, Config, TrainingConfig
from common.utils.devtools import verbolize


class TrainingMgr:

    def __init__(self,
                 cfg: Config | dict,
                 fold: int,
                 dbg_run: bool = False,
                 early_stop_epoch: int = None,
                 exists_handling: Literal["warm_start", "overwrite", "skip", "error"] = "skip",
                 **kwargs):

        self.cfg = cfg if isinstance(cfg, Config) else Config(cfg)
        self.fold = fold
        self.dbg_run = dbg_run
        self.early_stop_epoch = early_stop_epoch
        self.exists_handling = exists_handling
        self._kwargs = kwargs

    def as_dict(self) -> dict:
        return dict(cfg=self.cfg.__dict__, fold=self.fold, dbg_run=self.dbg_run)

    @verbolize()
    def _split_sameness_by_fold(self, sameness_data: SamenessData):
        fold = self.fold
        if fold == -1:
            return sameness_data, None
        assert fold in range(self.cfg.training.cv.folds)
        rng = np.random.default_rng(self.cfg.training.cv.rand_seed)
        val_start = int(sameness_data.n * fold / self.cfg.training.cv.folds)
        val_stop = int(round(sameness_data.n * (fold + 1) / self.cfg.training.cv.folds))
        val_items = rng.permutation(sameness_data.n)[val_start: val_stop]
        mutex_groups = sameness_data.mutex_indexes(val_items)
        sameness_data_2 = sameness_data.modcopy(index_mask=mutex_groups == 1)
        sameness_data = sameness_data.modcopy(index_mask=mutex_groups == 2)
        return sameness_data, sameness_data_2

    def load_split_sameness(self) -> tuple[SamenessData, SamenessData]:
        sameness_data, _, _ = DataMgr(self.cfg.data).load_sameness()
        train_sameness, val_sameness = self._split_sameness_by_fold(sameness_data)
        return train_sameness, val_sameness

    def init_model(self, input_size: int) -> torch.nn.Module:
        model = LinearEmbedder(input_size=input_size, **self.cfg.model)
        return model

    @property
    def training_kws(self) -> dict:
        kws = dictools.modify_dict(self.cfg.training, exclude=['cv'], copy=True)
        kws.update(self._kwargs)
        if self.early_stop_epoch:
            assert self.early_stop_epoch <= kws['epochs']
            kws['epochs'] = self.early_stop_epoch
        return kws

    @property
    def model_file(self) -> Path:
        model_file = CvModelsManager.make_model_file_path(cfg=self.cfg, fold=self.fold)
        if self.dbg_run:
            model_file = model_file.with_stem(model_file.stem + '.DBG')
        return model_file

    @property
    def tensorboard_dir(self) -> Path:
        return paths.TENSORBOARD_DIR / self.model_file.stem

    def dispatch(self) -> bool:

        model_file = self.model_file
        tensorboard_dir = self.tensorboard_dir

        if model_file.is_file():

            skip = False
            if self.exists_handling == "skip":
                skip = True
            elif self.exists_handling == "warm_start":
                completed_epochs = 0
                try:
                    checkpoint_meta = dlutils.SnapshotMgr(model_file).get_metas()['checkpoint']
                    completed_epochs = checkpoint_meta['epoch']
                except:
                    pass
                skip = self.training_kws['epochs'] <= (completed_epochs + 1)

            if skip:
                print(str(model_file), "SKIPPED")
                return False

        print(str(model_file), "RUNNING")
        model_file.touch()  # mark model as exists as soon as we decide to work on it

        train_sameness, val_sameness = self.load_split_sameness()
        model = self.init_model(input_size=train_sameness.X.shape[1])

        triplet_train(train_sameness=train_sameness,
                      val_sameness=val_sameness,
                      model=model,
                      model_file=model_file,
                      tensorboard_dir=tensorboard_dir,
                      **self.training_kws,
                      base_meta=self.as_dict(),
                      dbg_run=self.dbg_run,
                      exists_handling=self.exists_handling)

        return True


def cv_train(exists_handling: Literal["warm_start", "overwrite", "skip", "error"] = "skip",
             dbg_run: bool = False,
             early_stop_epoch: int = None,
             cfg_name_include: str = None,
             cfg_name_exclude: str = None):

    cfgs = [cfg for cfg in Config.yield_from_grid()]
    max_folds = max([cfg.training.cv.folds for cfg in cfgs])

    grid_cfgs_df = pd.DataFrame.from_records(dictools.variance_dicts([cfg.__dict__ for cfg in cfgs],
                                                                     force_keep=['data.pairing.balance']))

    CvModelsManager.refresh_results_file()
    for fold in range(max_folds):
        for cfg_ix, cfg in enumerate(cfgs):

            if fold >= cfg.training.cv.folds:
                continue
            if cfg_name_include is not None and cfg_name_include not in cfg.str():
                continue
            if cfg_name_exclude is not None and cfg_name_exclude in cfg.str():
                continue

            print(f"cfg {cfg_ix + 1}/{len(cfgs)}, fold {fold}:")
            print(grid_cfgs_df.loc[cfg_ix].to_string())

            training_mgr = TrainingMgr(cfg, fold=fold, dbg_run=dbg_run, early_stop_epoch=early_stop_epoch,
                                       exists_handling=exists_handling)
            success = training_mgr.dispatch()
            if success:
                CvModelsManager.refresh_results_file()


def SCRIPT_cv_train_by_modifying_cfg():
    cfg_or_rank = 0
    dbg_run = False
    fold = 0
    early_stop_epoch = 20
    exists_handling = "overwrite"

    if isinstance(cfg_or_rank, int):
        cfg, _ = CvModelsManager.get_config_and_files_by_rank(cfg_or_rank)
    else:
        assert isinstance(cfg_or_rank, Config)
        cfg = Config

    print("Modifying config", cfg.output_name)

    # cfg.data.pairing.metric = 'rigid'
    # training_mgr = TrainingMgr(cfg, fold=fold, dbg_run=dbg_run, early_stop_epoch=early_stop_epoch,
    #                            exists_handling=exists_handling)
    # success = training_mgr.dispatch()
    # CvModelsManager.refresh_results_file()


if __name__ == "__main__":
    #SCRIPT_cv_train_by_modifying_cfg()
    #CvModelsManager.get_config_and_files_by_rank(2)
    cv_train(exists_handling="skip", dbg_run=False, early_stop_epoch=20)
    #CvModelsManager.refresh_results_file()
    #cv_train(exists_handling="overwrite", dbg_run=False, cfg_name_include="f2688c")
    #cv_train(exists_handling="overwrite", dbg_run=False, cfg_name_exclude="f2688c")
