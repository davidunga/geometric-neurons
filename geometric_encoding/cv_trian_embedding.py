import json
from itertools import product
from datetime import datetime
import numpy as np
import pandas as pd
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

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @verbolize()
    def split_sameness_by_fold(self, sameness_data: SamenessData, fold: int):
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

    @property
    def training_kws(self) -> dict:
        return dictools.modify_dict(self.cfg.training, exclude=['cv'], copy=True)

    def dispatch(self, fold: int, skip_existing: bool = True) -> bool:

        model_file = CvModelsManager.make_model_file_path(cfg=self.cfg, fold=fold)
        tensorboard_dir = paths.TENSORBOARD_DIR / model_file.stem

        training_status = CvModelsManager.get_meta_and_training_status(model_file)[0]
        if training_status != 'none' and skip_existing:
            print(str(model_file), f"- SKIPPED [training {training_status}]")
            return False

        print(str(model_file), "- RUNNING")

        sameness_data, _, _ = DataMgr(self.cfg.data).load_sameness()
        train_sameness, val_sameness = self.split_sameness_by_fold(sameness_data, fold)

        model = LinearEmbedder(input_size=train_sameness.X.shape[1], **self.cfg.model)
        base_meta = dict(base_name=self.cfg.str(), fold=fold, cfg=self.cfg.__dict__)
        triplet_train(train_sameness=train_sameness, val_sameness=val_sameness, model=model, model_file=model_file,
                      tensorboard_dir=tensorboard_dir, **self.training_kws, base_meta=base_meta)

        return True


def cv_train(skip_existing: bool = True):

    cfgs = [cfg for cfg in Config.yield_from_grid()]
    max_folds = max([cfg.training.cv.folds for cfg in cfgs])

    CvModelsManager.refresh_results_file()

    for fold in range(max_folds):
        for cfg_ix, cfg in enumerate(cfgs):
            if fold >= cfg.training.cv.folds:
                continue

            print(f"cfg {cfg_ix + 1}/{len(cfgs)}, fold {fold}:", end=" ")
            success = TrainingMgr(cfg).dispatch(fold=fold, skip_existing=skip_existing)

            if success:
                print("Results so far: \n" + CvModelsManager.get_catalog(full=False).to_string() + "\n")
                CvModelsManager.refresh_results_file()

#
# def plot_tensorboard(model_hints):
#     from common.utils import dlutils
#     from common.utils import ostools
#     import matplotlib.pyplot as plt
#     import json
#
#     # model hints:
#     # file path
#     # full name
#     # basename + fold
#     # scoreby + rank
#
#     # if not isinstance(model_name_or_ranks, Iterable):
#     #     model_name_or_ranks = [model_name_or_ranks]
#     #
#     # agg_results_df = CvModelsManager.get_aggregated_results(sort_by=score_by)
#     #
#     # for model_name_or_rank in model_name_or_ranks:
#     #     if isinstance(model_name_or_rank, int):
#     #         rank = model_name_or_rank
#     #     else:
#     #         assert isinstance(model_name_or_rank, str)
#     #         rank = None
#     #         for i in range(len(agg_results_df)):
#     #             if model_name_or_rank.startswith(agg_results_df.iloc[i]['base_name']):
#     #                 assert rank is None
#     #                 rank = i
#     #         assert isinstance(rank, int)
#     #
#     #     row = agg_results_df.iloc[rank]
#     #     model_name = str(row['base_name'])
#     #
#     #     txt = f"max_epochs={row['max_epochs']}, stop_reason={row['stop_reason']} "
#     #     txt += json.dumps({k.replace('grid.', ''): v for k, v in row.to_dict().items() if k.startswith('grid')})
#     #
#     #     tbdirs = ostools.ls(paths.TENSORBOARD_DIR / (model_name + '*'))
#
#     tbdirs = CvModelsManager.get_tensorboard_dirs_from_hint(model_hints)
#     for tbdir in tbdirs:
#         dlutils.plot_tensorboard(tbdir)
#
#     plt.show()
#
#

if __name__ == "__main__":
    #plot_tensorboard(('time', 0))
    #plot_tensorboard(range(0,6), score_by='mean_improve.loss_val')
    #plot_tensorboard(('TP_RJ bin10 lag100 dur200 procAffine 010fea', 0))
    #CvModelsManager.refresh_results_file(sort_by='mean_improve.loss_val')
    #CvModelsManager.refresh_results_file(sort_by='time')
    cv_train()
