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
from common.utils import dictools
import os
from time import time


class CvModelsManager:

    RESULTS_FILE = paths.CV_DIR / "results.txt"

    @staticmethod
    def make_model_file_path(cfg: Config | str, fold: int | str) -> Path:
        cfg_token = cfg if isinstance(cfg, str) else cfg.str()
        fold_token = fold if isinstance(fold, str) else ('Full' if fold == -1 else f'Fold{fold}')
        return paths.MODELS_DIR / f'{cfg_token}.{fold_token}.pth'

    @staticmethod
    def get_model_files(cfg: Config = None, fold: int = None) -> list[str]:
        return glob(CvModelsManager.make_model_file_path(cfg="*" if cfg is None else cfg,
                                                         fold="*" if fold is None else fold).as_posix())

    @staticmethod
    def get_catalog(full: bool = False) -> pd.DataFrame:
        model_files = CvModelsManager.get_model_files()
        items = []
        for model_file in model_files:
            status, meta = CvModelsManager.get_meta_and_training_status(model_file)
            if status != 'complete':
                continue
            train_stop = '{stop_reason} [epoch {stop_epoch}]'.format(**meta['train_status'])
            items.append({"file": model_file, "base_name": meta["base_name"], "train_stop": train_stop,
                          "fold": meta["fold"], "cfg": meta["cfg"], **meta["val"]})
        df = pd.DataFrame.from_records(items)
        variance_cfgs = dictools.variance_dicts([item["cfg"] for item in items])
        grid_df = pd.DataFrame.from_records(variance_cfgs)
        grid_df.rename(columns={c: f'grid.{c}' for c in grid_df.columns}, inplace=True)
        if not full:
            metric_cols = [col for col in df.columns if df[col].dtype.kind == 'f']
            df = df[["base_name", "fold", "train_stop"] + metric_cols]
        df = pd.concat([df, grid_df], axis=1)
        return df

    @staticmethod
    def get_aggregated_results():
        catalog_df = CvModelsManager.get_catalog()
        grid_cols = [col for col in catalog_df.columns if col.startswith("grid")]
        nonmetric_cols = ["base_name", "fold", "train_stop"] + grid_cols
        metric_cols = [col for col in catalog_df.columns if col not in nonmetric_cols]
        items = []
        for base_name in catalog_df['base_name'].unique():
            rows = catalog_df[catalog_df['base_name'] == base_name]
            item = {'base_name': base_name, 'fold_count': len(rows)}
            for col in metric_cols:
                metric_values = rows[col].to_numpy()
                item[f'mean_{col}'] = np.mean(metric_values)
            item.update(rows.iloc[0][grid_cols].to_dict())
            items.append(item)
        df = pd.DataFrame.from_records(items)
        df = df.sort_values(by='mean_auc', ascending=False, ignore_index=True)
        return df

    @staticmethod
    def refresh_results_file():
        agg_df = CvModelsManager.get_aggregated_results()
        catalog_df = CvModelsManager.get_catalog()
        with CvModelsManager.RESULTS_FILE.open("w") as f:
            f.write("Refresh time: " + str(pd.Timestamp.now()))
            f.write("\nSummary:\n")
            f.write(agg_df.to_string())
            f.write("\n\nCatalog:\n")
            f.write(catalog_df.to_string())

    @staticmethod
    def get_meta_and_training_status(model_file: PathLike, train_time_tol: float = 1800.):
        if Path(model_file).is_file():
            meta = checkpoint.get_meta(model_file)
            if meta["train_status"]["done"]:
                return 'complete', meta
        for file in glob(str(model_file) + "*"):
            if time() - os.path.getmtime(file) < train_time_tol:
                meta = checkpoint.get_meta(file)
                return 'in progress', meta
        return 'none', {}


def single_fold_train(cfg: Config, fold: int, sameness_data: Optional[SamenessData] = None,
                      skip_existing: bool = True, verbose: int = 1) -> dict:

    """
    Train a single fold or over full data (no fold)
    Args:
        cfg: config
        fold: index of fold, or -1 for no folds
        base_name: base name for model, default = from config
        sameness_data: SamenessData object, to save loading time
    Returns:
        result dict
    """

    model_file = CvModelsManager.make_model_file_path(cfg=cfg, fold=fold)
    tensorboard_dir = paths.TENSORBOARD_DIR / model_file.stem

    result = {'skipped': False,
              'training_status': CvModelsManager.get_meta_and_training_status(model_file)[0],
              'model_file': model_file,
              'tensorboard_dir': tensorboard_dir,
              'sameness_data': sameness_data}

    if result['training_status'] != 'none' and skip_existing:
        result['skipped'] = True
        if verbose: print(model_file.as_posix(), f"- SKIPPED [training {result['training_status']}]")
        return result

    if verbose: print(model_file.as_posix(), "- RUNNING")

    if not sameness_data:
        sameness_data, _, _ = DataMgr(cfg.data).load_sameness()
        result['sameness_data'] = sameness_data

    if fold == -1:
        shuffled_items = np.random.default_rng(cfg.training.cv.folds).permutation(np.arange(sameness_data.n))
        train_sameness = sameness_data.copy(shuffled_items)
        val_sameness = None
    else:
        assert fold in range(cfg.training.cv.folds)
        shuffled_items = np.random.default_rng(fold).permutation(np.arange(sameness_data.n))
        n_train_items = int((len(sameness_data) / cfg.training.cv.folds) ** .5)
        train_sameness = sameness_data.copy(shuffled_items[:n_train_items])
        val_sameness = sameness_data.copy(shuffled_items[n_train_items:])

    model = LinearEmbedder(input_size=sameness_data.X.shape[1], **cfg.model)
    triplet_train(train_sameness=train_sameness, val_sameness=val_sameness, model=model, model_file=model_file,
                  tensorboard_dir=tensorboard_dir, **dictools.modify_dict(cfg.training, exclude=['cv'], copy=True))

    checkpoint.update_meta(model_file, base_name=cfg.str(), fold=fold, cfg=cfg.__dict__)

    return result


def cv_train(skip_existing: bool = True, cfg_before_folds: bool = True):
    """
    Args:
        skip_existing: skip existing (fully trained) models
        cfg_before_folds: iterate over configs before iterating over folds
    """

    cfgs = [cfg for cfg in Config.yield_from_grid()]
    max_folds = max([cfg.training.cv.folds for cfg in cfgs])

    CvModelsManager.refresh_results_file()
    cfg_ix = fold = 0
    sameness_data = None
    for itr in range(len(cfgs) * max_folds):

        if itr > 0:
            if cfg_before_folds:
                cfg_ix = (cfg_ix + 1) % len(cfgs)
                sameness_data = None  # new config -> reset sameness
                if cfg_ix == 0:
                    fold += 1
            else:
                fold = (fold + 1) % cfgs[cfg_ix].training.cv.folds
                if fold == 0:
                    cfg_ix += 1
                    sameness_data = None  # new config -> reset sameness

        if fold == max_folds or cfg_ix == len(cfgs):
            break
        if fold == cfgs[cfg_ix].training.cv.folds:
            continue

        print(f"cfg {cfg_ix + 1}/{len(cfgs)}, fold {fold}:", end=" ")
        result = single_fold_train(cfg=cfgs[cfg_ix], fold=fold,
                                   sameness_data=sameness_data, skip_existing=skip_existing)
        sameness_data = result['sameness_data']
        if not result['skipped']:
            print("Results so far: \n" + CvModelsManager.get_catalog(full=False).to_string() + "\n")
            CvModelsManager.refresh_results_file()


if __name__ == "__main__":
    CvModelsManager.refresh_results_file()
    cv_train()
