import json
from itertools import product
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


class CvModelsManager:

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
    def get_catalog() -> pd.DataFrame:
        model_files = CvModelsManager.get_model_files()
        items = []
        for model_file in model_files:
            is_trained, meta = CvModelsManager.check_trained_and_get_meta(model_file)
            if not is_trained:
                continue
            items.append({"file": model_file, "base_name": meta["base_name"],
                          "fold": meta["fold"], "cfg": meta["cfg"], **meta["val"]})
        df = pd.DataFrame.from_records(items)
        return df

    @staticmethod
    def check_trained_and_get_meta(model_file: PathLike):
        if not Path(model_file).is_file():
            return False, {}
        meta = checkpoint.get_meta(model_file)
        is_trained = meta["train_status"]["done"]
        return is_trained, meta


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
              'existed': CvModelsManager.check_trained_and_get_meta(model_file)[0],
              'model_file': model_file,
              'tensorboard_dir': tensorboard_dir,
              'sameness_data': sameness_data}

    if result['existed'] and skip_existing:
        result['skipped'] = True
        if verbose: print(model_file.as_posix(), "- SKIPPED")
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
    triplet_train(train_sameness=train_sameness, val_sameness=val_sameness, model=model, model_dump_file=model_file,
                  tensorboard_dir=tensorboard_dir, **dictools.modify_dict(cfg.training, exclude=['cv'], copy=True))

    checkpoint.update_meta(model_file, base_name=cfg.str(), fold=fold, cfg=cfg.__dict__)

    return result


def cv_train(skip_existing: bool = True):
    cfgs = [cfg for cfg in Config.yield_from_grid()]
    for cfg_num, cfg in enumerate(cfgs, start=1):
        sameness_data = None
        for fold in range(cfg.training.cv.folds):
            print(f"cfg {cfg_num}/{len(cfgs)}, fold {fold}:", end=" ")
            result = single_fold_train(cfg=cfg, fold=fold, sameness_data=sameness_data, skip_existing=skip_existing)
            sameness_data = result['sameness_data']
            print("Results so far: \n" + CvModelsManager.get_catalog().to_string() + "\n")


if __name__ == "__main__":
    cv_train()