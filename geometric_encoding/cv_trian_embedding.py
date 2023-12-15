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


class CvModelsManager:

    RESULTS_FILE = paths.CV_DIR / "results.txt"

    @staticmethod
    def make_model_file_path(cfg: Config | str, fold: int | str) -> Path:
        cfg_token = cfg if isinstance(cfg, str) else cfg.str()
        fold_token = fold if isinstance(fold, str) else ('Full' if fold == -1 else f'Fold{fold}')
        return paths.MODELS_DIR / f'{cfg_token}.{fold_token}.pth'

    @staticmethod
    def get_model_files(cfg: Config = None, fold: int = None) -> list[Path]:
        return ostools.ls(CvModelsManager.make_model_file_path(
            cfg="*" if cfg is None else cfg, fold="*" if fold is None else fold), aspath=True)

    @staticmethod
    def get_tensorboard_dirs_from_hint(hint) -> list[Path]:
        model_files = CvModelsManager.model_files_from_hint(hint)
        tbdirs = []
        for model_file in model_files:
            tbdir = paths.TENSORBOARD_DIR / model_file.stem
            assert tbdir.is_dir()
            tbdirs.append(tbdir)
        return tbdirs

    @staticmethod
    def model_files_from_hint(hint) -> list[Path]:
        # model hints:
        # (wildcard to) file path
        # (wildcard to) full name
        # (wildcard to) basename + fold
        # sort_by + rank

        if isinstance(hint, (str, Path)):
            # wildcard to name or filepath
            for maybe in [hint, paths.MODELS_DIR / hint, paths.MODELS_DIR / (hint + '.pth')]:
                files = ostools.ls(maybe, aspath=True)
                if files:
                    return files
            raise ValueError(f"Unable to find model file for {hint}")

        assert isinstance(hint, (list, tuple))
        assert len(hint) == 2

        is_sort_by = isinstance(hint[0], str) and " " not in hint[0]
        if is_sort_by:
            # (sort_by, rank)
            sort_by, ranks = hint
            agg_df = CvModelsManager.get_aggregated_results(sort_by=sort_by)
            if isinstance(ranks, int):
                ranks = [ranks]
            files = []
            for rank in ranks:
                row = agg_df.iloc[rank]
                file = CvModelsManager.get_model_files(row['base_name'])[0]
                files.append(file)
            return files
        else:
            # should be (cfg, fold) or (name, fold)
            return CvModelsManager.get_model_files(*hint)

    @staticmethod
    def improvement_scores(model_hint, win_size = .9) -> dict[str, float]:
        tbdir = CvModelsManager.get_tensorboard_dirs_from_hint(model_hint)
        assert len(tbdir) == 1
        tbdir = tbdir[0]
        df = dlutils.load_tensorboard_as_df(tbdir, smooth_sigma=3)
        scores = {}
        for dir_name in df['dir_name'].unique():
            if 'tscore' in dir_name.lower():
                continue
            v = df[df['dir_name'] == dir_name]['smoothed_value'].to_numpy()
            if win_size < 1:
                sz = int(len(v) * win_size)
            else:
                sz = min(win_size, len(v))
            v = v[-sz:]
            scores[dir_name] = (v[-1] - v[0]) / np.std(v)
            if len(v) > 30:
                scores[dir_name] = .0
            #scores[dir_name] = float(np.median(np.diff(v) / v[:-1]) * np.std(v))
            if 'loss' in dir_name.lower():
                scores[dir_name] *= -1
        return scores

    @staticmethod
    def get_catalog(full: bool = False) -> pd.DataFrame:
        model_files = CvModelsManager.get_model_files()
        items = []
        for model_file in model_files:
            status, meta = CvModelsManager.get_meta_and_training_status(model_file)
            if status != 'complete':
                continue
            create_time = ostools.stats(model_file)['create'].strftime("%Y-%m-%d %H:%M")
            imprv_scores = CvModelsManager.improvement_scores(model_file.stem)
            imprv_scores = {f'improve.{k.lower()}': v for k, v in imprv_scores.items()}
            train_stop = '{stop_reason} [epoch {stop_epoch}]'.format(**meta['train_status'])
            items.append({"time": create_time, "file": str(model_file), "base_name": meta["base_name"],
                          "train_stop": train_stop, "fold": meta["fold"], "cfg": meta["cfg"], **meta["val"],
                          **imprv_scores})
        df = pd.DataFrame.from_records(items)
        variance_cfgs = dictools.variance_dicts([item["cfg"] for item in items])
        grid_df = pd.DataFrame.from_records(variance_cfgs)
        grid_df.rename(columns={c: f'grid.{c}' for c in grid_df.columns}, inplace=True)
        df["stop_epochs"] = [int(s[-4:-1]) for s in df['train_stop']]
        df["stop_reason"] = [s.split(' [')[0] for s in df['train_stop']]
        df.drop(columns=['train_stop'], inplace=True)
        if not full:
            metric_cols = [col for col in df.columns if df[col].dtype.kind == 'f' and not col.startswith("improve")]
            df = df[["time", "base_name", "fold", "stop_epochs", "stop_reason", "improve.loss_val"] + metric_cols]
        df = pd.concat([df, grid_df], axis=1)
        return df

    @staticmethod
    def get_aggregated_results(sort_by: str = 'mean_auc'):
        catalog_df = CvModelsManager.get_catalog()
        grid_cols = [col for col in catalog_df.columns if col.startswith("grid")]
        nonmetric_cols = ["time", "base_name", "fold", "stop_epochs", "stop_reason"] + grid_cols
        metric_cols = [col for col in catalog_df.columns if col not in nonmetric_cols]
        items = []
        for base_name in catalog_df['base_name'].unique():
            rows = catalog_df[catalog_df['base_name'] == base_name]
            stop_epochs = rows['stop_epochs'].to_numpy()
            max_epochs = stop_epochs.max()
            time_for_max_epochs = rows['time'].iloc[np.argmax(stop_epochs)]
            stop_reason_for_max_epochs = rows['stop_reason'].iloc[np.argmax(stop_epochs)]
            item = {'time': time_for_max_epochs, 'base_name': base_name, 'fold_count': len(rows),
                    'max_epochs': max_epochs, 'stop_reason': stop_reason_for_max_epochs}
            for col in metric_cols:
                metric_values = rows[col].to_numpy()
                item[f'mean_{col}'] = np.mean(metric_values)
            item.update(rows.iloc[0][grid_cols].to_dict())
            items.append(item)
        df = pd.DataFrame.from_records(items)
        df = df.sort_values(by=sort_by, ascending=False, ignore_index=True)
        return df

    @staticmethod
    def refresh_results_file(sort_by: str = 'mean_auc'):
        agg_df = CvModelsManager.get_aggregated_results(sort_by=sort_by)
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

    print("Model:")
    print(model)
    print("\n")

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


def plot_tensorboard(model_hints):
    from common.utils import dlutils
    from common.utils import ostools
    import matplotlib.pyplot as plt
    import json

    # model hints:
    # file path
    # full name
    # basename + fold
    # scoreby + rank

    # if not isinstance(model_name_or_ranks, Iterable):
    #     model_name_or_ranks = [model_name_or_ranks]
    #
    # agg_results_df = CvModelsManager.get_aggregated_results(sort_by=score_by)
    #
    # for model_name_or_rank in model_name_or_ranks:
    #     if isinstance(model_name_or_rank, int):
    #         rank = model_name_or_rank
    #     else:
    #         assert isinstance(model_name_or_rank, str)
    #         rank = None
    #         for i in range(len(agg_results_df)):
    #             if model_name_or_rank.startswith(agg_results_df.iloc[i]['base_name']):
    #                 assert rank is None
    #                 rank = i
    #         assert isinstance(rank, int)
    #
    #     row = agg_results_df.iloc[rank]
    #     model_name = str(row['base_name'])
    #
    #     txt = f"max_epochs={row['max_epochs']}, stop_reason={row['stop_reason']} "
    #     txt += json.dumps({k.replace('grid.', ''): v for k, v in row.to_dict().items() if k.startswith('grid')})
    #
    #     tbdirs = ostools.ls(paths.TENSORBOARD_DIR / (model_name + '*'))

    tbdirs = CvModelsManager.get_tensorboard_dirs_from_hint(model_hints)
    for tbdir in tbdirs:
        dlutils.plot_tensorboard(tbdir)

    plt.show()



if __name__ == "__main__":
    #plot_tensorboard(('time', 0))
    #plot_tensorboard(range(0,6), score_by='mean_improve.loss_val')
    #plot_tensorboard(('TP_RJ bin10 lag100 dur200 procAffine 010fea', 0))
    #CvModelsManager.refresh_results_file(sort_by='mean_improve.loss_val')
    #CvModelsManager.refresh_results_file(sort_by='time')
    cv_train()
