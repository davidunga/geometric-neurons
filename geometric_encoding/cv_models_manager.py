import numpy as np
import pandas as pd
from analysis.config import Config
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
        if not len(df):
            return df
        variance_cfgs = dictools.variance_dicts([item["cfg"] for item in items])
        grid_df = pd.DataFrame.from_records(variance_cfgs)
        grid_df.rename(columns={c: f'grid.{c}' for c in grid_df.columns}, inplace=True)
        df["stop_epochs"] = [int(s.split(' ')[-1][:-1]) for s in df['train_stop']]
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
        if not len(catalog_df):
            return catalog_df
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
    def get_meta_and_training_status(model_file: PathLike, train_time_tol: float = 7200.):
        if Path(model_file).is_file():
            meta = checkpoint.get_meta(model_file)
            if meta["train_status"]["done"]:
                return 'complete', meta
        for file in glob(str(model_file) + "*"):
            if time() - os.path.getmtime(file) < train_time_tol:
                meta = checkpoint.get_meta(file)
                return 'in progress', meta
        return 'none', {}
