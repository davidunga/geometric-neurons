import json
from datetime import datetime
import numpy as np
import pandas as pd
from analysis.config import Config
import paths
from common.utils.typings import *
from common.utils.dlutils import SnapshotMgr
from common.utils import ostools
from common.utils import dictools
import shutil


class CvModelsManager:

    RESULTS_FILE = paths.CV_DIR / "results.txt"

    @staticmethod
    def make_model_file_path(cfg: Config | str, fold: int | str) -> Path:
        """
        Args:
            cfg:    Config object or wildcard of config name
            fold:   Fold number or wildcard of fold name
        """
        cfg_token = cfg if isinstance(cfg, str) else cfg.output_name
        fold_token = fold if isinstance(fold, str) else ('Full' if fold == -1 else f'Fold{fold}')
        return paths.MODELS_DIR / f'{cfg_token}.{fold_token}.pth'

    @staticmethod
    def get_model_files(cfg: Config = None, fold: int = None, **ls_kwargs) -> list[Path]:
        model_files_wildcard = CvModelsManager.make_model_file_path(
            cfg="*" if cfg is None else cfg,
            fold="*" if fold is None else fold)
        return ostools.ls(model_files_wildcard, **ls_kwargs)

    @staticmethod
    def modify_configs(flat_cfg_subdict: dict = None, modifier_func: Callable[dict, dict] = None):
        """ Modify config meta in model files, and save under updated filename. Original files are backed up.
            flat_cfg_subdict: a flat cfg subdict
            modifier_func: a function which receives current cfg dict, and returns a modified dict
        """

        if flat_cfg_subdict is not None:
            assert modifier_func is None

            def _update_config(cfg_):
                flat_cfg_ = dictools.flatten_dict(cfg_)
                flat_cfg_.update(flat_cfg_subdict)
                return dictools.unflatten_dict(flat_cfg_)
            modifier_func = _update_config

        backup_dir = paths.MODELS_DIR / "backup" / datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=False)

        model_files = [str(fn) for fn in CvModelsManager.get_model_files(allow_zero_size=False)]

        new_metas_per_file = {}
        new_filename_per_file = {}
        for model_file in model_files:
            metas = SnapshotMgr(model_file).get_metas()

            curr_cfg = new_cfg = fold = None
            for tag, meta in metas.items():
                if curr_cfg is None:
                    fold = meta['fold']
                    curr_cfg = meta['cfg']
                    new_cfg = modifier_func(curr_cfg)
                assert curr_cfg == meta['cfg']
                assert fold == meta['fold']
                meta['cfg'] = new_cfg

            new_metas_per_file[model_file] = metas
            new_filename_per_file[model_file] = CvModelsManager.make_model_file_path(cfg=Config(new_cfg), fold=fold)

            print(new_filename_per_file[model_file])
            assert not new_filename_per_file[model_file].exists()

        for model_file in model_files:
            new_filename = new_filename_per_file[model_file]
            new_metas = new_metas_per_file[model_file]
            shutil.copy(model_file, new_filename)
            SnapshotMgr(new_filename).set_metas(new_metas)
            metas = SnapshotMgr(new_filename).get_metas()
            assert new_metas == metas
            shutil.move(model_file, backup_dir)

        print(f"Modified config in {len(model_files)} files. Backup path: {str(backup_dir)}")

    @staticmethod
    def get_config_and_files_by_rank(rank: int, rank_by: str = 'mean_auc') -> tuple[Config, list[Path]]:
        df, agg_df = CvModelsManager.make_results_dfs(sort_agg_by=rank_by)
        base_name = agg_df.iloc[rank]['base_name']
        df_rows = df[df['base_name'] == base_name]
        cfg_json_dicts = df_rows['cfg'].values
        files = [Path(fn) for fn in df_rows['file'].values]
        assert np.all(cfg_json_dicts == cfg_json_dicts[0])
        assert len(set(files)) == len(df_rows)
        cfg = Config(cfg_json_dicts[0])
        assert cfg.output_name == base_name
        return cfg, files

    @staticmethod
    def make_results_dfs(sort_agg_by: str = 'mean_auc') -> tuple[pd.DataFrame, pd.DataFrame]:

        METRIC_NAMES = ['auc', 'loss', 'tscore']

        model_files = CvModelsManager.get_model_files(allow_zero_size=False)
        if not model_files:
            return pd.DataFrame(), pd.DataFrame()

        # -----
        # none-aggregated:

        items = []
        for model_file in model_files:
            file_info = ostools.FileInfo(model_file)
            metas = SnapshotMgr(model_file).get_metas()

            checkpoint_meta = metas['checkpoint']
            best_meta = metas['best']
            init_meta = metas['init']

            metrics_dict = {}
            for k in METRIC_NAMES:
                best_val = best_meta['val'][k]
                init_val = init_meta['val'][k]
                metrics_dict[k] = best_val
                metrics_dict['init_' + k] = init_val
                if k in ['auc']:
                    metrics_dict['rel_' + k] = 100 * (best_val - init_val) / init_val

            items.append({"recency_rank": 0,
                          "time": file_info.modified.strftime("%Y-%m-%d %H:%M"),
                          **metrics_dict,
                          "base_name": Config(checkpoint_meta["cfg"]).output_name,
                          "fold": checkpoint_meta["fold"],
                          "train_epochs": checkpoint_meta['train_status']['epoch'],
                          "train_state": checkpoint_meta['train_status']['state'],
                          "file": str(model_file),
                          "cfg": checkpoint_meta["cfg"]})

        df = pd.DataFrame.from_records(items)
        variance_cfgs = dictools.variance_dicts([item["cfg"] for item in items])
        grid_df = pd.DataFrame.from_records(variance_cfgs)
        grid_df.rename(columns={c: c[c.index('.') + 1:] for c in grid_df.columns}, inplace=True)
        grid_insert_ix = list(df.columns).index("base_name")
        first_cols = df.columns[:grid_insert_ix]
        last_cols = df.columns[grid_insert_ix:]
        df = pd.concat([df[first_cols], grid_df, df[last_cols]], axis=1)
        df['recency_rank'] = len(df) - np.argsort(df['time'].to_numpy(str)).argsort() - 1

        # ----------
        # aggregated per config (over folds):

        agg_items = []
        grid_cols = list(grid_df.columns)
        metric_cols = [col for col in df.columns if col.endswith(tuple(METRIC_NAMES))]
        for base_name in df['base_name'].unique():

            rows = df.loc[df['base_name'] == base_name]

            # ---
            # sanities
            # should be one per base name (same for all folds)
            for col in ['base_name'] + grid_cols:
                assert len(rows[col].unique()) == 1
            # should be one per fold:
            for col in ['fold']:
                assert len(rows[col].unique()) == len(rows)
            # ---

            latest_time = max(rows['time'].tolist())
            fold_count = len(rows)
            mean_metrics = {f'mean_{col}': np.mean(rows[col].values) for col in metric_cols}
            mean_train_epochs = int(round(rows['train_epochs'].mean()))

            agg_items.append({
                "recency_rank": 0,
                "latest_time": latest_time,
                **mean_metrics,
                **rows.iloc[0][grid_cols].to_dict(),
                "base_name": base_name,
                "fold_count": fold_count,
                "mean_train_epochs": mean_train_epochs,
            })

        agg_df = pd.DataFrame.from_records(agg_items)
        agg_df['recency_rank'] = len(agg_df) - np.argsort(agg_df['latest_time'].to_numpy(str)).argsort() - 1

        agg_df = agg_df.sort_values(by=sort_agg_by, ascending=False, ignore_index=True)

        return df, agg_df

    @staticmethod
    def refresh_results_file(sort_agg_by: str = 'mean_auc'):
        df, agg_df = CvModelsManager.make_results_dfs(sort_agg_by=sort_agg_by)
        with CvModelsManager.RESULTS_FILE.open("w") as f:
            f.write("Refresh time: " + str(pd.Timestamp.now()))
            f.write("\nSummary:\n")
            f.write(agg_df.to_string())
            f.write("\n\nCatalog:\n")
            f.write(df.to_string())
        print("Refreshed results file.")
