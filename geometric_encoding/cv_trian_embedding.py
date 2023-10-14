import numpy as np
import pandas as pd
from analysis.config import Config
from geometric_encoding.embedding import LinearEmbedder
from analysis.data_manager import DataMgr
from common.pairwise.triplet_train import triplet_train
import paths
from common.utils.devtools import printer
from common.utils.dlutils import checkpoint
from datetime import datetime

printer.level = 'debug'


def cv_train():

    fold_results = []
    cv_csv = paths.CV_DIR / f'results {datetime.now().strftime("%m-%d--%H-%M-%S")}.csv'
    cv_csv.parent.mkdir(exist_ok=True, parents=True)

    for cfg in Config.yield_from_grid():

        data_mgr = DataMgr(cfg.data)
        sameness_data, _, _ = data_mgr.load_sameness()
        model = LinearEmbedder(input_size=sameness_data.X.shape[1], **cfg.model)

        n_train_items = int((len(sameness_data) / cfg.training.cv.folds) ** .5)
        base_name = cfg.str() + " " + datetime.now().strftime("%m%d%H%M%S")
        for fold in range(cfg.training.cv.folds):
            shuffled_items = np.random.default_rng(fold).permutation(np.arange(sameness_data.n))
            train_sameness = sameness_data.copy(shuffled_items[:n_train_items])
            val_sameness = sameness_data.copy(shuffled_items[n_train_items:])
            name = base_name + f' Fold{fold}'
            model_file = paths.MODELS_DIR / (name + '.pth')
            tensorboard_dir = paths.TENSORBOARD_DIR / name
            triplet_train(train_sameness=train_sameness,
                          val_sameness=val_sameness,
                          model=model,
                          model_dump_file=model_file,
                          tensorboard_dir=tensorboard_dir,
                          **{k: v for k, v in cfg.training.items() if k != 'cv'})
            meta = checkpoint.update_meta(model_file, {'cfg': cfg.__dict__()})
            fold_results.append({"name": base_name, "fold": fold, **meta['val'], "cfg": cfg.jsons()})
            cv_df = pd.DataFrame.from_records(fold_results)
            cv_df.to_csv(cv_csv)
            print("Results so far: \n" + cv_df.to_string() + "\n")


if __name__ == "__main__":
    cv_train()
