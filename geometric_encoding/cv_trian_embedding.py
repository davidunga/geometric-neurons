import numpy as np
from analysis.config import Config
from geometric_encoding.embedding import LinearEmbedder
from analysis.data_manager import DataMgr
from common.pairwise.triplet_train import triplet_train
import paths
from common.utils.devtools import printer
from datetime import datetime

printer.level = 'debug'


def cv_train(cfg: Config):

    data_mgr = DataMgr(cfg.data)
    sameness_data, pairs, segments = data_mgr.load_sameness()
    model = LinearEmbedder(input_size=sameness_data.X.shape[1], **cfg.embedding.model)

    sameness_data.init_triplet_sampling()
    n_train_items = int((len(sameness_data) / cfg.model_selection.folds) ** .5)
    base_name = cfg.str() + " " + datetime.now().strftime("%m%d%H%M%S")
    for fold in range(cfg.model_selection.folds):
        shuffled_items = np.random.default_rng(fold).permutation(np.arange(sameness_data.n))
        train_sameness = sameness_data.copy(shuffled_items[:n_train_items])
        val_sameness = sameness_data.copy(shuffled_items[n_train_items:])
        name = base_name + f' Fold{fold}'
        triplet_train(train_sameness=train_sameness,
                      val_sameness=val_sameness,
                      model=model,
                      model_dump_file=paths.MODELS_DIR / (name + '.pth'),
                      tensorboard_dir=paths.TENSORBOARD_DIR / name,
                      **cfg.embedding.training)


if __name__ == "__main__":
    cfg = Config.from_default()
    cv_train(cfg)
