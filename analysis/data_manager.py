import time

import pandas as pd
from motorneural.data import Segment, Data
import numpy as np
from pathlib import Path
import paths
import pickle
from common.symmetric_pairs import SymmetricPairsData
from common.pairwise.sameness import SamenessData
from analysis.config import DataConfig, Config
from common.utils.devtools import verbolize
from common.utils.typings import *


class DataMgr:

    def __init__(self, data_cfg: DataConfig, data_root: (str, Path) = paths.DATA_DIR, verbose: int = 1):
        self.cfg: DataConfig = data_cfg
        self.data_root: Path = Path(data_root)
        self.verbose = verbose
        assert self.data_root.exists()

    @classmethod
    def from_default_config(cls):
        return cls(Config.from_default().data)

    def pkl_path(self, level: DataConfig.Level) -> Path:
        return paths.DATA_DIR / (self.cfg.str(level) + f'.{level}.pkl')

    @verbolize()
    def load_base_data(self) -> Data:
        return pickle.load(self.pkl_path(DataConfig.BASE).open('rb'))

    @verbolize()
    def load_segments(self) -> list[Segment]:
        return pickle.load(self.pkl_path(DataConfig.SEGMENTS).open('rb'))

    @verbolize()
    def load_pairing(self) -> SymmetricPairsData:

        verbolize.open("loading pickle", inline=True)
        pairs = pickle.load(self.pkl_path(DataConfig.PAIRING).open('rb'))
        verbolize.close()

        if 'proc_dist_rank' not in pairs.data:
            pairs.data['proc_dist_rank'] = pairs.data['proc_dist'].argsort().argsort()
            pickle.dump(pairs, self.pkl_path(DataConfig.PAIRING).open('wb'))

        ranks = pairs.data['proc_dist_rank']
        same_rank = int(self.cfg.pairing.same_pctl * len(ranks))
        notSame_rank = int(self.cfg.pairing.notSame_pctl * len(ranks))
        exclude_rank = int(self.cfg.pairing.exclude_pctl * len(ranks))

        if self.cfg.pairing.balance:

            # from ranks to counts
            n_same = same_rank
            n_notSame = exclude_rank - notSame_rank

            # counts after balance
            n_same = n_notSame = min(n_same, n_notSame)

            # from balanced counts back to ranks
            same_rank = n_same
            notSame_rank = exclude_rank - n_notSame

        sameness = (ranks < same_rank).astype(int)
        sameness[(ranks < exclude_rank) & (ranks >= notSame_rank)] = -1

        pairs.data['sameness'] = sameness

        return pairs

    @verbolize()
    def get_predictor_df(self, segmets: list[Segment]) -> pd.DataFrame:
        """ i-th row = processed (reduced and flattened) activations of i-th segment.
            column names = '<neuron_name>.<time_bin>'
        """

        preds = [s[self.cfg.predictor.variable].get_binned(bin_sz=self.cfg.predictor.bin_sz)._df
                 for s in segmets]

        verbolize.inform(f"Time re-binned {self.cfg.base.bin_sz} -> {self.cfg.predictor.bin_sz} ="
                         f" (x{int(.5+self.cfg.predictor.bin_sz/self.cfg.base.bin_sz)} decimation)")

        if self.cfg.predictor.normalize:
            mu = np.mean(np.concatenate(preds, axis=0), axis=0)
            sd = np.maximum(np.std(np.concatenate(preds, axis=0), axis=0), 1e-6)
            preds = [(d - mu) / sd for d in preds]

        serieses = [d.stack() for d in preds]
        df = pd.concat(serieses, axis=1)
        df.index = [f'{col_name}.{index}' for index, col_name in df.index]
        df = df.T

        n_bins, n_units = preds[0].shape
        assert df.shape[1] == n_bins * n_units
        assert df.shape[0] == len(segmets)

        match self.cfg.predictor.variable:
            case 'neural':
                unit = 'neurons'
            case 'neural_pc':
                unit = 'neuralPcs'
            case _:
                assert self.cfg.predictor.variable.startswith('kin')
                unit = 'features'

        verbolize.inform(f"Predictor size = ({n_units} {unit}) x ({n_bins} bins) = {df.shape[1]} flat bins.")
        return df

    @verbolize()
    def load_sameness(self) -> tuple[SamenessData, SymmetricPairsData, list[Segment]]:
        pairs = self.load_pairing()
        segments = self.load_segments()
        self.assert_pairs_and_segments_compatibility(pairs, segments)
        if self.cfg.predictor.shuffle:
            pairs.data['sameness'] = np.random.default_rng(0).permutation(pairs['sameness'].to_numpy())
            verbolize.alert("Shuffled sameness")
        sameness_data = SamenessData.from_sameness_sign(
            X=self.get_predictor_df(segments), sameness=pairs['sameness'],
            triplet_min_prevalence=self.cfg.predictor.triplet_min_prevalence)

        return sameness_data, pairs, segments

    def get_pairing_X(self, segments: list[Segment]):
        return self.make_pairing_X(self.cfg.pairing.variable, segments)

    @staticmethod
    def assert_pairs_and_segments_compatibility(pairs: SymmetricPairsData, segments: list[Segment]):
        """ check that pairs are based on segments, including their ordering """
        n_samples = 50
        for seg_ix in range(0, len(segments), len(segments) // n_samples):
            uids = pairs.data_of_item(seg_ix)[['seg1', 'seg2']].values
            assert np.all(np.sum(uids == segments[seg_ix].uid, axis=1) == 1)

    @staticmethod
    def make_pairing_X(pairing_variable: str, segments: list[Segment]):
        def _to_2d(x):
            return x[:, :2] if x.ndim >= 2 else np.c_[np.linspace(0, 1, len(x)), x]
        return [_to_2d(s[pairing_variable]) for s in segments]

    @verbolize
    @staticmethod
    def split_sameness_by_fold(sameness_data: SamenessData, fold: int, n_folds: int = None, shuff_seed: int = 0):
        if fold == -1:
            return sameness_data, None
        assert fold in range(n_folds)
        val_start = int(sameness_data.n * fold / n_folds)
        val_stop = int(round(sameness_data.n * (fold + 1) / n_folds))
        val_items = np.random.default_rng(shuff_seed).permutation(sameness_data.n)[val_start: val_stop]
        mutex_groups = sameness_data.mutex_indexes(val_items)
        sameness_data_2 = sameness_data.modcopy(index_mask=mutex_groups == 1)
        sameness_data = sameness_data.modcopy(index_mask=mutex_groups == 2)
        return sameness_data, sameness_data_2
