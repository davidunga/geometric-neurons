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

        @verbolize()
        def _calc_sameness_sign():
            cfg = self.cfg
            dists = pairs.data[cfg.pairing.dist]
            pctls = np.array([cfg.pairing.same_pctl, cfg.pairing.notSame_pctl, cfg.pairing.exclude_pctl]) * 100
            same_thresh, notSame_thresh, exclude_thresh = np.percentile(dists, pctls)
            sameness = np.zeros_like(dists, int)
            sameness[dists <= same_thresh] = 1
            sameness[(dists < exclude_thresh) & (dists >= notSame_thresh)] = -1
            return sameness

        pairs = pickle.load(self.pkl_path(DataConfig.PAIRING).open('rb'))
        pairs.data['sameness'] = _calc_sameness_sign()
        return pairs

    @verbolize()
    def get_neurals_df(self, segmets: list[Segment]) -> pd.DataFrame:
        """ i-th row = processed (reduced and flattened) activations of i-th segment.
            column names = '<neuron_name>.<time_bin>'
        """
        neural_data = [s.neural.get_binned(bin_sz=self.cfg.sameness.flat_neural_bin_sz)._df for s in segmets]
        if self.cfg.sameness.normalize_neural:
            mu = np.mean(np.concatenate(neural_data, axis=0), axis=0)
            sd = np.maximum(np.std(np.concatenate(neural_data, axis=0), axis=0), 1e-6)
            neural_data = [(d - mu) / sd for d in neural_data]
        serieses = [d.stack() for d in neural_data]
        df = pd.concat(serieses, axis=1)
        df.index = [f'{col_name}.{index}' for index, col_name in df.index]
        df = df.T
        n_bins, n_neurons = neural_data[0].shape
        assert df.shape[1] == n_bins * n_neurons
        assert df.shape[0] == len(segmets)
        print(f"Neural data segment size = ({n_neurons} neurons) x ({n_bins} bins) = {df.shape[1]} flat bins.")
        return df

    @verbolize()
    def load_sameness(self) -> tuple[SamenessData, SymmetricPairsData, list[Segment]]:
        pairs = self.load_pairing()
        segments = self.load_segments()
        self.assert_pairs_and_segments_compatibility(pairs, segments)
        sameness_data = SamenessData.from_sameness_sign(
            X=self.get_neurals_df(segments), sameness=pairs['sameness'],
            triplet_min_prevalence=self.cfg.sameness.triplet_min_prevalence)
        return sameness_data, pairs, segments

    @staticmethod
    def assert_pairs_and_segments_compatibility(pairs: SymmetricPairsData, segments: list[Segment]):
        """ check that pairs are based on segments, including their ordering """
        n_samples = 50
        for seg_ix in range(0, len(segments), len(segments) // n_samples):
            uids = pairs.data_of_item(seg_ix)[['seg1', 'seg2']].values
            assert np.all(np.sum(uids == segments[seg_ix].uid, axis=1) == 1)
