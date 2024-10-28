import numpy as np
import pandas as pd
import common.utils.picklestore as pickle
import paths
from analysis.config import DataConfig, Config
from common.utils import symmetric_pairs
from common.utils.devtools import verbolize
from common.utils.typings import *
from motorneural.data import Segment, postprocess_data_slices, Trial, validate_data_slices
from common.utils import hashtools
import json
from common.utils.conics import get_conic, Conic
from analysis.costume_dataframes.pairs_df_funcs import uniform_sample_pairs


class DataMgr:

    def __init__(self, cfg: DataConfig, persist: bool = False):
        self.cfg: DataConfig = cfg
        self.persist = persist  # keep loaded segments in memory
        self._segments: list[Segment] = []

    @property
    def monkey(self) -> str:
        return self.cfg.trials.name.split('_')[-1]

    def pkl_path(self, level: DataConfig.Level, mod: str = '') -> Path:
        parts = [self.cfg.str(level), str(level), mod, 'pkl']
        return paths.DATA_DIR / '.'.join([p for p in parts if p])

    @verbolize()
    def load_trials(self) -> tuple[list[Trial], dict]:
        trials, meta = pickle.load(self.pkl_path(DataConfig.TRIALS))
        validate_data_slices(trials)
        return trials, meta

    @verbolize()
    def load_segments(self) -> list[Segment]:
        if self._segments: return self._segments
        segments = pickle.load(self.pkl_path(DataConfig.SEGMENTS))
        validate_data_slices(segments, same_len=True)
        verbolize.inform(f"Loaded {len(segments)} segments")
        if self.persist:
            self._segments = segments
        return segments

    @verbolize()
    def load_fitted_conics(self) -> tuple[list[Conic], pd.DataFrame]:
        conics_file = paths.PROCESSED_DIR / (self.cfg.str(DataConfig.SEGMENTS) + '.CONICS.json')
        items = json.load(conics_file.open('r'))
        conics = [get_conic(**fit_result['conic']) for fit_result in items['conic_fits']]
        scores_df = pd.DataFrame([{'seg_ix': fit_result['seg_ix'], ** fit_result['scores']}
                                  for fit_result in items['conic_fits']])
        return conics, scores_df

    @verbolize()
    def load_pairing(self, n_pairs: int = None) -> pd.DataFrame:
        """
        Returns:
            DataFrame
                Columns: seg1, seg2, dist, rank, isSame
                Index: pairs index
                attrs: segment_uids, num_pairs, num_segments
        """

        pkl = self.pkl_path(DataConfig.PAIRING)
        verbolize.open("loading pickle " + str(pkl), inline=True)
        pairs = pickle.load(pkl, self.cfg.pairing.dist_metric)
        verbolize.close()

        if 'segment_uids' not in pairs.attrs:
            convert_segment_uid_to_num_inplace(df=pairs, uids=[s['uid'] for s in self.load_segments()])
            pickle.dump(pairs, pkl, self.cfg.pairing.dist_metric)

        num_segments = len(pairs.attrs['segment_uids'])
        num_pairs = symmetric_pairs.num_pairs(num_segments)
        assert num_pairs == len(pairs)

        same_rank = int(self.cfg.pairing.same_pctl * len(pairs))
        notSame_rank = int(self.cfg.pairing.notSame_pctl * len(pairs))
        exclude_rank = int(self.cfg.pairing.exclude_pctl * len(pairs))

        if self.cfg.pairing.balance:

            # from ranks to counts
            n_same = same_rank
            n_notSame = exclude_rank - notSame_rank

            # counts after balance
            n_same = n_notSame = min(n_same, n_notSame)

            # from balanced counts back to ranks
            same_rank = n_same
            notSame_rank = exclude_rank - n_notSame

        ranks = pairs['rank'].to_numpy()
        sameness = (ranks < same_rank).astype(int)
        sameness[(ranks < exclude_rank) & (ranks >= notSame_rank)] = -1

        is_valid = sameness != 0
        is_same = sameness[is_valid] > 0

        if self.cfg.pairing.shuffle:
            is_same = np.random.default_rng(0).permutation(is_same)
            verbolize.alert("Shuffled sameness")

        pairs = pairs.loc[is_valid]
        pairs['isSame'] = is_same

        if n_pairs is not None:
            if n_pairs < 1:
                n_pairs = int(n_pairs * num_pairs)
                print("n_pairs=", n_pairs)
            ixs = uniform_sample_pairs(pairs_df=pairs, max_n_pairs=n_pairs)
            pairs = pairs.iloc[ixs]

        pairs.attrs.update({'num_segments': num_segments,
                            'num_pairs': num_pairs})

        return pairs

    def get_pairing_trajectories(self, segments: list[Segment] = None) -> list[NDArray]:
        segments = segments if segments is not None else self.load_segments()
        return self.make_pairing_trajectories(self.cfg.pairing.variable, segments)

    @staticmethod
    def make_pairing_trajectories(pairing_variable: str, segments: list[Segment]) -> list[NDArray]:
        def _to_2d(x):
            return x[:, :2] if x.ndim >= 2 else np.c_[np.linspace(0, 1, len(x)), x]
        return [_to_2d(s[pairing_variable]) for s in segments]

    # def get_inputs_meta(self) -> dict[str, list[str]]:
    #     _, column_names = self.get_inputs()
    #     column_neurons = [c.split('.')[0] for c in column_names]
    #     neuron_names = sorted(set(column_neurons))
    #     return {'column_names': column_names,
    #             'column_neurons': column_neurons,
    #             'neuron_names': neuron_names}

    @verbolize()
    def get_inputs(self):
        """
            return inputs matrix and columns names, such that
            i-th row = processed (reduced and flattened) activations of i-th segment.
            column names = '<neuron_name>.<time_bin>'
        """
        hash_size = 6
        inputs_hash = hashtools.calc_hash(self.cfg.inputs.__dict__, fmt='hex')[:hash_size]
        pkl = self.pkl_path(DataConfig.SEGMENTS, f'inputs{inputs_hash}')
        if not pkl.is_file():
            inputs, input_full_names = self._make_inputs()
            pickle.dump([inputs, input_full_names], pkl)
        inputs, input_full_names = pickle.load(pkl)

        input_neuron_names = [c.split('.')[0] for c in input_full_names]
        neuron_names = sorted(set(input_neuron_names))

        n_units = len(neuron_names)
        n_bins_in_segment = int(round(self.cfg.segments.dur / self.cfg.inputs.bin_sz))
        verbolize.inform(f"Segment's input vector size: "
                         f"({n_units} {self.cfg.inputs.variable} units) x ({n_bins_in_segment} bins)"
                         f" = {inputs.shape[1]} flat bins.")

        meta = {'input_full_names': input_full_names,
                'input_neuron_names': input_neuron_names}

        return inputs, meta

    def _make_inputs(self) -> tuple[NDArray, list[str]]:
        """
            make inputs matrix and columns names, such that
            i-th row = processed (reduced and flattened) activations of i-th segment.
            column names = '<neuron_name>.<time_bin>'
        """

        segments = self.load_segments()
        inputs, _ = postprocess_data_slices(
            data_slices=segments, variable=self.cfg.inputs.variable,
            new_bin_sz=self.cfg.inputs.bin_sz, inplace=False,
            drop_zero_variance=self.cfg.inputs.drop_zero_variance,
            normalize=self.cfg.inputs.normalize)

        n_units = inputs.shape[1]
        n_bins_in_segment = int(round(self.cfg.segments.dur / self.cfg.inputs.bin_sz))
        assert len(inputs) == n_bins_in_segment * len(segments)

        # reshape such that each row corresponds to a segment
        colnames = [f"{col}.{i}" for i in range(n_bins_in_segment) for col in inputs.columns]
        inputs = inputs.to_numpy().reshape(len(segments), -1)

        # reorder columns
        si = np.argsort(colnames)
        inputs = inputs[:, si]
        colnames = [colnames[i] for i in si]

        return inputs, colnames

    @verbolize()
    def get_reduced_kinematics(self, segments: list[Segment] = None, reduce: str = 'median',
                               names: list[str] = None) -> dict[str, np.ndarray[float]]:

        segments = segments if segments else self.load_segments()
        reduce_func = getattr(np, reduce)
        names = segments[0].kin.columns if names is None else names
        m = np.zeros((len(segments), len(names)), float)
        for i, s in enumerate(segments):
            m[i] = reduce_func(s.kin[names], axis=0)
        ret = {name: m[:, j] for j, name in enumerate(names)}
        return ret


def convert_segment_uid_to_num_inplace(df: pd.DataFrame, uids: list[str]) -> None:
    mapping = {uid: i for i, uid in enumerate(uids)}
    df['seg1'] = df['seg1'].map(mapping)
    df['seg2'] = df['seg2'].map(mapping)
    assert df['seg1'].to_list() == [i for i, _ in symmetric_pairs.iter_pairs(len(uids))]
    assert df['seg2'].to_list() == [j for _, j in symmetric_pairs.iter_pairs(len(uids))]
    df.attrs['segment_uids'] = uids



if __name__ == "__main__":
    cfg = Config.from_default()
    data_mgr = DataMgr(cfg.data)
    pairs = data_mgr.load_pairing()
    segments = data_mgr.load_segments()
