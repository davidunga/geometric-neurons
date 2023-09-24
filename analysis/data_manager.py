import pandas as pd
from scipy import stats
from motorneural.datasets.hatsopoulos import HatsoData, HATSO_DATASETS, Data
from motorneural.data import Segment, DataDef
from motorneural.motor import KinData
from motorneural.neural import NeuralData
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from procrustes import Procrustes
from common import linalg
import paths
import os
from pathlib import Path
import pickle
from typing import Sequence
from common.symmetric_pairs import SymmetricPairsData



def make_data(dataset: str, lag: float = .1, bin_sz: float = .01) -> Data:
    """
    dataset = "TP_RS" / "TP_RJ"
    """

    if dataset in HATSO_DATASETS:
        data_dir = paths.GLOBAL_DATA_DIR + "/hatsopoulos"
        data = HatsoData.make(data_dir, dataset, lag=lag, bin_sz=bin_sz)
    else:
        raise ValueError("Cannot determine data source")

    return data


#
# @dataclass
# class SegmentPairs:
#
#     def __init__(self, segment_uids: list[str], dists: pd.DataFrame):
#         self.__setstate__({'segment_uids': segment_uids, 'dists': dists})
#
#     def __setstate__(self, state):
#         self._pair_data = SymmetricPairwiseData(data=state['dists'], n=len(state['segment_uids']))
#         self._segment_uids = state['segment_uids']
#         self._participating_seg_ixs = set(self.dists[['seg1', 'seg2']])
#
#     def __getstate__(self):
#         return {'segment_uids': self._segment_uids, 'dists': self.dists}
#
#     @property
#     def dists(self) -> pd.DataFrame:
#         return self._pair_data.data
#
#     @property
#     def num_segments(self) -> int:
#         return len(self._participating_seg_ixs)
#
#     def dists(self, seg_ix: int):
#         ixs = symm_pair_indices(seg_ix, self.num_segments)
#         return self._dists.iloc[ixs]


def calc_segment_pairs(segments: list[Segment], procrustes_kind: str = 'affine') -> SymmetricPairsData:

    procrustes = Procrustes(kind=procrustes_kind)
    print("computing...")
    num_pairs = len(segments) * (len(segments) - 1) // 2
    dists = [{} for _ in range(num_pairs)]
    count = -1
    for i in range(len(segments) - 1):
        seg1 = segments[i]
        for j in range(i + 1, len(segments)):
            count += 1
            if count % 500 == 0:
                print(f'{count}/{num_pairs}')
            seg2 = segments[j]
            proc_dist, A = procrustes(seg1.kin.X, seg2.kin.X)
            b, ang, t, is_reflected, ortho_score = linalg.planar.decompose(A)
            scale = b if b > 1 else 1 / b
            loc = float(np.linalg.norm(t))
            pair_item = {'seg1': seg1.uid,
                         'seg2': seg2.uid,
                         'scale': scale,
                         'ang': ang,
                         'loc': loc,
                         'proc_dist': proc_dist,
                         'uniform_scale_score': ortho_score,
                         'reflected': is_reflected}
            dists[count] = pair_item

    dists = pd.DataFrame.from_records(dists)
    seg_pairs = SymmetricPairsData(data=dists, n=len(segments))
    return seg_pairs


@dataclass
class DataFiles:

    datadef: DataDef
    data_root: Path = paths.PROJECT_DATA_DIR

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        assert self.data_root.exists()

    @property
    def data_pkl(self) -> Path:
        return self.data_root / (self.datadef.data_str() + ".data.pkl")

    def load_data(self) -> Data:
        return pickle.load(self.data_pkl.open('rb'))

    @property
    def segments_pkl(self) -> Path:
        return self.data_root / (self.datadef.segments_str() + ".segs.pkl")

    def load_segments(self) -> list[Segment]:
        return pickle.load(self.segments_pkl.open('rb'))

    @property
    def dists_pkl(self) -> Path:
        return self.data_root / (self.datadef.dists_str() + ".dists.pkl")

    def load_dists(self) -> SymmetricPairsData:
        return pickle.load(self.dists_pkl.open('rb'))


def make_and_save(datadef: DataDef, force: bool = False):

    datafiles = DataFiles(datadef)

    if force or not datafiles.data_pkl.exists():
        data = make_data(dataset=datadef.name, lag=datadef.lag, bin_sz=datadef.bin_sz)
        pickle.dump(data, datafiles.data_pkl.open('wb'))
    data = datafiles.load_data()

    if force or not datafiles.segments_pkl.exists():
        segments = extract_segments(data, segment_dur=datadef.seg_dur, radcurv_bounds=datadef.seg_radcurv_bounds)
        pickle.dump(segments, datafiles.segments_pkl.open('wb'))
    segments = datafiles.load_segments()

    if force or not datafiles.dists_pkl.exists():
        dists = calc_segment_pairs(segments=segments, procrustes_kind=datadef.proc_kind)
        pickle.dump(dists, datafiles.dists_pkl.open('wb'))
    dists = datafiles.load_dists()

    return segments


def extract_segments(data: Data, segment_dur: float, radcurv_bounds: tuple[float, float]) -> list[Segment]:

    SANE_SEGMENT_SIZE_RANGE = 10, 50
    assert len(radcurv_bounds) == 2 and radcurv_bounds[0] < radcurv_bounds[1]

    k2_min = 1 / radcurv_bounds[1]
    k2_max = 1 / radcurv_bounds[0]

    DBG_PLOT = False

    r = int(round(.5 * segment_dur / data.bin_sz))
    segment_size = 2 * r + 1
    assert SANE_SEGMENT_SIZE_RANGE[0] < segment_size < SANE_SEGMENT_SIZE_RANGE[1]

    segments = []
    for trial in data:

        if trial.ix > 10:
            break

        if DBG_PLOT:
            plt.plot(*trial.kin.X.T, 'k')
            plt.gca().set_aspect(1)
            plt.title(f"{trial.dataset} Trial {trial.ix}")

        k2 = np.abs(trial.kin['EuCrv'])

        # exclude boundaries
        k2[:r] = 0
        k2[-r:] = 0

        # exclude high curvatures
        for ix in np.nonzero(k2 > k2_max)[0]:
            k2[(ix - r): (ix + r + 1)] = 0

        while True:

            ix = np.argmax(k2)
            if k2[ix] < k2_min:
                break

            k2_val = k2[ix]
            k2[ix - r: ix + r + 1] = 0

            segment = trial.get_segment(ix - r, ix + r + 1)
            assert len(segment) == segment_size
            segments.append(segment)

            if DBG_PLOT:
                color = np.random.rand(3,)
                plt.plot(*segment.kin.X.T, '-', color=color)
                plt.plot(*trial.kin.X[ix], '.', color=color)
                plt.plot(*trial.kin.X[ix-r], 's', color=color)
                plt.plot(*trial.kin.X[ix + r], 's', color=color)
                plt.text(*segment.kin.X.mean(axis=0), segment.uid, color=color)
                #plt.text(*segment.kin.X.mean(axis=0), f"{1/k2_val:2.3f}", color=color)

        if DBG_PLOT: plt.show()

    return segments


if __name__ == "__main__":
    get_segments()
