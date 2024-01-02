import pandas as pd
from motorneural.datasets.hatsopoulos import make_hatso_data, HATSO_DATASET_SPECS
from motorneural.data import Segment, NeuralData, Data
import numpy as np
import matplotlib.pyplot as plt
from common.utils.procrustes import Procrustes
from common.utils import linalg
import paths
import pickle
from common.utils.typings import *
from common.symmetric_pairs import SymmetricPairsData
from analysis.config import DataConfig, Config
from analysis.data_manager import DataMgr


def make_and_save(cfg: DataConfig, force: bool = False) -> None:

    loaded = {}

    def _calc_level(level: DataConfig.Level):
        if level == DataConfig.BASE:
            return make_base_data(dataset=cfg.base.name, lag=cfg.base.lag, bin_sz=cfg.base.bin_sz,
                                  kin_as_neural=cfg.base.kin_as_neural)
        elif level == DataConfig.SEGMENTS:
            return extract_segments(data=loaded[DataConfig.BASE], dur=cfg.segments.dur,
                                    radcurv_bounds=cfg.segments.radcurv_bounds)
        elif level == DataConfig.PAIRING:
            return calc_pairing(segments=loaded[DataConfig.SEGMENTS], pairing_variable=cfg.pairing.variable,
                                pairing_metric=cfg.pairing.metric)
        else:
            raise ValueError("Unknown level")

    data_mgr = DataMgr(cfg)
    for level in [DataConfig.BASE, DataConfig.SEGMENTS, DataConfig.PAIRING]:
        pkl = data_mgr.pkl_path(level)
        if force or not pkl.exists():
            pickle.dump(_calc_level(level), pkl.open('wb'))
        loaded[level] = pickle.load(pkl.open('rb'))

    return


# ====================================================================================


def make_base_data(dataset: str, lag: float, bin_sz: float, kin_as_neural: bool) -> Data:
    assert abs(lag) <= 1, "Lag should be in seconds"
    assert .001 < bin_sz <= 1, "Bin size should be in seconds"
    if dataset in HATSO_DATASET_SPECS:
        data = make_hatso_data(paths.GLOBAL_DATA_DIR / "hatsopoulos", dataset, lag=lag, bin_sz=bin_sz)
    else:
        raise ValueError("Cannot determine data source")
    if kin_as_neural:
        for trial in data:
            trial.neural = NeuralData(df=trial.kin._df, t=trial.kin.t)
    return data


def extract_segments(data: Data, dur: float, radcurv_bounds: tuple[float, float]) -> list[Segment]:

    SANE_SEGMENT_SIZE_RANGE = 10, 50
    assert len(radcurv_bounds) == 2 and radcurv_bounds[0] < radcurv_bounds[1]

    k2_min = 1 / radcurv_bounds[1]
    k2_max = 1 / radcurv_bounds[0]

    DBG_PLOT = False
    DRYRUN = False

    r = int(round(.5 * dur / data.bin_sz))
    segment_size = 2 * r + 1
    assert SANE_SEGMENT_SIZE_RANGE[0] < segment_size < SANE_SEGMENT_SIZE_RANGE[1]

    print(f"Extracting segments from {len(data)} trials..")

    segments = []
    for trial in data:

        if DRYRUN and trial.ix > 100:
            break

        if trial.ix % 100 == 0:
            print(f'{trial.ix + 1}/{len(data)}')

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

    n_trials = len(set(s.trial_ix for s in segments))
    print(f"Done. Extracted {len(segments)} segments from {n_trials} trials. "
          f"That's {len(segments) / n_trials:2.2f} segment/trial on average.")
    return segments


def calc_pairing(segments: list[Segment],
                 pairing_variable: str,
                 pairing_metric: str) -> SymmetricPairsData:

    X = [s[pairing_variable][:, :2] for s in segments]
    procrustes = Procrustes(kind=pairing_metric)

    print(f"Computing segments pairing {pairing_variable} {pairing_metric} ...")
    num_pairs = len(segments) * (len(segments) - 1) // 2
    dists = [{} for _ in range(num_pairs)]
    count = -1

    for i in range(len(segments) - 1):
        for j in range(i + 1, len(segments)):
            count += 1
            if count % 50_000 == 0:
                print(f'{count}/{num_pairs} ({count/num_pairs:3.2%})')
            proc_dist, A, _ = procrustes(X[i], X[j])
            b, ang, t, is_reflected, ortho_score = linalg.planar.decompose(A)
            scale = b if b > 1 else 1 / b
            loc = float(np.linalg.norm(t))
            pair_item = {'seg1': segments[i].uid,
                         'seg2': segments[j].uid,
                         'scale': scale,
                         'ang': ang,
                         'loc': loc,
                         'proc_dist': proc_dist,
                         'uniform_scale_score': ortho_score,
                         'reflected': is_reflected}
            dists[count] = pair_item

    print("Done.")
    dists = pd.DataFrame.from_records(dists)
    seg_pairs = SymmetricPairsData(data=dists, n=len(segments))
    return seg_pairs


def run__make_and_save():
    force = True
    for pairing_variable in ['neuralPc']:
        for pairing_metric in ['ortho', 'affine']:
            for bin_sz in [.01, .02]:
                for dataset in ['TP_RJ', 'TP_RS']:
                    data_cfg = Config.from_default().data
                    data_cfg.base.name = dataset
                    data_cfg.base.bin_sz = bin_sz
                    data_cfg.pairing.metric = pairing_metric
                    data_cfg.pairing.variable = pairing_variable
                    make_and_save(data_cfg, force=force)


if __name__ == "__main__":
    run__make_and_save()
    pass
