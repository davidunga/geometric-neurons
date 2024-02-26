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
from common import symmetric_pairs
from analysis.config import DataConfig, Config
from analysis.data_manager import DataMgr
from multiprocessing import Pool
from datetime import datetime
from glob import glob
import os
#import functools
#print = functools.partial(print, flush=True)


def make_and_save(cfg: DataConfig, force: bool = False) -> None:

    loaded = {}

    def _calc_level(level: DataConfig.Level):
        if level == DataConfig.BASE:
            return make_base_data(dataset=cfg.base.name, lag=cfg.base.lag, bin_sz=cfg.base.bin_sz)
        elif level == DataConfig.SEGMENTS:
            return extract_segments(data=loaded[DataConfig.BASE], dur=cfg.segments.dur,
                                    radcurv_bounds=cfg.segments.radcurv_bounds)
        elif level == DataConfig.PAIRING:
            return calc_pairing(segments=loaded[DataConfig.SEGMENTS], pairing_variable=cfg.pairing.variable,
                                pairing_metric=cfg.pairing.metric, n_workers=8)
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


def make_base_data(dataset: str, lag: float, bin_sz: float) -> Data:
    assert abs(lag) <= 1, "Lag should be in seconds"
    assert .001 < bin_sz <= 1, "Bin size should be in seconds"
    if dataset in HATSO_DATASET_SPECS:
        data = make_hatso_data(paths.GLOBAL_DATA_DIR / "hatsopoulos", dataset, lag=lag, bin_sz=bin_sz)
    else:
        raise ValueError("Cannot determine data source")
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


class PairingCalcWorker:

    def __init__(self, X: NpPoints, uids: list[str], procrustes: Procrustes, dump_root: Path | str,
                 name: str = "", report_every: int | float = .05, dump_every: int | float = .05):

        self.X = X
        self.uids = uids
        self.procrustes = procrustes
        self.dump_root = Path(dump_root)
        self.name = name
        self.report_every = report_every
        self.dump_every = dump_every

    @classmethod
    def from_segments(cls, segments: list[Segment], pairing_variable: str, **kwargs):
        X = DataMgr.make_pairing_X(pairing_variable, segments)
        return cls(X=X, uids=[s.uid for s in segments], **kwargs)

    def __call__(self, start_ix: int = 0, stop_ix: int = None):

        n_pairs_tot = symmetric_pairs.num_pairs(len(self.X))
        stop_ix = n_pairs_tot if stop_ix is None else stop_ix
        n_pairs = stop_ix - start_ix

        def _dump(items_):
            n_digits = str(len(str(n_pairs_tot)) + 1)
            fmt = "{:0" + n_digits + "d} - {:0" + n_digits + "d}.pkl"
            pkl_file = self.dump_root / fmt.format(items_[0]['pair_index'], items_[-1]['pair_index'])
            pickle.dump(items_, open(pkl_file, 'wb'))

        report_every = int(self.report_every * n_pairs) if self.report_every < 1 else self.report_every
        dump_every = int(self.dump_every * n_pairs) if self.dump_every < 1 else self.dump_every

        items = []
        pair_ix = -1
        for i, j in symmetric_pairs.iter_pairs(len(self.X)):

            pair_ix += 1
            if pair_ix < start_ix:
                continue
            if pair_ix == stop_ix:
                break

            k = pair_ix - start_ix
            if k % report_every == 0:
                name_str = "" if not self.name else (self.name + ": ")
                print(f'  {name_str}{k}/{n_pairs} ({k/n_pairs:3.2%})')

            proc_dist, _, AXj = self.procrustes(self.X[i], self.X[j])
            absAvg_dist = float(np.abs(np.mean(self.X[i] - AXj)))

            items.append({
                'pair_index': pair_ix,
                'seg1': self.uids[i],
                'seg2': self.uids[j],
                'proc_dist': proc_dist,
                'absAvg_dist': absAvg_dist
            })

            if pair_ix > 0 and pair_ix % dump_every == 0:
                _dump(items)
                items = []

        if items:
            _dump(items)

        return True


def run_pairing_calc_worker(worker: PairingCalcWorker, start_ix: int, stop_ix: int):
    return worker(start_ix, stop_ix)


def construct_dataframe_from_pkls(
        pkls: list[PathLike] | PathLike,
        delete_pkls_when_done: bool = False,
        report_step: float = .05) -> SymmetricPairsData:

    if not isinstance(pkls, list):
        pkls = glob(str(Path(pkls) / "*.pkl"))

    report_every = int(len(pkls) * report_step)

    dists = []
    ix = 0
    for pkl_ix, pkl in enumerate(pkls):
        if pkl_ix % report_every == 0:
            print(f"{pkl_ix}/{len(pkls)} ({(pkl_ix + 1) / len(pkls):2.1%})")
        worker_dists = pickle.load(open(pkl, 'rb'))
        if not len(dists):
            for _ in pkls:
                dists += worker_dists
        dists[ix: len(worker_dists)] = worker_dists
        ix += len(worker_dists)
    dists = dists[:ix]

    dists = pd.DataFrame.from_records(dists)
    dists.sort_values(by='pair_index', inplace=True, ignore_index=True)
    assert all(dists.index == dists['pair_index'])
    dists = dists.drop('pair_index', axis=1)
    n = symmetric_pairs.num_items(len(dists))
    assert len(set(dists["seg1"].tolist()).union(dists["seg2"].tolist())) == n
    seg_pairs = SymmetricPairsData(data=dists, n=n)

    if delete_pkls_when_done:
        for pkl in pkls:
            os.remove(pkl)

    return seg_pairs



def calc_pairing(segments: list[Segment],
                 pairing_variable: str,
                 pairing_metric: str,
                 n_workers: int = 8) -> SymmetricPairsData:

    print(f"Calculating pairing over {len(segments)} segments")
    print("Pairing metric=", pairing_metric)
    print("Pairing variable=", pairing_variable)

    pairs = list(symmetric_pairs.iter_pairs(len(segments)))

    print("Num pairs=", len(pairs))
    print(f"Preparing {n_workers} workers")

    split_ixs = np.round(np.linspace(0, len(pairs), n_workers + 1)).astype(int)
    X = DataMgr.make_pairing_X(pairing_variable, segments)

    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    dump_root = paths.DATA_DIR / "tmp" / f"{pairing_variable}-{pairing_metric}-{len(pairs)}-{time_str}".replace('.', '')
    dump_root.mkdir(exist_ok=False, parents=True)
    print("Dumping to:", str(dump_root))

    args = []
    for worker_ix in range(n_workers):
        start_ix = split_ixs[worker_ix]
        stop_ix = split_ixs[worker_ix + 1]
        worker = PairingCalcWorker(X=X.copy(),
                                   uids=[s.uid for s in segments],
                                   procrustes=Procrustes(kind=pairing_metric),
                                   name=f"Worker{worker_ix}",
                                   dump_root=dump_root)
        args.append((worker, start_ix, stop_ix))

    print(f"Dispatching workers")

    assert len(args) == n_workers
    if n_workers == 1:
        run_pairing_calc_worker(*args[0])
    else:
        with Pool(n_workers) as pool:
            pool.starmap(run_pairing_calc_worker, args)

    print("Workers are done. Temp files are under:", str(dump_root))

    print("Constructing dataframe")
    seg_pairs = construct_dataframe_from_pkls(dump_root, delete_pkls_when_done=True)
    print("Done.")

    return seg_pairs


def run__make_and_save():
    force = False
    for pairing_metric in ['none']:
        for bin_sz in [.01]:
            for dataset in ['TP_RS']:
                data_cfg = Config.from_default().data
                data_cfg.base.name = dataset
                data_cfg.base.bin_sz = bin_sz
                data_cfg.pairing.metric = pairing_metric
                make_and_save(data_cfg, force=force)


if __name__ == "__main__":
    #construct_dataframe_from_pkls("/Users/davidu/geometric-neurons/resources/data/tmp/kinX-ortho-25357881-20240206-124339")
    run__make_and_save()
    pass
