import pandas as pd
from motorneural.datasets.hatsopoulos import make_hatso_data, HATSO_DATASET_SPECS, get_hatso_datasets
from motorneural.data import Segment, Trial, validate_data_slices
import numpy as np
import matplotlib.pyplot as plt
import paths
from common.utils import picklestore as pickle
from common.utils.typings import *
from common import symmetric_pairs
from analysis.config import DataConfig, Config
from analysis.data_manager import DataMgr, convert_segment_uid_to_num_inplace
from multiprocessing import Pool
from datetime import datetime
from glob import glob
import os
from common.utils.planar_align import PlanarAligner
from common.utils.distance_metrics import get_metric_func


def make_and_save(cfg: DataConfig, force: int = 0, upto: DataConfig.Level = None) -> None:

    data_mgr = DataMgr(cfg)

    trials_pkl = data_mgr.pkl_path(DataConfig.TRIALS)
    if force > 2 or not trials_pkl.exists():
        trials, meta = make_trials_data(dataset=cfg.trials.name, lag=cfg.trials.lag, bin_sz=cfg.trials.bin_sz)
        pickle.dump([trials, meta], trials_pkl)
    trials, meta = data_mgr.load_trials()
    if upto == DataConfig.TRIALS:
        return

    segments_pkl = data_mgr.pkl_path(DataConfig.SEGMENTS)
    if force > 1 or not segments_pkl.exists():
        segments = extract_segments(trials=trials, dur=cfg.segments.dur, radcurv_bounds=cfg.segments.radcurv_bounds)
        pickle.dump(segments, segments_pkl)
    segments = data_mgr.load_segments()
    if upto == DataConfig.SEGMENTS:
        return

    pairing_pkl = data_mgr.pkl_path(DataConfig.PAIRING)
    if force > 0 or not pairing_pkl.exists():
        pairing = calc_pairing(segments=segments, variable=cfg.pairing.variable,
                               align_kind=cfg.pairing.align_kind, n_workers=8)
        pickle.store(pairing, str(pairing_pkl))
    pairing = data_mgr.load_pairing()

    return


# ====================================================================================


def make_trials_data(dataset: str, lag: float, bin_sz: float) -> tuple[list[Trial], dict]:
    assert abs(lag) <= 1, "Lag should be in seconds"
    assert .001 < bin_sz <= 1, "Bin size should be in seconds"
    if dataset in HATSO_DATASET_SPECS:
        trials, meta = make_hatso_data(paths.GLOBAL_DATA_DIR / "hatsopoulos", dataset, lag=lag, bin_sz=bin_sz)
    else:
        raise ValueError("Cannot determine data source")
    return trials, meta


def extract_segments(trials: list[Trial], dur: float, radcurv_bounds: tuple[float, float]) -> list[Segment]:

    validate_data_slices(trials)

    SANE_SEGMENT_SIZE_RANGE = 10, 50
    assert len(radcurv_bounds) == 2 and radcurv_bounds[0] < radcurv_bounds[1]

    k2_min = 1 / radcurv_bounds[1]
    k2_max = 1 / radcurv_bounds[0]

    DBG_PLOT = False
    DRYRUN = False

    r = int(round(.5 * dur / trials[0].bin_size))
    segment_size = 2 * r + 1
    assert SANE_SEGMENT_SIZE_RANGE[0] < segment_size < SANE_SEGMENT_SIZE_RANGE[1]

    print(f"Extracting segments from {len(trials)} trials..")

    segments = []
    for trial in trials:

        if DRYRUN and trial.ix > 100:
            break

        if trial.ix % 100 == 0:
            print(f'{trial.ix + 1}/{len(trials)}')

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

            segment = trial.get_segment(ix - r, ix + r + 1, ix=len(segments))
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

    n_trials = len(set(s.parent for s in segments))
    print(f"Done. Extracted {len(segments)} segments from {n_trials} trials. "
          f"That's {len(segments) / n_trials:2.2f} segment/trial on average.")
    return segments


class PairingCalcWorker:

    def __init__(self, X: NpPoints,
                 uids: list[str],
                 aligner: PlanarAligner,
                 dump_root: Path | str,
                 name: str = "",
                 report_every: int | float = .05,
                 dump_every: int | float = .05):

        self.X = X
        self.uids = uids
        self.aligner = aligner
        self.dump_root = Path(dump_root)
        self.name = name
        self.report_every = report_every
        self.dump_every = dump_every
        self.metrics = ['nmahal', 'absavg']

    def __call__(self, start_ix: int = 0, stop_ix: int = None):

        n_pairs_tot = symmetric_pairs.num_pairs(len(self.X))
        stop_ix = n_pairs_tot if stop_ix is None else stop_ix
        n_pairs = stop_ix - start_ix

        metric_funcs = {metric: get_metric_func(metric) for metric in self.metrics}

        def _dump(items_):
            n_digits = str(len(str(n_pairs_tot)) + 1)
            fmt = "{:0" + n_digits + "d} - {:0" + n_digits + "d}.pkl"
            pkl_file = self.dump_root / fmt.format(items_[0]['pair_index'], items_[-1]['pair_index'])
            pickle.dump(items_, open(pkl_file, 'wb'))

        report_every = int(self.report_every * n_pairs) if self.report_every < 1 else self.report_every
        dump_every = int(self.dump_every * n_pairs) if self.dump_every < 1 else self.dump_every

        items = []
        for pair_ix, (i, j) in enumerate(symmetric_pairs.iter_pairs(len(self.X))):

            if pair_ix < start_ix:
                continue
            if pair_ix == stop_ix:
                break

            k = pair_ix - start_ix
            if k % report_every == 0:
                name_str = "" if not self.name else (self.name + ": ")
                print(f'  {name_str}{k}/{n_pairs} ({k/n_pairs:3.2%})')

            AXj, _ = self.aligner(self.X[i], self.X[j])

            item = {'pair_index': pair_ix,
                    'seg1': self.uids[i],
                    'seg2': self.uids[j]}

            for metric_name, metric_func in metric_funcs.items():
                item[metric_name] = metric_func(self.X[i], AXj)

            items.append(item)

            if pair_ix > 0 and pair_ix % dump_every == 0:
                _dump(items)
                items = []

        if items:
            _dump(items)

        return True


def run_pairing_calc_worker(worker: PairingCalcWorker, start_ix: int, stop_ix: int):
    return worker(start_ix, stop_ix)


def construct_dataframes_from_pkls(
        pkls: list[PathLike] | PathLike,
        segments: list[Segment],
        delete_pkls_when_done: bool = False,
        report_step: float = .05) -> dict[str, pd.DataFrame]:

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
    assert symmetric_pairs.num_items(len(dists)) == len(segments)

    convert_segment_uid_to_num_inplace(dists, uids=[s['uid'] for s in segments])

    dists = dists.drop("pair_index", axis=1)
    metrics = [col for col in dists.columns if col not in ('seg1', 'seg2')]
    result = {}
    for metric in metrics:
        df = dists[['seg1', 'seg2', metric]].copy().rename(columns={metric: 'dist'})
        df['rank'] = df['dist'].to_numpy().argsort().argsort()
        result[metric] = df

    if delete_pkls_when_done:
        for pkl in pkls:
            os.remove(pkl)

    return result


def calc_pairing(segments: list[Segment],
                 variable: str,
                 align_kind: str,
                 n_workers: int = 8) -> dict[str, pd.DataFrame]:

    print(f"Calculating pairing over {len(segments)} segments")
    print("Pairing alignment kind=", align_kind)
    print("Pairing variable=", variable)

    pairs = list(symmetric_pairs.iter_pairs(len(segments)))

    print("Num pairs=", len(pairs))
    print(f"Preparing {n_workers} workers")

    split_ixs = np.round(np.linspace(0, len(pairs), n_workers + 1)).astype(int)
    X = DataMgr.make_pairing_trajectories(variable, segments)

    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    dump_root = paths.DATA_DIR / "tmp" / f"{variable}-{align_kind}-{len(pairs)}-{time_str}".replace('.', '')
    dump_root.mkdir(exist_ok=False, parents=True)
    print("Dumping to:", str(dump_root))

    args = []
    for worker_ix in range(n_workers):
        start_ix = split_ixs[worker_ix]
        stop_ix = split_ixs[worker_ix + 1]
        worker = PairingCalcWorker(X=X.copy(),
                                   uids=[s.uid for s in segments],
                                   aligner=PlanarAligner(kind=align_kind),
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
    metric_dfs = construct_dataframes_from_pkls(dump_root, segments, delete_pkls_when_done=True)
    print("Done.")

    return metric_dfs


def run__make_and_save():
    force = 0
    datasets = ['TP_RS', 'TP_RJ']
    datasets = datasets[:1]
    #datasets = get_hatso_datasets(task='TP')
    upto = None # DataConfig.TRIALS
    bin_sizes = [None]
    for align_kind in ['affine', 'ortho']:
        for bin_sz in bin_sizes:
            for dataset in datasets:
                data_cfg = Config.from_default().data
                data_cfg.trials.name = dataset
                if bin_sz is not None:
                    data_cfg.trials.bin_sz = bin_sz
                data_cfg.pairing.align_kind = align_kind
                make_and_save(data_cfg, force=force, upto=upto)


if __name__ == "__main__":
    #construct_dataframe_from_pkls("/Users/davidu/geometric-neurons/resources/data/tmp/kinX-ortho-25357881-20240206-124339")
    run__make_and_save()
    pass
