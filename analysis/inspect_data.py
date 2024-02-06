import numpy as np
import pandas as pd
from common.utils import polytools
from analysis.config import Config
from geometric_encoding.embedding import LinearEmbedder
from analysis.data_manager import DataMgr
from common.pairwise.triplet_train import triplet_train
from common.pairwise.sameness import SamenessData
from common.pairwise import embedding_eval
from motorneural.data import Segment
from common.utils.procrustes import Procrustes
from common.utils import linalg
from common.utils import dictools
from common.utils import sigproc
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from screeninfo import get_monitors

DEFAULT_STYLE = {'marker': 'o', 'linestyle': '-', 'alpha': .75}
STYLES = {
    'anchor': {'color': 'k'},
    'pos': {'color': 'c'},
    'neg': {'color': 'r'}
}
for k in STYLES:
    STYLES[k] = dictools.modify_dict(base_dict=DEFAULT_STYLE, copy=False, update_dict=STYLES[k])

def get_screen_size_of_figure(fig):
    """ Get the size of the screen where the figure is displayed. """

    fig_center = fig.get_tightbbox()._bbox.get_points().mean(axis=0)

    for m in get_monitors():
        if m.x <= fig_center[0] < m.x + m.width and m.y <= fig_center[1] < m.y + m.height:
            return m.width, m.height
    return None

def resize_figure(fig, scale=0.9):
    """ Resize the figure based on the screen size and figure's aspect ratio. """
    screen_width, screen_height = get_screen_size_of_figure(fig)

    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    aspect_ratio = bbox.width / bbox.height

    screen_aspect_ratio = screen_width / screen_height

    if aspect_ratio > screen_aspect_ratio:
        fig_width = screen_width * scale / fig.dpi
        fig_height = fig_width / aspect_ratio
    else:
        fig_height = screen_height * scale / fig.dpi
        fig_width = fig_height * aspect_ratio

    fig.set_size_inches(fig_width, fig_height, forward=True)


def draw_segment_in_trial(data, segment: Segment, trial_style='y.-', segment_style='m.-'):
    trial = data[segment.trial_ix]
    plt.plot(trial.kin.X[:, 0], trial.kin.X[:, 1], trial_style)
    plt.plot(segment.kin.X[:, 0], segment.kin.X[:, 1], segment_style)


def show_extreme_proc_segment_pairs(cfg: Config = None):
    """
    For each procrustes kind, show segment pairs of minimal & maximal distance (from random subsample of data)
    """

    n_anchors = 500
    seed = 1

    if not cfg:
        cfg = Config.from_default()

    sameness, pairs, segments = DataMgr(cfg.data).load_sameness()
    sameness.init_triplet_sampling()
    print(sameness.triplet_summary_string())

    anchors, positives, negatives = sameness.sample_triplet_items(n_anchors, rand_seed=seed)
    pairs, _ = embedding_eval.make_triplet_pairs(anchors, positives, negatives)
    kinX = [s.kin['X'] for s in segments]

    for kind in Procrustes.KINDS:
        procrustes = Procrustes(kind=kind)

        dists = np.array([procrustes(kinX[i], kinX[j])[0] for (i, j) in pairs]).squeeze()

        ixs = {
            'max': np.argmax(dists),
            'min': np.argmin(dists),
        }

        _, axs = plt.subplots(ncols=len(ixs))
        for k, (name, ix) in enumerate(ixs.items()):
            plt.sca(axs[k])
            i, j = pairs[ix]
            XXj = procrustes(kinX[i], kinX[j])[2]
            plt.plot(*kinX[i].T, 'bo-', label=f"Seg {i}")
            plt.plot(*XXj.T, 'ro-', label=f"Seg {j}")
            plt.title(f"{name} {kind}")
            plt.legend()

    plt.show()


def compare_rigid_to_speed_dist(cfg: Config = None):
    """
    Compare Rigid procrustes distance to speed distance (=difference of average speeds)
    Meant for visually validating rigid procrustes and generally checking sanity,
    as for similar shapes the rigid distance should correlate with speed distance
    """

    n_anchors = 500
    seed = 1

    if not cfg:
        cfg = Config.from_default()

    sameness, pairs, segments = DataMgr(cfg.data).load_sameness()
    sameness.init_triplet_sampling()
    print(sameness.triplet_summary_string())

    anchors, positives, negatives = sameness.sample_triplet_items(n_anchors, rand_seed=seed)
    pairs, is_same = embedding_eval.make_triplet_pairs(anchors, positives, negatives)

    spd = np.array([s.kin['EuSpd'] for s in segments]).mean(axis=1, keepdims=True)
    spd_dists = embedding_eval.pairs_dists(x=spd, pairs=pairs).squeeze()

    rigid_procrustes = Procrustes(kind='offset')
    X = DataMgr.make_pairing_X('kin.EuSpd', segments)
    kinX = X # [s.kin['X'] for s in segments]
    rigid_dists = np.array([rigid_procrustes(X[i], X[j])[0] for (i, j) in pairs]).squeeze()

    normalized_rigid_dists = (rigid_dists - np.mean(rigid_dists)) / np.std(rigid_dists)
    normalized_spd_dists = (spd_dists - np.mean(spd_dists)) / np.std(spd_dists)

    ixs = {
        'rigid>speed': np.argmax(normalized_rigid_dists - normalized_spd_dists),
        'rigid<speed': np.argmin(normalized_rigid_dists - normalized_spd_dists),
        'rigid=speed': np.argmin(np.abs(normalized_rigid_dists - normalized_spd_dists)),
        'min rigid': np.argmin(normalized_rigid_dists)
    }
    colors = 'red', 'limeGreen', 'gold', 'black'

    rigid_dists = normalized_rigid_dists
    spd_dists = normalized_spd_dists

    plt.figure()
    plt.scatter(spd_dists, rigid_dists, alpha=.5)
    for k, (name, ix) in enumerate(ixs.items()):
        plt.plot(spd_dists[ix], rigid_dists[ix], '^', color=colors[k], label=name)
    plt.legend()
    plt.xlabel("Speed distances")
    plt.ylabel("Rigid distances")

    _, axs = plt.subplots(ncols=len(ixs))
    for k, (name, ix) in enumerate(ixs.items()):
        plt.sca(axs[k])
        i, j = pairs[ix]
        XXj = rigid_procrustes(kinX[i], kinX[j])[2]
        plt.plot(*kinX[i].T, 'bo-', label=f"Seg {i} Arclen={polytools.total_arclen(kinX[i]):2.3f}")
        plt.plot(*XXj.T, 'ro-', label=f"Seg {j} Arclen={polytools.total_arclen(kinX[j]):2.3f}")
        plt.title(f"{name}\nSpeed Dist={spd_dists[ix]:2.3f}\nRigid Dist={rigid_dists[ix]:2.3f}")
        plt.legend()

    plt.show()


def show_triplets(cfg: Config = None):

    if not cfg:
        cfg = Config.from_default()
    cfg.data.pairing.metric = 'rigid'

    data_mgr = DataMgr(cfg.data)
    data = data_mgr.load_base_data()
    sameness, pairs, segments = data_mgr.load_sameness()
    sameness.init_triplet_sampling()
    print(sameness.triplet_summary_string())

    anchors, positives, negatives = sameness.sample_triplet_items(500, rand_seed=0)
    pairs, is_same = embedding_eval.make_triplet_pairs(anchors, positives, negatives)

    native_procrustes = Procrustes(kind=cfg.data.pairing.metric)

    def _draw_segments(aXX, pXX, nXX, labels=None):
        if not labels:
            labels = {'a': None, 'p': None, 'n': None}
        plt.plot(aXX[:, 0], aXX[:, 1], **STYLES['anchor'], label=labels['a'])
        plt.plot(pXX[:, 0], pXX[:, 1], **STYLES['pos'], label=labels['p'])
        plt.plot(nXX[:, 0], nXX[:, 1], **STYLES['neg'], label=labels['n'])

    spd = np.array([s.kin['EuSpd'] for s in segments]).mean(axis=1, keepdims=True)
    spd_dists = embedding_eval.pairs_dists(x=spd, pairs=pairs)

    rigid_procrustes = Procrustes(kind='rigid')
    kinX = [s.kin['X'] for s in segments]
    rigid_dists = np.array([rigid_procrustes(kinX[i], kinX[j])[0] for (i, j) in pairs])

    plt.figure()
    plt.plot(np.array(spd_dists).squeeze(), np.array(rigid_dists).squeeze(), '.')
    plt.xlabel("Span")
    plt.ylabel("Rigid")
    plt.show()

    # X = pairing_Xs[anchors]
    # P = pairing_Xs[positives]
    # N = pairing_Xs[negatives]
    #
    #
    # p_dist, _, _ = procrustes(X, P)
    # n_dist, _, _ = procrustes(X, N)



    loss_func = embedding_eval.EmbeddingLoss(kind="triplet", margin=cfg.training.loss_margin)

    print("kinSpd_avg:")
    print(embedding_eval.evaluate_embedded_vecs(
        embedded_vecs=kinSpd_avg, pairs=pairs, is_same=is_same, loss_func=loss_func))

    print("kinX_span:")
    print(embedding_eval.evaluate_embedded_vecs(
        embedded_vecs=kinX_span, pairs=pairs, is_same=is_same, loss_func=loss_func))

    sameness_eval_result = sameness.make_evaluator(loss_margin=cfg.training.loss_margin).evaluate()
    print(native_pairing_metric, ":")
    print(sameness_eval_result)

    #speed_df = pd.DataFrame({'dists': dists, 'is_same': is_same})

    plt.figure()
    plt.plot(np.array(kinX_span).squeeze(), np.array(kinSpd_avg).squeeze(), '.')
    plt.xlabel("Span")
    plt.ylabel("Speed")
    plt.show()



    pairing_metrics = ['ortho', 'rigid', 'offset']
    if native_pairing_metric == 'affine':
        pairing_metrics[0] = 'affine'

    axs = None
    axs2 = None
    for (a, p, n) in zip(anchors, positives, negatives):

        X = pairing_Xs[a]
        P = pairing_Xs[p]
        N = pairing_Xs[n]

        p_dist, _, P_pairing_transformed = procrustes(X, P)
        n_dist, _, N_pairing_transformed = procrustes(X, N)

        p_dist_pcntl = np.mean(dists < p_dist)
        n_dist_pcntl = np.mean(dists < n_dist)

        if n_dist_pcntl < .4:
            continue

        if axs is None:
            _, axs = plt.subplots(ncols=len(pairing_metrics), nrows=2)
            axs2 = axs[1]
            axs = axs[0]
        else:
            for ax in axs.flatten():
                ax.cla()
            for ax in axs2.flatten():
                ax.cla()

        plt.sca(axs2[0])
        draw_segment_in_trial(data, segments[a], segment_style='k.-')
        plt.sca(axs2[1])
        draw_segment_in_trial(data, segments[p], segment_style='c.-')
        plt.sca(axs2[2])
        draw_segment_in_trial(data, segments[n], segment_style='r.-')
        plt.draw()

        labels = {
            'a': f'Anchor [Seg{a}]',
            'p': f'Pos [Seg{p}] Dist={p_dist_pcntl * 100:5.2f}',
            'n': f'Neg [Seg{n}] Dist={n_dist_pcntl * 100:5.2f}',
        }
        plt.sca(axs[0])
        _draw_segments(X, P_pairing_transformed, N_pairing_transformed, labels=labels)
        plt.legend()
        plt.title(native_pairing_metric + " [PairingProc]")

        draw_count = 1
        for pairing_metric in pairing_metrics:
            if pairing_metric != native_pairing_metric:
                _, _, P_transformed = Procrustes(kind=pairing_metric)(X, P)
                _, _, N_transformed = Procrustes(kind=pairing_metric)(X, N)
                plt.sca(axs[draw_count])
                _draw_segments(X, P_transformed, N_transformed)
                plt.title(pairing_metric)
                draw_count += 1

        axs[0].set_ylabel('Alignment')
        axs2[0].set_ylabel('Segments')
        plt.draw()
        plt.show(block=False)
        while not plt.waitforbuttonpress():
            pass


def to_string(v, f=3.2):
    def _float_fmt():
        if isinstance(f, int):
            return f'3.{f}f'
        elif isinstance(f, float):
            return str(f) + 'f'
        return f
    if isinstance(v, (list, tuple, np.ndarray)):
        return str(type(v)([to_string(u) for u in v]))
    elif isinstance(v, dict):
        return {key: to_string(val) for key, val in v.items()}
    if isinstance(v, float):
        if int(v) == v:
            v = int(v)
        else:
            v = f'{v:{_float_fmt()}}'
    if isinstance(v, int):
        v = f'{v:d}'
    return v


def stats(a):
    pcntls = percentiles_dict(a, [10, 90])
    ret = {"max": np.max(a), "min": np.min(a), "avg": np.mean(a), "var": np.var(a),
           "med": np.median(a), **pcntls}
    ret["std"] = np.sqrt(ret["var"])
    ret["min/max"] = ret["min"] / ret["max"]
    return ret


def percentiles_dict(a, ps, named: bool = True) -> dict:
    d = dict(zip(ps, np.percentile(a, ps)))
    if named:
        d = {f'p{k}': v for k, v in d.items()}
    return d


def check_sampling_times(cfg: Config = None):
    if not cfg:
        cfg = Config.from_default()

    data = DataMgr(cfg.data).load_base_data()
    for ix in np.linspace(0, len(data) - 1, 5):
        trial = data[int(ix)]
        kin_t = trial.kin.t
        neural_t = trial.neural.t
        assert len(kin_t) == len(neural_t)
        print(f"Trial {trial.ix}:")
        print("Durations: Kin={Kin}, Neural={Neural}, Diff={Diff}, RelDiff={RelDiff}".format(**to_string({
            "Kin": trial.kin.duration, "Neural": trial.neural.duration,
            "Diff": trial.kin.duration - trial.neural.duration,
            "RelDiff": (trial.kin.duration - trial.neural.duration) / trial.neural.duration})))
        deltas = kin_t - neural_t
        print("Kin-Neural offset: Med={med} Min={min} Max={max} p10={p10} p90={p90}".format(**to_string(stats(
            deltas))))

        print("DT stats:")
        print("Kin:   ", to_string(stats(np.diff(kin_t))))
        print("Neural:", to_string(stats(np.diff(neural_t))))


def show_neural_by_kins(cfg: Config = None, kin_names: str = 'EuAccAng', nbins: int = 20, norm: bool = True):

    if not cfg:
        cfg = Config.from_default()
    data = DataMgr(cfg.data).load_base_data()

    neurals = np.concatenate([tr.neural[:] for tr in data])
    if isinstance(kin_names, str):
        kin_names = kin_names.split(',')

    fig, axs = plt.subplots(ncols=2, nrows=len(kin_names))
    fig.tight_layout()

    for kin_name in kin_names:
        kin = np.concatenate([tr.kin[kin_name] for tr in data])
        if kin_name.lower().endswith('ang'):
            kin = kin % 360
            kin_range = 0, 360
        else:
            kin_range = np.percentile(kin, [1, 99])
        kin_bin_edges = np.linspace(kin_range[0], kin_range[1], nbins + 1)
        kin_bins = np.digitize(kin, kin_bin_edges)

        for shuff in (False, True):
            if shuff:
                kin_bins_ = np.random.default_rng(0).permutation(kin_bins)
            else:
                kin_bins_ = kin_bins
            neural_per_bin = [neurals[kin_bins_ == kin_bin].mean(axis=0)
                              for kin_bin in range(1, nbins + 1)]
            neural_per_bin = np.array(neural_per_bin, dtype=float)

            sd, mu = sigproc.scales_and_offsets(neural_per_bin, axis=0, kind='std')
            top_p = 10
            mask = sd.squeeze() > np.percentile(sd.squeeze(), 100 - top_p)
            if not shuff:
                print(kin_name, f"top {top_p}% variance neurons:", list(np.nonzero(mask)[0]))

            if norm:
                neural_per_bin -= mu
                neural_per_bin /= sd
                neural_per_bin[:, ~mask] = 0

            plt.sca(axs[kin_names.index(kin_name)][int(shuff)])
            plt.imshow(neural_per_bin, cmap='hot')
            plt.xlabel('Neuron')
            plt.title(kin_name + (' SHUFF' if shuff else ''))

    resize_figure(fig)
    plt.show()


def show_kin_sorted_trials(cfg: Config = None, norm: bool = True):
    if not cfg:
        cfg = Config.from_default()

    data = DataMgr(cfg.data).load_base_data()
    win_r = 50
    summ = np.zeros((2 * win_r + 1, data.num_neurons), float)
    countt = np.zeros((2 * win_r + 1, data.num_neurons), int)
    for i, tr in enumerate(data):
        c = tr['max_spd']
        i_start = max(0, c - win_r)
        i_stop = min(len(tr), c + win_r + 1)
        neural = tr.neural[:][i_start: i_stop]
        offset = win_r - (c - i_start)
        summ[offset: offset + neural.shape[0], :] += neural
        countt[offset: offset + neural.shape[0], :] += 1

    m = summ / np.maximum(countt, 1)
    if norm:
        m = sigproc.normalize(m, axis=0)

    plt.imshow(m, cmap='gray')
    lag_in_bins = +data.lag / data.bin_sz

    plt.plot([0, m.shape[1] - 1], [win_r + lag_in_bins, win_r + lag_in_bins], 'r')
    plt.show()

    segments = DataMgr(cfg.data).load_segments()

    avg_speeds = [tr.kin.EuSpd.mean() for tr in segments]
    si = np.argsort(avg_speeds)
    avg_neurals = np.stack([segments[trial_ix].neural[:].mean(axis=0) for trial_ix in si], axis=1)

    plt.figure()
    plt.imshow(avg_neurals)
    plt.figure()
    plt.plot(avg_neurals.mean(axis=0))
    plt.show()

def _pca_sandbox():

    n_samples = 5_000
    n_neurons = 100
    n_dims = 50

    m = np.random.default_rng(0).random(size=(n_neurons, n_dims))
    #m[:n_dims] = np.eye(n_dims)
    underlying_X = np.random.default_rng(0).standard_normal(size=(n_dims, n_samples))
    X = (m @ underlying_X).T

    X -= X.mean(axis=0)
    X /= np.std(X, axis=0)

    from sklearn.decomposition import PCA
    v = PCA(n_components=n_neurons).fit(X).explained_variance_ratio_.cumsum()
    i = np.argmin(np.diff(v))
    plt.plot(v, '.-')
    plt.plot(i, v[i], 'ro')
    plt.text(i, v[i] - .01, f"PC {i + 1}", color='r')
    plt.xlabel('PCs')
    plt.ylabel('Explain Variance')
    plt.title(f"n_dims={n_dims}, n_neurons={n_neurons}")
    plt.show()

#_pca_sandbox()
#show_neural_by_kins(kin_names=['EuSpdAng','EuAccAng', 'EuSpd'], nbins=12)
#show_kin_sorted_trials()
#check_sampling_times()
#show_triplets()
#show_extreme_proc_segment_pairs()
compare_rigid_to_speed_dist()