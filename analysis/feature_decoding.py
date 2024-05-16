import matplotlib.pyplot as plt
import numpy as np
from data_manager import DataMgr
import cv_results_mgr
from common.utils import dlutils
from common.utils.scoring import BootstrapEvaluator
from sklearn.model_selection import cross_val_predict, cross_validate
from common.utils import sigproc
from common.utils import mltools
from show_embeddings import plot_binned_stats, _new_subplots
from analysis.neural_population import NEURAL_POP, NeuralPopulation
from itertools import product
from common.utils import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


def decode_features_from_neural(model_file):

    # ------------
    # PARAMS:

    seed = 1

    kin_vars = ['AfSpd', 'EuSpd', 'EuAcc']
    evaluator = BootstrapEvaluator(n_shuffs=100, seed=1)

    # exclude outlier kinematic values:
    inliers = stats.Inliers('percentiles', [0, .75])

    # normalization:
    norm_kind = 'mad'
    normalize_neural = True
    normalize_true_kin = True

    # balancing:
    balance_bins = stats.BinSpec(10, 'u')
    balance = True

    # crossval:
    calibrate = True

    # spec for regressor calibration & band-plot:
    bins = stats.BinSpec(10, 'u')
    stat = 'med'

    # drawing:
    show_points = True
    plot_kws = dict(bins=bins, loc=stat, band='error', lw=1, marker='.', idline=True)

    # sanity checks:
    shuffle_neural = False

    # ------------
    # PREP:

    # load model and config
    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    cfg = cfg.get_as_eval_config()
    data_mgr = DataMgr(cfg.data, persist=True)

    # get reduced kinematics per segment
    reduced_kinematics = data_mgr.get_reduced_kinematics(names=kin_vars)

    # get flat neural activity per segment, and the population label of each neuron
    neural_vecs, neural_meta = data_mgr.get_inputs()
    pop = NeuralPopulation.from_model(model_file)
    pop.draw()

    if shuffle_neural:
        print("\n\n !! SHUFFLING NEURAL !! \n\n")
        neural_vecs = neural_vecs[np.random.default_rng(1).permutation(len(neural_vecs))]

    def _get_neural_subspace(embed: bool, pop_label: NEURAL_POP, mask: np.ndarray = None):
        X = neural_vecs.copy() if mask is None else neural_vecs[mask].copy()
        match pop_label:
            case NEURAL_POP.MINORITY: X[:, pop.inputs_mask(NEURAL_POP.MAJORITY)] = 0
            case NEURAL_POP.MAJORITY: X[:, pop.inputs_mask(NEURAL_POP.MINORITY)] = 0
            case NEURAL_POP.FULL: pass
            case _: raise ValueError("Unknown population type " + population)
        if embed:
            X = dlutils.safe_predict(model, X)
        return X

    # ------------
    # CORE:

    axs = {}
    results = {}
    for kin_name, kin_values in reduced_kinematics.items():
        # for each kinematic variable..

        segments_mask = inliers.is_inlier(kin_values)

        y_true = kin_values[segments_mask].copy()
        if normalize_true_kin:
            y_true = sigproc.normalize(y_true, kind=norm_kind)

        sample_weight = None if not balance else mltools.calc_balancing_weights(y_true, bins=balance_bins)

        populations = [NEURAL_POP.FULL, NEURAL_POP.MINORITY]
        for embed, population in product([False, True], populations):
            # for each neural subspace..

            X = _get_neural_subspace(embed, population, segments_mask)
            if normalize_neural:
                X = sigproc.normalize(X, kind=norm_kind, axis=0)

            assert np.isfinite(y_true).all() and np.isfinite(X).all()

            iter_name = f"{kin_name} from {population} {'Embedded' if embed else 'NonEmbedded'}"
            print(iter_name, "...")

            regressor = LinearRegression()
            if calibrate:
                regressor = mltools.CenterCalibRegressor(base_estimator=regressor, bins=bins, stat=stat)

            y_pred = cross_val_predict(regressor, X=X, y=y_true, cv=KFold(random_state=seed, shuffle=True),
                                       n_jobs=-1, fit_params={'sample_weight': sample_weight})

            result = evaluator.evaluate(y_true, y_pred, sample_weight=sample_weight)
            results[iter_name] = result

            print("  ZSCORES:", {metric: result[metric]['zscore'] for metric in result})
            print("   VALUES:", {metric: result[metric]['value'] for metric in result})

            ax = axs.get(kin_name, None)
            if ax is None:
                _, ax = _new_subplots()
                axs[kin_name] = ax
                if show_points:
                    alpha = .5 if sample_weight is None else sample_weight / sample_weight.max()
                    ax.scatter(y_true, y_pred, alpha=alpha, color='pink')

            ax, p = plot_binned_stats(x=y_true, y=y_pred, ax=ax, **plot_kws)

            disp_metric = 'medae'
            disp_metric_kind = 'value'
            label = iter_name + f' {disp_metric}={result[disp_metric][disp_metric_kind]:2.4f}'

            p[0].set_label(label)
            p[1].set_label(None)

    for name, ax in axs.items():
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(name)
        ax.get_figure().canvas.manager.set_window_title(name)
        ax.legend()
    plt.show()


if __name__ == "__main__":
    file = "/Users/davidu/geometric-neurons/outputs/models/TP_RS bin10 lag100 dur200 affine-kinX-nmahal f70c5c.Fold0.pth"
    #draw_trajectories_grouped_by_embedded_dist(file)
    decode_features_from_neural(file)
