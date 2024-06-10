import matplotlib.pyplot as plt
import numpy as np
from data_manager import DataMgr
import cv_results_mgr
import pandas as pd
from neural_population import NeuralPopulation, NEURAL_POP
from common.utils import plotting
import seaborn as sns
from analysis import segment_shapes_tools
from analysis import segment_processing


def draw_projections(projs: list[tuple], ixs_of_shape: dict, density_type: str = 'kde'):
    pop_names = sorted(set(proj[0] for proj in projs))
    methods = sorted(set(proj[1] for proj in projs))
    colors = plotting.get_nice_colors(ixs_of_shape)

    axs = plotting.named_subplots(cols=pop_names, rows=methods, eq=True)
    plotting.set_outter_labels(axs, t=pop_names, y=methods)
    for (pop_name, method, pc_vecs, score) in projs:
        ax = axs[(method, pop_name)]
        for shape, seg_ixs in ixs_of_shape.items():
            color = colors[shape]
            if density_type == 'ellipse':
                plotting.plot_2d_gaussian_ellipse(pc_vecs[seg_ixs], ax=ax, edgecolor=color, facecolor='none', lw=1)
            elif density_type == 'kde':
                sns.kdeplot(x=pc_vecs[seg_ixs, 0], y=pc_vecs[seg_ixs, 1], ax=ax, color=color, fill=False)
            ax.scatter(*pc_vecs[seg_ixs].T, alpha=.5, label=shape, color=color)
        ax.legend()


def draw_shape_embeddings(model_file, n: int = 15, n_to_draw: int = None):

    shape_specs = segment_shapes_tools.get_chosen_shape_specs()
    print("Using Specs:")
    print(shape_specs)

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data, persist=True)
    input_vecs, _ = data_mgr.get_inputs()
    segments = data_mgr.load_segments()
    neural_pop = NeuralPopulation.from_model(model_file)

    conics, scores_df = data_mgr.load_fitted_conics()
    valid_ixs = segment_shapes_tools.get_valid_conic_ixs(scores_df)
    seg_groups = segment_processing.digitize_segments(segments, n=n, by='EuSpd', include_ixs=valid_ixs)

    ixs_of_shape = segment_shapes_tools.match_conics_to_specs(
        conics=conics, segment_labels=seg_groups, shape_specs=shape_specs, n=n)

    projs = segment_processing.compute_projections(model, input_vecs, neural_pop, groups=ixs_of_shape, n_pcs=2,
                                                   pop_names=[NEURAL_POP.MAJORITY, NEURAL_POP.MINORITY])

    colors = plotting.get_nice_colors(ixs_of_shape)

    if n_to_draw is not None:
        for shape_name, ixs in ixs_of_shape.items():
            if len(ixs) < n_to_draw:
                continue
            x = input_vecs[ixs]
            dists = np.sum((x - x.mean(axis=0)) ** 2, axis=1)
            ixs_of_shape[shape_name] = [ixs[i] for i in np.argsort(dists)[:n_to_draw]]

    axs = plotting.named_subplots(cols=ixs_of_shape, eq=True)
    for name, ixs in ixs_of_shape.items():
        pts = conics[ixs[2]].get_standardized().parametric_pts()
        pts /= np.std(pts, axis=0)
        axs[name].plot(*pts.T, color=colors[name])
        axs[name].set_title(name)
    plt.suptitle(data_mgr.cfg.trials.name + " - EXAMPLE SHAPES")

    draw_projections(projs, ixs_of_shape, density_type='kde')
    plt.suptitle(data_mgr.cfg.trials.name + " - PROJECTIONS")

    data = {'Speed': [], 'Acceleration': [], 'Shape': []}
    for shape, seg_ixs in ixs_of_shape.items():
        data['Speed'] += [segments[i].kin['EuSpd'].mean() for i in seg_ixs]
        data['Acceleration'] += [segments[i].kin['EuAcc'].mean() for i in seg_ixs]
        data['Shape'] += [shape] * len(seg_ixs)
    data = pd.DataFrame(data)

    params = [col for col in data.columns if col != 'Shape']
    axs = plotting.named_subplots(rows=params)
    plt.suptitle(data_mgr.cfg.trials.name + " - Euclidean Variables")
    for param in params:
        sns.kdeplot(data=data, x=param, ax=axs[param], hue='Shape', common_norm=False, palette=colors, fill=True)


if __name__ == "__main__":
    for monkey, model_file in cv_results_mgr.get_chosen_model_per_monkey().items():
        draw_shape_embeddings(model_file)
    plt.show()
