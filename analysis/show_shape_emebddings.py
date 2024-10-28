import itertools
from common.utils.procrustes import PlanarAlign
from common.utils.polytools import uniform_resample
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
    lm = 6

    pop_names = sorted(set(proj[0] for proj in projs))
    methods = sorted(set(proj[1] for proj in projs))
    colors = plotting.get_nice_colors(ixs_of_shape)

    axs = plotting.named_subplots(cols=pop_names, rows=methods, eq=True)
    plotting.set_outter_labels(axs, t=pop_names, y=methods)
    for (pop_name, method, pc_vecs, (silhouette, gmm_likelihood)) in projs:
        ax = axs[(method, pop_name)]
        for shape, seg_ixs in ixs_of_shape.items():
            color = colors[shape]
            if density_type == 'ellipse':
                plotting.plot_2d_gaussian_ellipse(pc_vecs[seg_ixs], ax=ax, edgecolor=color, facecolor='none', lw=1)
            elif density_type == 'kde':
                sns.kdeplot(x=pc_vecs[seg_ixs, 0], y=pc_vecs[seg_ixs, 1], ax=ax, color=color, fill=False)
            ax.scatter(*pc_vecs[seg_ixs].T, alpha=.5, label=shape, color=color)
        ax.set_title(f"{pop_name}\nSilhouette={silhouette:2.2f}, Balanced Accuracy={gmm_likelihood:2.2f}")
        ax.set_xlim([-lm, lm])
        ax.set_ylim([-lm, lm])
        ax.legend()


def draw_shape_embeddings(model_file):
    from common.utils.shapes_specs import ShapeSpec

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    monkey = cfg.data.trials.name.split('_')[-1]

    shape_specs = cv_results_mgr.get_chosen_shape_specs(False)

    shape_specs = {f'Shape{i+1}': s for i, s in enumerate(shape_specs)}
    proj_kws = cv_results_mgr.get_chosen_projection_specs()
    methods = proj_kws['method']
    n = proj_kws['n_per_shape'][monkey]
    rectify_by = proj_kws['rectify_by']

    #shape_specs = segment_shapes_tools.get_chosen_shape_specs()
    print("Projecting Shapes:", shape_specs)
    print("Projection Params:", proj_kws)

    data_mgr = DataMgr(cfg.data, persist=True)
    input_vecs, _ = data_mgr.get_inputs()
    segments = data_mgr.load_segments()
    neural_pop = NeuralPopulation.from_model(model_file)

    conics, scores_df = data_mgr.load_fitted_conics()
    valid_ixs = segment_shapes_tools.get_valid_conic_ixs(scores_df)
    seg_groups = segment_processing.digitize_segments(segments, n=n, by=rectify_by, include_ixs=valid_ixs)

    ixs_of_shape = segment_shapes_tools.match_conics_to_specs(
        conics=conics, segment_labels=seg_groups, shape_specs=shape_specs, n=n)

    colors = plotting.get_nice_colors(ixs_of_shape)

    # -----

    projs = segment_processing.compute_projections(model, input_vecs, neural_pop, groups=ixs_of_shape, n_pcs=2,
                                                   pop_names=[NEURAL_POP.MAJORITY, NEURAL_POP.MINORITY], methods=methods)

    n_samples_to_draw = 3
    also_draw_segment = True
    for i_sample in range(n_samples_to_draw):
        axs = plotting.named_subplots(cols=ixs_of_shape, eq=True)
        for name, ixs in ixs_of_shape.items():
            pts = conics[ixs[i_sample]].get_standardized().parametric_pts()
            pts /= np.std(pts, axis=0)
            axs[name].plot(*pts.T, color=colors[name])
            if also_draw_segment:
                seg_pts, _ = uniform_resample(segments[ixs[i_sample]].kin.X, len(pts))
                seg_pts, _ = PlanarAlign('ortho')(pts, seg_pts)
                axs[name].plot(*seg_pts.T, color='k')

            axs[name].set_title(name)
            axs[name].axis('off')
        plt.suptitle(data_mgr.cfg.trials.name + " - EXAMPLE SHAPES")

    draw_projections(projs, ixs_of_shape, density_type='kde')
    plt.suptitle(data_mgr.monkey)

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
