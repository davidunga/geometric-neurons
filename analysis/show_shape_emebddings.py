import itertools
from common.utils.procrustes import PlanarAlign, PlanarAlignTo, AlignedDist
from common.utils.polytools import uniform_resample
import matplotlib.pyplot as plt
import numpy as np
from data_manager import DataMgr
import cv_results_mgr
import pandas as pd
from neural_population import NeuralPopulation, NEURAL_POP
from common.utils import plotting
import seaborn as sns
from common.utils import stats
from analysis import segment_shapes_tools
from analysis import segment_processing
from scipy.spatial.distance import cdist
from common.utils.clustering import ClusterEvaluator, k_medoids, visualize_with_projection


def draw_projections(projs: list[tuple], ixs_of_shape: dict, density_type: str = 'kde'):
    lm = None # 6

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
                plotting.plot_2d_gaussian_ellipse(pc_vecs[seg_ixs], ax=ax, edgecolor=color, facecolor='none', lw=1,
                                                  support_fraction=1, n_std=2)
            elif density_type == 'kde':
                sns.kdeplot(x=pc_vecs[seg_ixs, 0], y=pc_vecs[seg_ixs, 1], ax=ax, color=color, fill=False)
            ax.scatter(*pc_vecs[seg_ixs].T, alpha=.5, label=shape, color=color)
        ax.set_title(f"{pop_name}\nSilhouette={silhouette:2.2f}, Balanced Accuracy={gmm_likelihood:2.2f}")
        if lm is not None:
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

    # ----
    n_shape_to_take = 3
    max_clusters_to_make = 3 + n_shape_to_take
    cluster_sz = 20
    with_rectify = rectify_by != 'none'

    pairs_df = data_mgr.load_pairing()
    n_segs = pairs_df.attrs['num_segments']

    dists_dict = dict(zip(zip(pairs_df['seg1'], pairs_df['seg2']), pairs_df['dist'].to_numpy()))
    pairs_df = pairs_df[pairs_df['isSame']]


    # -----
    from costume_dataframes.pairs_df_funcs import to_sparse_matrix, to_dist_matrix

    def report_zscores(zscores: dict[str, stats.ZScoreRes]):
        for k, zs in zscores.items():
            print(f"{k:30s}: {zs.val:2.4f}, bl={zs.bl_loc:2.4f}, zscore={zs.z:2.4f}")

    affine_dist_mtx = to_dist_matrix(pairs_df, 'dist')
    assert affine_dist_mtx.shape == (n_segs, n_segs)

    affine_cluster_labels = k_medoids(affine_dist_mtx, k=3, cluster_sz=20, precomp=True)

    print("Quality of affine clusters:")

    report_zscores(ClusterEvaluator(bootstrap_itrs=500).evaluate(labels_pred=affine_cluster_labels, d=affine_dist_mtx))


    # -----
    #
    # clusters = []
    # for _ in range(max_clusters_to_make):
    #     print(f"Clustering {len(clusters)}...")
    #     if with_rectify:
    #         seg_rectify_labels = segment_processing.digitize_segments(segments, n=cluster_sz, by=rectify_by)
    #         nearest_seg = {seg_ix: [None for _ in range(cluster_sz)] for seg_ix in range(n_segs)}
    #         nearest_dist = {seg_ix: [float('inf') for _ in range(cluster_sz)] for seg_ix in range(n_segs)}
    #
    #         def _update_dists(seg_, other_seg_, dist_):
    #             other_lbl = seg_rectify_labels[other_seg_] - 1
    #             assert other_lbl >= 0
    #             assert nearest_dist[seg_][other_lbl] > dist_
    #             nearest_dist[seg_][other_lbl] = dist_
    #             nearest_seg[seg_][other_lbl] = other_seg_
    #
    #         for seg1, seg2, dist in pairs_df[['seg1', 'seg2', 'dist']].itertuples(index=False):
    #             lbl1 = seg_rectify_labels[seg1] - 1
    #             lbl2 = seg_rectify_labels[seg2] - 1
    #             if min(lbl1, lbl2) < 0:
    #                 continue
    #             if nearest_dist[seg1][lbl2] > dist: _update_dists(seg1, seg2, dist)
    #             if nearest_dist[seg2][lbl1] > dist: _update_dists(seg2, seg1, dist)
    #         cluster_dists = {seg_ix: np.max(d) for seg_ix, d in nearest_dist.items()}
    #     else:
    #         dists = {seg_ix: [] for seg_ix in range(n_segs)}
    #         for row in pairs_df.itertuples(index=False):
    #             dists[row.seg1].append(row.dist)
    #             dists[row.seg2].append(row.dist)
    #         cluster_dists = {seg_ix: sorted(d)[cluster_sz - 1]
    #                          for seg_ix, d in dists.items() if len(d) >= cluster_sz}
    #
    #     if not cluster_dists:
    #         break
    #
    #     seg_ix = list(cluster_dists.keys())[np.argmin(list(cluster_dists.values()))]
    #
    #     if with_rectify:
    #         cluster_seg_ixs = [seg_ix] + nearest_seg[seg_ix]
    #     else:
    #         mask = ((pairs_df['seg1'] == seg_ix) | (pairs_df['seg2'] == seg_ix)) \
    #                & (pairs_df['dist'] < cluster_dists[seg_ix])
    #         cluster_seg_ixs = sorted(set(pairs_df.loc[mask, ['seg1', 'seg2']].to_numpy().ravel()))
    #
    #     if any(ix is None for ix in cluster_seg_ixs):
    #         break
    #
    #     clusters.append(cluster_seg_ixs)
    #     pairs_df = pairs_df.loc[~(pairs_df['seg1'].isin(cluster_seg_ixs) | pairs_df['seg2'].isin(cluster_seg_ixs))]
    #
    # for i, c in enumerate(clusters):
    #     print(i, c)
    #
    # assert len(set(itertools.chain(*clusters))) == len(list(itertools.chain(*clusters)))
    #
    # cluster_marginal_dists = [[] for _ in range(len(clusters))]
    # cluster_dist_stats = {}
    # cluster_kin_dist_mtx = np.zeros((len(clusters), len(clusters)), float)
    #
    # aligned_dist = AlignedDist()
    #
    # for i, j in itertools.combinations(range(len(clusters)), 2):
    #     pairwise_dists = []
    #     for seg_ix1_, seg_ix2_ in itertools.product(clusters[i], clusters[j]):
    #         seg_ix1, seg_ix2 = (seg_ix1_, seg_ix2_) if seg_ix1_ < seg_ix2_ else (seg_ix2_, seg_ix1_)
    #         pairwise_dists.append(aligned_dist(segments[seg_ix1].kin.X, segments[seg_ix2].kin.X))
    #
    #         if (seg_ix1, seg_ix2) in dists_dict:
    #             assert np.allclose(dists_dict[seg_ix1, seg_ix2], pairwise_dists[-1])
    #
    #     cluster_dist_stats[i, j] = stats.calc_stats(pairwise_dists)
    #     reduced_dist = cluster_dist_stats[i, j]['med']
    #     cluster_kin_dist_mtx[i, j] = reduced_dist
    #     cluster_marginal_dists[i].append(reduced_dist)
    #     cluster_marginal_dists[j].append(reduced_dist)
    #
    # cluster_mean_dists = [np.median(d) for d in cluster_marginal_dists]
    # chosen_cluster_inds = sorted(np.argsort(cluster_mean_dists)[:n_shape_to_take])
    # clusters = [clusters[i] for i in chosen_cluster_inds]
    # cluster_kin_dist_mtx = cluster_kin_dist_mtx[chosen_cluster_inds][:, chosen_cluster_inds]
    # cluster_kin_dist_mtx += cluster_kin_dist_mtx.T

    from common.utils import clustering

    ixs_of_shape = clustering.labels_to_ind_groups(
        affine_cluster_labels, label_to_key=lambda lbl: f'Shape{lbl + 1}')

    is_participating = affine_cluster_labels >= 0

    template_pts = {}
    for shape_name, seg_ixs in ixs_of_shape.items():
        X = segments[seg_ixs[0]].kin.X
        template_pts[shape_name] = (X - np.mean(X, axis=0)) / np.std(X)

    n_templates = len(template_pts)
    # --------

    #is_participating = np.ones_like(is_participating, bool)
    for pop_name in [NEURAL_POP.MINORITY, NEURAL_POP.MIDMAJ]:
        vecs = input_vecs.copy()
        vecs = vecs[:, neural_pop.inputs_mask(pop_name)]
        neural_labels = k_medoids(x=vecs, k=n_templates)
        print(str(pop_name) + ":")
        report_zscores(ClusterEvaluator().evaluate(labels_true=affine_cluster_labels, labels_pred=neural_labels))
        #validate_clustering_with_ground_truth(affine_cluster_labels[is_participating], pred_labels[is_participating])

        visualize_with_projection(vecs, true_labels=affine_cluster_labels,
                                  pred_labels=neural_labels, method='tsne')
        plt.title(str(pop_name))
    plt.show()


    # --------
    flexible_aligner = PlanarAlignTo('affine')
    rigid_aligner = PlanarAlignTo('rigid')
    axs = plotting.subplots(ncols=3, nrows=len(ixs_of_shape), eq=True)
    for shape_ix, (shape_name, seg_ixs) in enumerate(ixs_of_shape.items()):
        cluster_segs = [segments[i] for i in seg_ixs]
        flexible_aligner.X_dst = template_pts[shape_name]
        rigid_aligner.X_dst = template_pts[shape_name]
        for seg in cluster_segs:
            axs[shape_ix, 0].plot(*seg.kin.X.T)
            axs[shape_ix, 1].plot(*rigid_aligner(seg.kin.X).T)
            axs[shape_ix, 2].plot(*flexible_aligner(seg.kin.X).T)
        axs[0, 0].set_title('As-Is')
        axs[0, 1].set_title('Offset')
        axs[0, 2].set_title('Affine')
        axs[shape_ix, 0].set_ylabel(shape_name)

    for j in range(axs.shape[1]):
        xlms = np.array([ax.get_xlim() for ax in axs[:, j].ravel()])
        ylms = np.array([ax.get_ylim() for ax in axs[:, j].ravel()])
        xlm = xlms.min(), xlms.max()
        ylm = ylms.min(), ylms.max()
        lm = max(max(np.abs(ax.get_xlim()).max(), np.abs(ax.get_ylim()).max()) for ax in axs[:, j].ravel())
        for ax in axs[:, j].ravel():
            ax.set_xlim(*xlm)
            ax.set_ylim(*ylm)

    # --------

    colors = plotting.get_nice_colors(ixs_of_shape)

    # -----

    projs = segment_processing.compute_projections(model, input_vecs, neural_pop, groups=ixs_of_shape, n_pcs=2,
                                                   pop_names=[NEURAL_POP.MAJORITY, NEURAL_POP.MINORITY],
                                                   methods=methods)

    n_samples_to_draw = 0
    also_draw_segment = True
    for i_sample in range(n_samples_to_draw):
        axs = plotting.named_subplots(cols=ixs_of_shape, eq=True)
        for name, ixs in ixs_of_shape.items():
            tmplt_pts = template_pts[name]
            axs[name].plot(*tmplt_pts.T, color=colors[name])
            if also_draw_segment:
                seg_pts, _ = uniform_resample(segments[ixs[i_sample]].kin.X, len(tmplt_pts))
                seg_pts, _ = PlanarAlign('affine')(tmplt_pts, seg_pts)
                axs[name].plot(*seg_pts.T, color='k')

            axs[name].set_title(name)
            axs[name].axis('off')
        plt.suptitle(data_mgr.cfg.trials.name + " - EXAMPLE SHAPES")

    draw_projections(projs, ixs_of_shape, density_type='ellipse')
    plt.suptitle(data_mgr.monkey)

    data = {'Direction': [], 'Speed': [], 'Acceleration': [], 'Shape': []}
    for shape, seg_ixs in ixs_of_shape.items():
        data['Direction'] += [segments[i].kin.ang for i in seg_ixs]
        data['Speed'] += [segments[i].kin['EuSpd'].mean() for i in seg_ixs]
        data['Acceleration'] += [segments[i].kin['EuAcc'].mean() for i in seg_ixs]
        data['Shape'] += [shape] * len(seg_ixs)
    data = pd.DataFrame(data)

    params = [col for col in data.columns if col != 'Shape']
    shapes = list(ixs_of_shape)
    axs = plotting.named_subplots(cols=params, rows=shapes)
    plt.suptitle(data_mgr.cfg.trials.name + " - Euclidean Variables")
    for (shape, param), ax in axs.items():
        if param == 'Direction':
            ax = plotting.rebuild_axis(ax, {'projection': 'polar'})
        bins = np.linspace(0, data[param].max(), 8)
        data_ = data[data['Shape'] == shape]
        sns.histplot(data=data_, x=param, ax=ax, hue='Shape', common_norm=False, palette=colors, fill=True, bins=bins)
        ax.set_title(shape + " " + param)

    return
    n_clusters = len(clusters)
    cluster_dist_mtxs = {'kin': cluster_kin_dist_mtx}
    shape_names = list(ixs_of_shape)

    for (pop_name, method, pc_vecs, (silhouette, gmm_likelihood)) in projs:
        key = pop_name + " " + method
        cluster_dist_mtxs[key] = np.zeros((n_clusters, n_clusters), float)
        for i, j in itertools.combinations(range(n_clusters), 2):
            x1 = pc_vecs[ixs_of_shape[shape_names[i]]]
            x2 = pc_vecs[ixs_of_shape[shape_names[j]]]
            cluster_dist_mtxs[key][i, j] = np.mean(cdist(x1, x2))
            cluster_dist_mtxs[key][j, i] = cluster_dist_mtxs[key][i, j]

    axs = plotting.named_subplots(cols=cluster_dist_mtxs, style='white')
    for key, ax in axs.items():
        d = cluster_dist_mtxs[key].copy()
        d = np.clip(d, 0, np.percentile(d, 99))
        d /= d.max()
        # for i in range(len(d)):
        #     for j, name in enumerate(ixs_of_shape):
        #         if i != j:
        #             ax.plot(i, d[i, j], 'o', color=colors[name])

        sns.heatmap(d, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, square=True)
        #ax.imshow(d, cmap='jet')
        ax.set_title(key)


if __name__ == "__main__":
    for monkey, model_file in cv_results_mgr.get_chosen_model_per_monkey().items():
        draw_shape_embeddings(model_file)
    plt.show()
