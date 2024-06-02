import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths
from common.utils import stats
from common.utils.procrustes import PlanarAlign
from common.utils.distance_metrics import normalized_mahalanobis
from motorneural.data import Segment
from data_manager import DataMgr
import cv_results_mgr
from neural_population import NeuralPopulation, NEURAL_POP
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from common.utils import plotting
from common.utils import dlutils
from common.utils.randtool import Rnd
from common.utils import polytools
import seaborn as sns
import embedtools
import seaborn as sns
from common.utils import gaussians
from common.utils import dictools
from common.utils import hashtools
from copy import deepcopy
from analysis.config import DataConfig

from common.utils.conics.conic_fitting import fit_conic_ransac, Conic, eval_conic_fit
#
#
# class ParamShape:
#
#     def __init__(self, params: dict):
#         self.params = params
#
#     def pts(self, n: int = 50, norm: bool = True) -> np.ndarray:
#         if self.params['kind'] in ('e', 'p'):
#             pts = make_conic_points(n=n, **self.params)
#         else:
#             raise ValueError('Unknown parametric shape kind')
#         if norm:
#             pts /= np.std(pts)
#         return pts
#
#     @property
#     def uid(self) -> str:
#         return self.params['kind'] + hashtools.calc_hash(self.params)
#
#
# def calc_segment_shape_dists(data_mgr: DataMgr, shapes: list[ParamShape], aligner_kind: str = 'ortho') -> pd.DataFrame:
#     aligner = PlanarAlign(kind=aligner_kind)
#     def _dist(shape_pts_, traj_):
#         aligned_traj = aligner(traj_, shape_pts)[0]
#         return normalized_mahalanobis(shape_pts_, aligned_traj)
#     dists_per_shape = {}
#     trajs = [seg.kin.X for seg in data_mgr.load_segments()]
#     for j, shape in enumerate(shapes):
#         shape_pts = shape.pts(len(trajs))
#         dists_per_shape[shape.uid] = [_dist(shape_pts, traj) for traj in trajs]
#     df = pd.DataFrame(data=dists_per_shape)
#     return df
#
#
# def get_shape_candidates():
#     shapes_grid = {
#         'shape1': dict(kind='p', m=[10, 15, 20], start=-1, stop=1),
#         'shape2': dict(kind='p', m=[1/2, 3, 4, 5, 6], start=-1, stop=[1.5, 2, 3]),
#         'shape3': dict(kind='e', m=[1, 1.5], start=[10, 20, 30], stop=[150, 160, 170]),
#     }
#     return list(ParamShape(p) for p in dictools.dict_product_from_grid(shapes_grid))
#
#
#
# def get_all_shape_specs() -> dict:
#     shapes_grid = {
#         'shape1': dict(kind='p', m=[10, 20], start=-1, stop=1),
#         'shape2': dict(kind='p', m=[4, 5, 6, 8], start=-1, stop=[1.5, 2, 3]),
#         'shape3': dict(kind='e', m=[1, 1.5], start=[10, 20], stop=[160, 170]),
#         #'shape4': dict(kind='e', m=[2, 3, 4], start=[20, 30, 40], stop=[140, 150])
#     }
#     shapes_grid = {name: list(dictools.dict_product_from_grid(grid)) for name, grid in shapes_grid.items()}
#     shape_specs = {get_specs_hash(spec): spec for spec in dictools.dict_product_from_grid(shapes_grid)}
#     return shape_specs
#
#
# def get_shapes(specs_hash: str = None, shape_specs: dict = None, n_pts: int = 100) -> dict[str, np.ndarray]:
#
#     #
#     # if shape_specs is None:
#     #     shape_specs = [
#     #         dict(kind='p', m=20, start=-1, stop=1),
#     #         dict(kind='p', m=5, start=-1, stop=3),
#     #         dict(kind='e', m=1, start=10, stop=170),
#     #         dict(kind='e', m=3, start=30, stop=150),
#     #     ]
#     #     # shape_specs = {
#     #     #     'shape1': {'kind': 'p', 'm': 20, 'start': -1, 'stop': 1, 'n': n_pts},
#     #     #     'shape2': {'kind': 'p', 'm': 4, 'start': -1, 'stop': 2, 'n': n_pts},
#     #     #     'shape3': {'kind': 'e', 'm': 1, 'start': 10, 'stop': 170, 'n': n_pts},
#     #     #     'shape4': {'kind': 'e', 'm': 2, 'start': 20, 'stop': 150, 'n': n_pts}
#     #     # }
#
#     def _process_shape(pts):
#         return pts / np.std(pts)
#
#     # if not isinstance(shape_specs, dict):
#     #     shape_specs = {f'shape{i+1}': spec for i, spec in enumerate(shape_specs)}
#
#     if specs_hash is not None:
#         assert shape_specs is None
#         shape_specs = get_all_shape_specs()[specs_hash]
#
#     shapes = {name: _process_shape(make_conic_points(n=n_pts, **spec)) for name, spec in shape_specs.items()}
#     return shapes
#
#
# def get_shape_colors(specs_hash) -> dict:
#     return dict(zip(get_shapes(specs_hash).keys(), plotting.get_nice_colors()))
#
#
# def draw_shapes(specs_hash):
#     shapes_dict = get_shapes(specs_hash)
#     axs = plotting.subplots(nrows=len(shapes_dict))
#     for ax, (shape_name, shape_pts) in zip(axs, shapes_dict.items()):
#         ax.plot(*shape_pts.T, color=get_shape_colors(specs_hash)[shape_name])
#         plotting.set_axis_equal(ax)
#         ax.axis('off')
#         ax.set_title(shape_name)
#
#
# def get_segments_of_shape(segments: list[Segment], shape_pts: np.ndarray, n: int | float = 10,
#                           stratify_by: str = 'EuSpd', exclude_ixs=None):
#
#     aligner = PlanarAlign(kind='ortho')
#     if n < 1:
#         n = int(round(n * len(segments)))
#     aligned_trajs = [aligner(seg.kin.X, shape_pts)[0] for seg in segments]
#     dists = np.fromiter((normalized_mahalanobis(shape_pts, traj) for traj in aligned_trajs), float)
#     if exclude_ixs is not None:
#         dists[exclude_ixs] = np.inf
#     if stratify_by is not None:
#         values = [seg[stratify_by].mean() for seg in segments]
#         labels = stats.safe_digitize(values, stats.BinSpec(n, 'p'))[0]
#         seg_ixs = np.zeros(n, int)
#         for label in range(n):
#             ixs = np.nonzero(labels == label)[0]
#             seg_ixs[label] = ixs[np.argmin(dists[ixs])]
#     else:
#         seg_ixs = np.argsort(dists)[:n]
#     return seg_ixs, dists[seg_ixs]
#
#
# def best_shapes():
#     import json
#     import pandas as pd
#     jsn = "/Users/davidu/geometric-neurons/outputs/cv/TP_RS bin10 lag100 dur200 affine-kinX-nmahal 669106 - shapes.json"
#     items = json.load(open(jsn, 'r'))
#     for item in items:
#         item['specs_hash'] = get_specs_hash(item['specs'])
#     data_RS = pd.DataFrame(items)
#
#     jsn = "/Users/davidu/geometric-neurons/outputs/cv/TP_RJ bin10 lag100 dur200 affine-kinX-nmahal f44c17 - shapes.json"
#     items = json.load(open(jsn, 'r'))
#     for item in items:
#         item['specs_hash'] = get_specs_hash(item['specs'])
#     data_RJ = pd.DataFrame(items)
#
#     assert (data_RS['specs_hash']==data_RJ['specs_hash']).all()
#
#     dist_cols = [col for col in data_RS.columns if col.endswith('_dist')]
#     data = data_RS.copy(deep=True)
#     for col in dist_cols:
#         data[col] = np.mean(np.c_[data_RS[col], data_RJ[col]], axis=1)
#
#     #data = data.loc[data['nullify']==str(NEURAL_POP.MAJORITY),:]
#     data = data.loc[data['embed_type']=='YES',:]
#     for col in dist_cols:
#         i = np.argmax(data[col].to_numpy())
#         dist = data.iloc[i][col]
#         specs = data.iloc[i]['specs']
#         print(f"Best {col} = {dist:2.3f} -- {i} {data.iloc[i]['specs_hash']} {specs}")
#
#
# def seek_shapes(model_file, n: int = 30):
#     import json
#     n_pcs = 2
#
#     model, cfg = cv_results_mgr.get_model_and_config(model_file)
#     data_mgr = DataMgr(cfg.data, persist=True)
#     segments = data_mgr.load_segments()
#     input_vecs, _ = data_mgr.get_inputs()
#     embeddings = embedtools.prep_embeddings(model, input_vecs)
#     neural_pop = NeuralPopulation.from_model(model_file)
#
#     from paths import CV_DIR
#     dump_file = CV_DIR / (cfg.output_name + " - shapes.json")
#
#     results = []
#     if dump_file.is_file():
#         results = json.load(dump_file.open('r'))
#
#     n_pts = len(segments[0])
#     from common.utils import hashtools
#
#     shape_specs = get_all_shape_specs()
#     scores = []
#
#     for spec_ix, (specs_hash, specs) in enumerate(shape_specs.items()):
#         print(f"{spec_ix}/{len(shape_specs)}", end=" ")
#         assert specs_hash == hashtools.calc_hash(specs, fmt='hex')
#
#         shapes = None
#         for embed_type, embed_vecs in embeddings.items():
#             for nullify_pop in [None, NEURAL_POP.MAJORITY, NEURAL_POP.MINORITY]:
#                 if nullify_pop is not None and embed_type != 'NO':
#                     continue
#
#                 nullify_str = str(nullify_pop) if nullify_pop is not None else 'null'
#                 exists = False
#                 for result_ in results:
#                     if result_['specs_hash'] == specs_hash and result_['embed_type'] == embed_type and result_['nullify'] == nullify_str:
#                         exists = True
#                         break
#                 if exists:
#                     print("EXISTS")
#                     continue
#
#                 if shapes is None:
#                     shapes = get_shapes(n_pts=n_pts, shape_specs=specs)
#                     collected_ixs = []
#                     ixs_of_shape = {}
#                     shape_labels = np.zeros(len(input_vecs), int)
#                     for i, (shape_name, shape_pts) in enumerate(shapes.items(), start=1):
#                         seg_ixs, _ = get_segments_of_shape(segments, shape_pts, n=n, exclude_ixs=collected_ixs)
#                         shape_labels[seg_ixs] = i
#                         collected_ixs += list(seg_ixs)
#                         ixs_of_shape[shape_name] = seg_ixs
#
#                 print(embed_type, nullify_pop)
#                 vecs = embed_vecs.copy()
#                 if nullify_pop is not None:
#                     vecs[:, neural_pop.inputs_mask(pop=nullify_pop)] = 0
#
#                 shape_names = list(ixs_of_shape.keys())
#                 try:
#                     pc_vecs = LinearDiscriminantAnalysis(n_components=n_pcs).fit(X=vecs[shape_labels>0], y=shape_labels[shape_labels>0]).transform(vecs)
#                     density_gausses = {}
#                     for shape, seg_ixs in ixs_of_shape.items():
#                         density_gausses[shape], _ = gaussians.gaussian_fit(pc_vecs[seg_ixs])
#                     dists = []
#                     for i in range(len(shape_names) - 1):
#                         g1 = density_gausses[shape_names[i]]
#                         for j in range(i + 1, len(shape_names)):
#                             g2 = density_gausses[shape_names[j]]
#                             dists.append(gaussians.bhattacharyya_distance(g1, g2))
#                 except:
#                     dists = [0] * ((len(shape_names) * (len(shape_names) - 1)) // 2)
#                     assert len(dists) == len(next(iter(results))['dists'])
#
#                 result = {
#                     'specs_hash': specs_hash,
#                     'embed_type': embed_type,
#                     'nullify': nullify_str,
#                     'specs': specs,
#                     'dists': dists,
#                     'min_dist': min(dists),
#                     'max_dist': max(dists),
#                     'med_dist': np.median(dists),
#                     'avg_dist': np.mean(dists),
#                 }
#                 results.append(result)
#                 json.dump(results, dump_file.open('w'), indent=4)
#
#     si = np.argsort(scores)[::-1]
#     for i in si:
#         print(i, scores[i])



def draw_shape_embeddings(model_file, n: int = 30, n_pcs: int = 2, density_type: str = 'ellipse',
                          specs_hash: str = None, nullify_pop=None):
    assert n_pcs in (2, 3)
    assert density_type in ('kde', 'ellipse', 'none')
    from common.utils import hashtools

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data, persist=True)
    segments = data_mgr.load_segments()

    from time import time
    from common.utils.devtools import progbar
    fit_kws = {'max_itrs': 100, 'normdist_thresh': .05, 'inlier_p_thresh': .9, 'seed': 1, 'n': 7, 'kinds': ['e', 'p']}
    conic_fits = []
    for s in progbar(segments[:10], span=5):
        conic, scores = fit_conic_ransac(s.kin.X, **fit_kws)
        conic_fits.append({'seg_ix': s.ix, 'conic': conic.to_json(), 'scores': scores})
    items = {
        'fit_kws': fit_kws,
        'conic_fits': conic_fits
    }
    conics_file = paths.PROCESSED_DIR / (data_mgr.cfg.str(DataConfig.SEGMENTS) + '.CONICS.json')
    conics_file.parent.mkdir(exist_ok=True)
    json.dump(conic_fits, conics_file.open('wb'))

    # print("Fitting...")
    # tic = time()
    # conic_fits = [fit_conic_ransac(s.kin.X, max_itrs=10, kinds=['e']) for s in segments]
    # print(time() - tic)

    shape_candidates = [] # get_shape_candidates()
    #calc_segment_shape_dists(data_mgr, shapes=shape_candidates)
    shapes = {}
    #shapes = get_shapes(n_pts=len(segments[0]), specs_hash=specs_hash)

    input_vecs, _ = data_mgr.get_inputs()

    collected_ixs = []
    ixs_of_shape = {}
    shape_labels = np.zeros(len(input_vecs), int)
    for i, (shape_name, shape_pts) in enumerate(shapes.items(), start=1):
        seg_ixs, _ = [], []
        shape_labels[seg_ixs] = i
        collected_ixs += list(seg_ixs)
        ixs_of_shape[shape_name] = seg_ixs


    if nullify_pop is not None:
        neural_pop = NeuralPopulation.from_model(model_file)
        subpop_size = len(neural_pop.neurons(NEURAL_POP.MINORITY))
        neurons_to_nullify = neural_pop.neurons(nullify_pop, n=subpop_size, ranks='b')
        input_vecs[:, neural_pop.inputs_mask(neurons_to_nullify)] = .0

    vecs = embedtools.prep_embeddings(model, input_vecs)['YES']

    pc_vecs = LinearDiscriminantAnalysis(n_components=n_pcs).fit(X=vecs[shape_labels>0], y=shape_labels[shape_labels>0]).transform(vecs)

    ax = plotting.subplots(ndim=n_pcs)[0]
    density_gausses = {}
    for shape, seg_ixs in ixs_of_shape.items():
        color = get_shape_colors(specs_hash)[shape]
        if density_type == 'ellipse':
            density_gausses[shape], _ = plotting.plot_2d_gaussian_ellipse(pc_vecs[seg_ixs], ax=ax,
                                                                    edgecolor=color, facecolor='none', linewidth=1)
        elif density_type == 'kde':
            sns.kdeplot(*pc_vecs[seg_ixs].T, ax=ax, color=color, shade=True)
        ax.scatter(*pc_vecs[seg_ixs].T, alpha=.5, label=shape, color=color)


    from common.utils import gaussians
    shape_names = list(ixs_of_shape.keys())
    for i in range(len(shape_names) - 1):
        for j in range(i, len(shape_names)):
            g1 = density_gausses[shape_names[i]]
            g2 = density_gausses[shape_names[j]]
            print("Dist", shape_names[i], shape_names[j], "=", gaussians.bhattacharyya_distance(g1, g2))

    plt.xlabel('Comp1')
    plt.ylabel('Comp2')
    plotting.set_axis_equal(ax)
    plt.title("\n".join([cfg.str(), "Affine Neural Subspace - 2D Projection", "color coded by trajectory shape", "null=" + str(nullify_pop)]))
    plt.legend()

    data = {'speed': [], 'accel': [], 'shape': []}
    for shape, seg_ixs in ixs_of_shape.items():
        data['speed'] += [segments[i].kin['EuSpd'].mean() for i in seg_ixs]
        data['accel'] += [segments[i].kin['EuAcc'].mean() for i in seg_ixs]
        data['shape'] += [shape] * len(seg_ixs)
    data = pd.DataFrame(data)

    # params = [col for col in data.columns if col != 'shape']
    # axs = plotting.named_subplots(rows=params)
    # for param in params:
    #     sns.kdeplot(data=data, x=param, ax=axs[param], hue='shape',
    #                 common_norm=False, palette=get_shape_colors(specs_hash), fill=True)


def calc_and_save_conic_fits(model_file):
    from common.utils.devtools import progbar
    from common.utils.conics import get_conic

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data, persist=True)
    data_mgr.load_fitted_conics()
    segments = data_mgr.load_segments()

    fit_kws = {'max_itrs': 100, 'normdist_thresh': .05, 'inlier_p_thresh': .9,
               'seed': 1, 'n': 7, 'kinds': ['e', 'p']}

    print("Computing conics for " + str(model_file))

    # conic_fits = []
    # for s in progbar(segments, span=20):
    #     conic, scores = fit_conic_ransac(s.kin.X, **fit_kws)
    #     conic_fits.append({'seg_ix': s.ix, 'conic': conic.to_json(), 'scores': scores})
    # items = {
    #     'fit_kws': fit_kws,
    #     'conic_fits': conic_fits
    # }

    conics_file = paths.PROCESSED_DIR / (data_mgr.cfg.str(DataConfig.SEGMENTS) + '.CONICS.json')
    conics_file.parent.mkdir(exist_ok=True)
    print("Saving to " + str(conics_file))
    #json.dump(items, conics_file.open('w'))

    # --
    print("Validating...")
    items = json.load(conics_file.open('r'))
    conics = [get_conic(**fit_result['conic']) for fit_result in items['conic_fits']]
    print("Okay. Done.")



if __name__ == "__main__":
    # for monkey, model_file in cv_results_mgr.get_chosen_model_per_monkey().items():
    #     seek_shapes(model_file)

    #default_specs_hash = '66f69a09831a79ab72c73f40fb0744ecf98ef9af'
    #best_shapes()
    for monkey, model_file in cv_results_mgr.get_chosen_model_per_monkey().items():
        calc_and_save_conic_fits(model_file)
    #     for null_pop in [None, NEURAL_POP.MINORITY, NEURAL_POP.MAJORITY]:
    #         draw_shape_embeddings(model_file, n=30, n_pcs=2, specs_hash=default_specs_hash, nullify_pop=null_pop)
    # #draw_shapes(default_specs_hash)
    # plt.show()
