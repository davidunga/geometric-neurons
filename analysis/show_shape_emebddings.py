import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
import paths
from collections import Counter
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
from common.utils import gaussians
from common.utils import dictools
from common.utils import hashtools
from copy import deepcopy
from common.utils import linalg
from analysis.config import DataConfig
from common.utils.conics.conic_fitting import fit_conic_ransac, Conic
from dataclasses import dataclass


@dataclass
class SegShape:
    seg_ix: int
    kind: str = None
    bias: int = None

    def name(self, signed_bias: bool = True) -> str:
        if not self.is_valid:
            return 'invalid'
        else:
            bias_str = f'{self.bias:+d}' if signed_bias else f'.{self.bias_level}'
            return f'{self.kind}{bias_str}'

    def __str__(self):
        return self.name()

    def is_match(self, pattern: str):
        name = self.name()
        assert len(pattern) == len(name)
        return all(p in ('.', n) for p, n in zip(pattern, name))

    @property
    def is_valid(self):
        return self.kind is not None

    @property
    def bias_level(self):
        return abs(self.bias)

    @property
    def bias_sign(self):
        return -1 if self.bias < 0 else 1


SHAPE_PATTERNS = ['P.1', 'P+2', 'E.1']


def get_shape_colors() -> dict:
    return dict(zip(SHAPE_PATTERNS, plotting.get_nice_colors()))


def draw_shape_embeddings(model_file, n: int = 30, n_pcs: int = 2, density_type: str = 'ellipse',
                          specs_hash: str = None, pop_name=NEURAL_POP.FULL):
    assert n_pcs in (2, 3)
    assert density_type in ('kde', 'ellipse', 'none')

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data, persist=True)
    segments = data_mgr.load_segments()

    conics, scores = data_mgr.load_fitted_conics()
    shape_labels = np.zeros(len(segments), int)
    fine_shapes = classify_shapes(conics, scores)
    shape_patterns = SHAPE_PATTERNS
    label_to_patt = {(i + 1): patt for i, patt in enumerate(shape_patterns)}
    patt_to_label = {patt: lbl for lbl, patt in label_to_patt.items()}
    ixs_of_shape = {patt: [] for patt in shape_patterns}
    for shape in fine_shapes:
        if shape.is_valid:
            matching_patt = [patt for patt in shape_patterns if shape.is_match(patt)]
            if len(matching_patt):
                assert len(matching_patt) == 1
                patt = matching_patt[0]
                ixs_of_shape[patt].append(shape.seg_ix)
                shape_labels[shape.seg_ix] = patt_to_label[patt]

    print(Counter(shape_labels))

    input_vecs, _ = data_mgr.get_inputs()

    neural_pop = NeuralPopulation.from_model(model_file)
    input_vecs[:, ~neural_pop.inputs_mask(pop_name)] = .0
    #subpop_size = len(neural_pop.neurons(NEURAL_POP.MINORITY))
    #neurons_to_nullify = neural_pop.neurons(nullify_pop, n=subpop_size, ranks='b')
    #input_vecs[:, neural_pop.inputs_mask(neurons_to_nullify)] = .0

    vecs = embedtools.prep_embeddings(model, input_vecs)['NO']

    pc_vecs = LinearDiscriminantAnalysis(n_components=n_pcs).fit(X=vecs[shape_labels>=0], y=shape_labels[shape_labels>=0]).transform(vecs)

    ax = plotting.subplots(ndim=n_pcs)[0]
    density_gausses = {}
    for shape, seg_ixs in ixs_of_shape.items():
        color = get_shape_colors()[shape]
        if density_type == 'ellipse':
            density_gausses[shape], _ = plotting.plot_2d_gaussian_ellipse(
                pc_vecs[seg_ixs], ax=ax, edgecolor=color, facecolor='none', linewidth=1)
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
    plt.title("\n".join([cfg.str(), "Affine Neural Subspace\n" + str(pop_name)]))
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


def classify_shapes(conics, scores) -> list[SegShape]:

    # -------
    # validity thresholds
    # scores-
    inls_thresh = .9
    mse_thresh = .005
    # conic-
    ax_ratio_thresh = 100
    bias_thresh = 1.5

    # -------
    # shape thresholds
    parab_e = .99
    ellipse_e = .9
    bias_high_thresh = .6
    bias_low_thresh = .2

    equalize_types = True

    def _is_valid_conic(conic):
        if conic.kind == 'e' and conic.m[0] > ax_ratio_thresh * conic.m[1]:
            return False
        if abs(conic.bounds_bias()) > bias_thresh:
            return False
        return True

    valid_score_ixs = scores.loc[(scores['inl'] > inls_thresh) & (scores['mse'] < mse_thresh)]['seg_ix'].tolist()

    es = np.array([c.eccentricity() for c in conics], float)
    parab_score = es.copy()
    parab_score[es < parab_e] = 0
    ellipse_score = 1 - es.copy()
    ellipse_score[es > ellipse_e] = 0

    shapes = [SegShape(seg_ix=i) for i in range(len(conics))]
    for i in valid_score_ixs:
        if parab_score[i] == ellipse_score[i] == 0:
            continue
        if not _is_valid_conic(conics[i]):
            continue
        bounds_bias = conics[i].bounds_bias()
        if abs(bounds_bias) < bias_low_thresh:
            bias = -1 if bounds_bias < 0 else 1
        elif abs(bounds_bias) > bias_high_thresh:
            bias = -2 if bounds_bias < 0 else 2
        else:
            continue

        shapes[i].kind = 'E' if ellipse_score[i] > parab_score[i] else 'P'
        shapes[i].bias = bias

    print("parab", np.mean(parab_score[[s.seg_ix for s in shapes if s.kind=='P']]), len([s.seg_ix for s in shapes if s.kind=='P']))
    print("ellipse", np.mean(ellipse_score[[s.seg_ix for s in shapes if s.kind=='E']]), len([s.seg_ix for s in shapes if s.kind=='E']))
    if equalize_types:
        p_ixs = np.array([s.seg_ix for s in shapes if s.kind == 'P'])
        e_ixs = np.array([s.seg_ix for s in shapes if s.kind == 'E'])
        n_drop = abs(len(p_ixs) - len(e_ixs))
        if len(p_ixs) > len(e_ixs):
            drop_ixs = p_ixs[np.argsort(parab_score[p_ixs])][:n_drop]
        else:
            drop_ixs = e_ixs[np.argsort(ellipse_score[e_ixs])][:n_drop]
        for i in drop_ixs:
            shapes[i] = SegShape(seg_ix=i)
    print("parab", np.mean(parab_score[[s.seg_ix for s in shapes if s.kind=='P']]), len([s.seg_ix for s in shapes if s.kind=='P']))
    print("ellipse", np.mean(ellipse_score[[s.seg_ix for s in shapes if s.kind=='E']]), len([s.seg_ix for s in shapes if s.kind=='E']))
    return shapes




def calc_and_save_conic_fits(model_file):
    from common.utils.devtools import progbar
    from common.utils.conics import get_conic
    from common.utils.conics.conic_fitting import fit_conic_ransac

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data, persist=True)
    conics, scores = data_mgr.load_fitted_conics()
    segments = data_mgr.load_segments()

    shapes = classify_shapes(conics, scores)

    from common.utils import strtools
    print(strtools.parts(isvalid=[s.is_valid for s in shapes]))
    valid_shapes = [s for s in shapes if s.is_valid]
    print(strtools.parts(shape=[s.name() for s in valid_shapes]))
    print(strtools.parts(shape=[s.name(signed_bias=False) for s in valid_shapes]))
    print(strtools.parts(kind=[s.kind for s in valid_shapes]))
    print(strtools.parts(bias_level=[s.bias_level for s in valid_shapes]))
    print(strtools.parts(bias=[s.bias for s in valid_shapes]))
    print(strtools.parts(bias_sign=[s.bias_sign for s in valid_shapes]))

    for shape in shapes:
        if not shape.is_valid:
            continue
        i = shape.seg_ix
        c = conics[i]
        s = segments[i]
        Xi = c.inv_transform(s.kin.X)
        ci = c.get_standardized()
        (ax1, ax2) = plotting.subplots(ncols=2)
        plt.sca(ax1)
        plotting.plot(s.kin.X, '.k', marks='0')
        plotting.plot(c.parametric_pts(), 'r-', marks='0')
        plotting.set_axis_equal()
        plt.sca(ax2)
        plotting.plot(Xi, '.k', marks='0')
        plotting.plot(ci.parametric_pts(), 'r-', marks='0')
        plotting.set_axis_equal()
        plt.suptitle(str(shape) + " -- " + str(c) + f" bias={c.bounds_bias():.2f} \n" + scores.loc[i]['str'])
        plt.show()


    #
    # valid_ixs = [score['seg_ix'] for score, conic in zip(scores, conics) if _is_valid(conic, score)]
    # print(len(valid_ixs))

    #
    # for ix in range(len(segments)):
    #     print("ix", ix)
    #     c, ev = fit_conic_ransac(segments[ix].kin.X, thresh=.05, inlier_p_thresh=1.1)
    #     valid = c.m[0] < 100 * c.m[1] if c.kind=='e' else True
    #     plt.plot(*segments[ix].kin.X.T, 'k.')
    #     plt.plot(*segments[ix].kin.X[0].T, 'ks')
    #     plt.plot(*segments[ix].kin.X[ev['_inl_mask']].T, 'r.')
    #     plt.plot(*c.parametric_pts().T, 'c-')
    #     plt.plot(*c.parametric_pts()[0].T, 'c*')
    #     #t0 = linalg.average_theta(np.radians(c.bounds)) * 180 / np.pi
    #     #vi = abs(t0) < 90
    #     #print("vi",  linalg.circdiff(t0, 180, mod='d'), "t0", t0)
    #     #vpt = c.vertex_pts()[1]
    #     #plt.plot(vpt[0], vpt[1], 'r*')
    #     bbias = f" bias={c.bounds_bias():.2f} "
    #     plt.title(str(c) + "\n" + ev['str'] + bbias + (" INVALID" if not valid else ""))
    #     plt.axis('equal')
    #     plt.show()

    # fit_kws = {'max_itrs': 500, 'thresh': .05, 'inlier_p_thresh': 1.1, 'seed': 1, 'n': 7}
    #
    # print("Computing conics for " + str(model_file))
    #
    # conic_fits = []
    # for s in progbar(segments, span=20):
    #     conic, scores = fit_conic_ransac(s.kin.X, **fit_kws)
    #     conic_fits.append({'seg_ix': s.ix, 'conic': conic.to_json(), 'scores': scores})
    #
    # items = {
    #     'fit_kws': fit_kws,
    #     'conic_fits': conic_fits
    # }
    #
    # conics_file = paths.PROCESSED_DIR / (data_mgr.cfg.str(DataConfig.SEGMENTS) + '.CONICS.json')
    # conics_file.parent.mkdir(exist_ok=True)
    # print("Saving to " + str(conics_file))
    # json.dump(items, conics_file.open('w'))
    #
    # # --
    # print("Validating...")
    # items = json.load(conics_file.open('r'))
    # conics = [get_conic(**fit_result['conic']) for fit_result in items['conic_fits']]
    # print("Okay. Done.")



if __name__ == "__main__":
    # for monkey, model_file in cv_results_mgr.get_chosen_model_per_monkey().items():
    #     seek_shapes(model_file)

    #default_specs_hash = '66f69a09831a79ab72c73f40fb0744ecf98ef9af'
    #best_shapes()
    for monkey, model_file in cv_results_mgr.get_chosen_model_per_monkey().items():
        if monkey == 'RS':
            for pop_name in [NEURAL_POP.FULL, NEURAL_POP.MINORITY, NEURAL_POP.MAJORITY]:
                draw_shape_embeddings(model_file, pop_name=pop_name)
        #calc_and_save_conic_fits(model_file)
    #     for null_pop in [None, NEURAL_POP.MINORITY, NEURAL_POP.MAJORITY]:
    #         draw_shape_embeddings(model_file, n=30, n_pcs=2, specs_hash=default_specs_hash, nullify_pop=null_pop)
    # #draw_shapes(default_specs_hash)
    plt.show()
