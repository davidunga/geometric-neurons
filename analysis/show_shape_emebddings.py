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


SHAPE_PATTERNS = ['P.1', 'P.2', 'E.1', 'E.2']


def get_shape_colors() -> dict:
    return dict(zip(SHAPE_PATTERNS, plotting.get_nice_colors()))


def draw_shape_embeddings(model_file, n_pcs: int = 2, density_type: str = 'ellipse', n: int = 50):

    assert n_pcs in (2, 3)
    assert density_type in ('kde', 'ellipse', 'none')
    seed = 1
    shape_patterns = SHAPE_PATTERNS
    rnd = Rnd(seed)

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data, persist=True)
    segments = data_mgr.load_segments()
    neural_pop = NeuralPopulation.from_model(model_file)

    conics, scores = data_mgr.load_fitted_conics()

    fine_shapes, shape_scores = classify_shapes(conics, scores)

    ixs_of_shape = {patt: [] for patt in shape_patterns}
    for shape in fine_shapes:
        if not shape.is_valid:
            continue
        matching_patt = [patt for patt in shape_patterns if shape.is_match(patt)]
        if len(matching_patt):
            assert len(matching_patt) == 1
            patt = matching_patt[0]
            ixs_of_shape[patt].append(shape.seg_ix)

    print({patt: len(ixs) for patt, ixs in ixs_of_shape.items()})

    if n is not None:
        for patt, ixs in ixs_of_shape.items():
            ixs = np.array(ixs)
            ixs_of_shape[patt] = ixs[np.argsort(shape_scores[ixs])[-n:]]
        print("-->", {patt: len(ixs) for patt, ixs in ixs_of_shape.items()})

    shape_labels = np.zeros(len(segments), int)
    for patt, ixs in ixs_of_shape.items():
        shape_labels[ixs] = shape_patterns.index(patt) + 1

    input_vecs, _ = data_mgr.get_inputs()
    for pop_name in [NEURAL_POP.FULL, NEURAL_POP.MAJORITY, NEURAL_POP.MINORITY]:

        vecs = input_vecs.copy()
        vecs[:, ~neural_pop.inputs_mask(pop_name)] = .0
        vecs = embedtools.prep_embeddings(model, vecs)['NO']

        pc_vecs = LinearDiscriminantAnalysis(n_components=n_pcs).fit(
            X=vecs[shape_labels > 0], y=shape_labels[shape_labels > 0]).transform(vecs)

        ax = plotting.subplots(ndim=n_pcs)[0]
        for shape, seg_ixs in ixs_of_shape.items():
            color = get_shape_colors()[shape]
            if density_type == 'ellipse':
                plotting.plot_2d_gaussian_ellipse(pc_vecs[seg_ixs], ax=ax, edgecolor=color, facecolor='none', lw=1)
            elif density_type == 'kde':
                sns.kdeplot(*pc_vecs[seg_ixs].T, ax=ax, color=color, shade=True)
            ax.scatter(*pc_vecs[seg_ixs].T, alpha=.5, label=shape, color=color)

        plt.xlabel('Comp1')
        plt.ylabel('Comp2')
        plotting.set_axis_equal(ax)
        plt.title("\n".join([cfg.str(), "Affine Neural Subspace\n" + str(pop_name)]))
        plt.legend()

    # data = {'speed': [], 'accel': [], 'shape': []}
    # for shape, seg_ixs in ixs_of_shape.items():
    #     data['speed'] += [segments[i].kin['EuSpd'].mean() for i in seg_ixs]
    #     data['accel'] += [segments[i].kin['EuAcc'].mean() for i in seg_ixs]
    #     data['shape'] += [shape] * len(seg_ixs)
    # data = pd.DataFrame(data)

    # params = [col for col in data.columns if col != 'shape']
    # axs = plotting.named_subplots(rows=params)
    # for param in params:
    #     sns.kdeplot(data=data, x=param, ax=axs[param], hue='shape',
    #                 common_norm=False, palette=get_shape_colors(specs_hash), fill=True)


def classify_shapes(conics, scores) -> tuple[list[SegShape], np.ndarray[float]]:

    # -------
    # validity thresholds
    # scores-
    inls_thresh = .9
    mse_thresh = .005
    # conic-
    ax_ratio_thresh = 100
    bias_thresh = 1.5
    parab_m_max = 30
    parab_m_min = 0

    # -------
    # shape thresholds
    max_ellipse_e = .9
    bias_high_thresh = .6
    bias_low_thresh = .2

    equalize_types = False

    def _is_valid_conic(conic):
        if conic.kind == 'e' and conic.m[0] > ax_ratio_thresh * conic.m[1]:
            return False
        if conic.kind == 'p' and not (parab_m_min < abs(conic.m) < parab_m_max):
            return False
        if abs(conic.bounds_bias()) > bias_thresh:
            return False
        return True

    valid_score_ixs = scores.loc[(scores['inl'] > inls_thresh) & (scores['mse'] < mse_thresh)]['seg_ix'].tolist()

    parabola_scores = np.zeros(len(conics), float)
    ellipse_scores = np.zeros(len(conics), float)
    shapes = [SegShape(seg_ix=i) for i in range(len(conics))]
    for i in valid_score_ixs:

        c = conics[i]

        if not _is_valid_conic(c):
            continue

        if c.kind == 'p':
            score = 1 - abs(c.m)
        else:
            e = c.eccentricity()
            score = 1 - e if e < max_ellipse_e else 0

        bounds_bias = c.bounds_bias()
        if abs(bounds_bias) < bias_low_thresh:
            bias = -1 if bounds_bias < 0 else 1
        elif abs(bounds_bias) > bias_high_thresh:
            bias = -2 if bounds_bias < 0 else 2
        else:
            continue

        shapes[i].kind = c.kind.upper()
        shapes[i].bias = bias
        if c.kind == 'p':
            parabola_scores[i] = score
        else:
            ellipse_scores[i] = score

    if equalize_types:
        p_ixs = np.array([s.seg_ix for s in shapes if s.kind == 'P'])
        e_ixs = np.array([s.seg_ix for s in shapes if s.kind == 'E'])
        n_drop = abs(len(p_ixs) - len(e_ixs))
        if len(p_ixs) > len(e_ixs):
            drop_ixs = p_ixs[np.argsort(parabola_scores[p_ixs])][:n_drop]
        else:
            drop_ixs = e_ixs[np.argsort(ellipse_scores[e_ixs])][:n_drop]
        for i in drop_ixs:
            shapes[i] = SegShape(seg_ix=i)
        parabola_scores[drop_ixs] = 0
        ellipse_scores[drop_ixs] = 0

    scores = np.maximum(parabola_scores, ellipse_scores)
    return shapes, scores


def calc_and_save_conic_fits(model_file):
    from common.utils.devtools import progbar
    from common.utils.conics import get_conic
    from common.utils.conics.conic_fitting import fit_conic_ransac

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data, persist=True)
    conics, scores = data_mgr.load_fitted_conics()
    segments = data_mgr.load_segments()

    shapes, shape_scores = classify_shapes(conics, scores)

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


if __name__ == "__main__":
    for monkey, model_file in cv_results_mgr.get_chosen_model_per_monkey().items():
        if monkey == 'RS':
            draw_shape_embeddings(model_file)
    plt.show()
