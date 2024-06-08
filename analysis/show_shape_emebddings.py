import json
import os.path
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from common.utils import hashtools
from glob import glob
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
from common.utils.conics import ConicEllipse, ConicParabola


@dataclass
class ShapeSpec:
    kind: str
    e: float
    bias: float

    @property
    def name(self) -> str:
        if not self.is_valid:
            return 'invalid'
        else:
            return f'{self.kind}{self.bias:+1.2f} {self.e:1.2f}'

    def __str__(self):
        return self.name

    def to_dict(self):
        return {'kind': self.kind, 'e': self.e, 'bias': self.bias}

    @property
    def is_valid(self):
        return self.kind is not None

    def dist2(self, other):
        e_scale = .05
        bias_scale = .2
        if other.kind != self.kind:
            return np.inf
        e_diff = (self.e - other.e) / e_scale
        bias_diff = (self.bias - other.bias) / bias_scale
        return e_diff ** 2 + bias_diff ** 2


SHAPE_SPECS = [ShapeSpec(kind='p', e=1, bias=1),
               ShapeSpec(kind='e', e=.9, bias=1),
               ShapeSpec(kind='e', e=.8, bias=0)]

SHAPE_SPECS = {spec.name: spec for spec in SHAPE_SPECS}

specs_file = paths.PROCESSED_DIR / 'shape_specs.json'


def get_random_specs(seed) -> dict[str, ShapeSpec]:
    bias_range = -2, 2
    e_range = .4, .95

    rng = np.random.default_rng(seed)
    kinds = ['e', 'p', 'p'] if rng.random() > .5 else ['p', 'e', 'e']

    ret = {}
    for i, kind in enumerate(kinds):
        bias = rng.uniform(*bias_range)
        if i == 1:
            bias = abs(bias)
        elif i == 2:
            bias = -abs(bias)
        e = rng.uniform(*e_range) if kind == 'e' else 1.
        spec = ShapeSpec(kind=kind, e=e, bias=bias)
        ret[spec.name] = spec

    return ret


def get_spec(conic: Conic) -> ShapeSpec:
    return ShapeSpec(kind=conic.kind, e=conic.eccentricity(), bias=conic.bounds_bias())


def get_examplar_conic_from_spec(spec: ShapeSpec):
    spec = spec.to_dict()
    bias = spec.get('bias', 0)
    lb = 1
    ub = lb * (abs(bias) + 1)
    if bias < 0:
        lb, ub = ub, lb
    bounds = (-lb, ub)
    if spec['kind'] == 'p':
        assert spec['e'] == 1
        conic = ConicParabola(m=1, loc=(0, 0), ang=0, bounds=bounds)
    else:
        assert spec['kind'] == 'e'
        r = np.sqrt(1 - spec['e'] ** 2)
        a, b = (r, 1) if r > 1 else (1, r)
        conic = ConicEllipse(m=(a, b), loc=(0, 0), ang=0, bounds=bounds)

    conic_spec = get_spec(conic).to_dict()
    assert all(conic_spec[k] == spec[k] if k == 'kind' else abs(conic_spec[k] - spec[k]) < max(abs(spec[k]) * .1, .1)
               for k in spec)

    return conic

# -- validation..
# for name, spec in SHAPE_SPECS.items():
#     get_examplar_conic_from_spec(spec)
# ------


#
# def calc_spec_dists(shape_specs: list[ShapeSpec], segment_specs_df: pd.DataFrame):
#
#     num_cols = [c for c in segment_specs_df.columns if c not in ('kind', 'seg_ix')]
#     dists2 = np.zeros((len(shape_specs), len(segment_specs_df)), float) + np.inf
#
#     segment_specs = {}
#     for kind in set(segment_specs_df['kind']):
#         ixs = np.nonzero(segment_specs_df['kind'].to_numpy() == kind)[0]
#         df = segment_specs_df.iloc[ixs]
#         mu = df.loc[:, num_cols].mean(axis=0)
#         sd = df.loc[:, num_cols].std(axis=0)
#         df.loc[:, num_cols] = (df.loc[:, num_cols] - mu) / sd
#         segment_specs[kind] = (df, mu, sd, ixs)
#
#     for i, name in enumerate(shape_specs_dicts):
#         spec_dict = shape_specs_dicts[name]
#         df, mu, sd, ixs = segment_specs[spec_dict['kind']]
#         cols = [k for k, v in spec_dict.items() if v is not None and k != 'kind']
#         vals = np.array([spec_dict[col] for col in cols], float)
#         vals = (vals - mu[cols]) / sd[cols]
#         dists2[i, ixs] = ((df.loc[:, cols] - vals) ** 2).sum(axis=1)
#
#     return dists2


def index_groups_to_labels(group_ixs: dict, n: int, keys=None, start_label: int = 1):
    labels = np.zeros(n, int)
    if keys is None:
        keys = group_ixs.keys()
    for label, key in enumerate(keys, start=start_label):
        labels[group_ixs[key]] = label
    return labels


def calc_stratification_labels(segments: list[Segment], n: int, by: str = 'EuSpd', include_ixs=None) -> np.ndarray[int]:
    if include_ixs is None:
        include_ixs = np.arange(len(segments))
    else:
        include_ixs = np.sort(include_ixs)
    values = [segments[i][by].mean() for i in include_ixs]
    labels = np.zeros(len(segments), int)
    labels[include_ixs] = stats.safe_digitize(values, stats.BinSpec(n, 'p'))[0]
    return labels


def get_valid_seg_ixs(scores_df: pd.DataFrame) -> list[int]:
    inls_thresh = .9
    mse_thresh = .005
    valid_ixs = scores_df.loc[(scores_df['inl'] > inls_thresh) & (scores_df['mse'] < mse_thresh)]['seg_ix'].tolist()
    return valid_ixs


def match_segments_to_shapes(conics: list[Conic], segment_labels: np.ndarray,
                             shape_specs: dict[str, ShapeSpec], n: int = 50):

    assert segment_labels.max() == n - 1
    valid_ixs = np.nonzero(segment_labels)[0]
    segment_labels = segment_labels[valid_ixs]

    dists2 = np.zeros((len(shape_specs), len(valid_ixs)))
    for i, spec in enumerate(shape_specs.values()):
        for j, seg_ix in enumerate(valid_ixs):
            dists2[i, j] = spec.dist2(get_spec(conics[seg_ix]))

    ixs_of_shape = {spec_name: [] for spec_name in shape_specs}
    for stratify_label in range(1, n):
        label_ixs = np.nonzero(segment_labels == stratify_label)[0]
        for i, (name, spec) in enumerate(shape_specs.items()):
            j = label_ixs[np.argmin(dists2[i][label_ixs])]
            if np.isfinite(dists2[i, j]):
                ixs_of_shape[name].append(valid_ixs[j])
                dists2[:, j] = np.inf

    return ixs_of_shape


def yield_spec_candidates():
    bias_grid = np.round(np.linspace(-2, 2, 7), 2)
    e_grid = np.array([0.55, 0.65, 0.7, 0.75, 0.8, 0.85, 0.95])
    for kinds in (['e', 'p', 'p'], ['p', 'e', 'e']):
        es = tuple(e_grid if kind == 'e' else [1] for kind in kinds)
        for b1, b2, b3, e1, e2, e3 in product([0], bias_grid[bias_grid >= 0], bias_grid[bias_grid < 0], *es):
            s1 = ShapeSpec(kind=kinds[0], e=e1, bias=b1)
            s2 = ShapeSpec(kind=kinds[1], e=e2, bias=b2)
            s3 = ShapeSpec(kind=kinds[2], e=e3, bias=b3)
            spec_cand = {s1.name: s1, s2.name: s2, s3.name: s3}
            yield spec_cand


def compute_projections(model, input_vecs, neural_pop: NeuralPopulation, shape_labels: np.ndarray,
                        pop_names: list[NEURAL_POP], n_pcs: int = 2, embed_type: str = 'NO'):

    methods = ['lda', 'pca1', 'pca2']

    results = []
    for pop_name in pop_names:
        mask = shape_labels > 0
        vecs = input_vecs.copy()
        vecs[:, ~neural_pop.inputs_mask(pop_name)] = .0
        vecs = embedtools.prep_embeddings(model, vecs, embed_types=[embed_type])[embed_type]

        XX = vecs[mask]
        yy = shape_labels[mask]
        for method in methods:
            if method == 'lda':
                projector = LinearDiscriminantAnalysis(n_components=n_pcs).fit(XX, yy)
            elif method == 'pca1':
                projector = PCA(n_components=n_pcs).fit(XX)
            elif method == 'pca2':
                label_centroids = np.array([np.mean(vecs[shape_labels == lbl], axis=0)
                                            for lbl in set(shape_labels[mask])])
                projector = PCA(n_components=n_pcs).fit(label_centroids)
            else:
                raise ValueError()

            pc_vecs = projector.transform(vecs)
            score = silhouette_score(pc_vecs[mask], yy)
            results.append((pop_name, method, pc_vecs, score))

    return results



def _round_spec(spec: dict) -> dict:
    e = round(round(spec['e'] / .05) * .05, 3)
    bias = round(round(spec['bias'] / .05) * .05, 3)
    return {'kind': spec['kind'], 'e': e, 'bias': bias}


def _load_specs(file_basename, nbest):
    json_files = glob(file_basename + "*.json")
    items = []
    for json_file in json_files:
        items += json.load(open(json_file, 'r'))
    df = pd.DataFrame(items)
    sorted_ixs = df['minority_score'].to_numpy().argsort()[::-1]
    ret = []
    for ix in sorted_ixs[:nbest]:
        ret.append({k: ShapeSpec(**v) for k, v in df.iloc[ix]['shape_specs'].items()})
    return ret, sorted_ixs[:nbest]
#
#
# def draw_projs(vecs, labels):
#     from sklearn.decomposition import PCA, FastICA
#     from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#     from sklearn.manifold import MDS
#     from sklearn.cross_decomposition import CCA
#     from sklearn.cross_decomposition import PLSRegression
#     from sklearn.preprocessing import StandardScaler
#
#     # Standardize the data
#     scaler = StandardScaler()
#     vecs_scaled = scaler.fit_transform(vecs)
#     vecs_scaled = vecs
#
#     # Initialize the methods
#     methods = {
#         'PCA': PCA(n_components=2),
#         'PCA2': PCA(n_components=2),
#         'LDA': LDA(n_components=2),
#         'MDS': MDS(n_components=2, normalized_stress='auto'),
#         'ICA': FastICA(n_components=2),
#         'PLS': PLSRegression(n_components=2)
#     }
#     label_centroids = np.array([np.mean(vecs_scaled[lbl == labels], axis=0) for lbl in set(labels)])
#
#     # Container for scores and projections
#     scores = {}
#     projections = {}
#
#     # Perform the projections and calculate scores
#     for method_name, method in methods.items():
#         if method_name == 'LDA':
#             projection = method.fit_transform(vecs_scaled, labels.ravel())
#         elif method_name == 'PCA2':
#             projection = method.fit(label_centroids).transform(vecs_scaled)
#         elif method_name == 'PLS':
#             labels_reshaped = StandardScaler().fit_transform(labels.reshape(-1, 1))
#             projection = method.fit_transform(vecs_scaled, labels_reshaped)[0]
#         else:
#             projection = method.fit_transform(vecs_scaled)
#
#         score = silhouette_score(projection, labels)
#         scores[method_name] = score
#         projections[method_name] = projection
#
#     # Visualize the results
#     fig, axes = plt.subplots(1, len(methods), figsize=(20, 5))
#     for i, (method_name, projection) in enumerate(projections.items()):
#         scatter = axes[i].scatter(projection[:, 0], projection[:, 1], c=labels, cmap='jet', s=50)
#         axes[i].set_title(f'{method_name} (Score: {scores[method_name]:.2f})')
#         axes[i].set_xlabel('Component 1')
#         axes[i].set_ylabel('Component 2')
#
#     print(scores)


def draw_projs(projs: list[tuple], ixs_of_shape: dict, density_type: str = 'ellipse'):
    pop_names = list(set(proj[0] for proj in projs))
    methods = sorted(set(proj[1] for proj in projs))
    colors = dict(zip(ixs_of_shape, plotting.get_nice_colors()))

    axs = plotting.named_subplots(rows=pop_names, cols=methods)
    plotting.set_outter_labels(axs, y=pop_names, t=methods)
    for (pop_name, method, pc_vecs, score) in projs:
        ax = axs[(pop_name, method)]
        for shape, seg_ixs in ixs_of_shape.items():
            color = colors[shape]
            if density_type == 'ellipse':
                plotting.plot_2d_gaussian_ellipse(pc_vecs[seg_ixs], ax=ax, edgecolor=color, facecolor='none', lw=1)
            elif density_type == 'kde':
                sns.kdeplot(x=pc_vecs[seg_ixs, 0], y=pc_vecs[seg_ixs, 1], ax=ax, color=color, fill=False)
            ax.scatter(*pc_vecs[seg_ixs].T, alpha=.5, label=shape, color=color)
        plotting.set_axis_equal(ax)
        ax.legend()


def draw_shape_embeddings(model_file, n_pcs: int = 2, density_type: str = 'ellipse',
                          n: int = 10, n_to_draw: int = None):

    assert n_pcs in (2, 3)
    assert density_type in ('kde', 'ellipse', 'none')

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data, persist=True)
    input_vecs, _ = data_mgr.get_inputs()
    segments = data_mgr.load_segments()
    neural_pop = NeuralPopulation.from_model(model_file)

    conics, scores_df = data_mgr.load_fitted_conics()
    valid_ixs = get_valid_seg_ixs(scores_df)
    seg_groups = calc_stratification_labels(segments, n=n, by='EuSpd', include_ixs=valid_ixs)

    df = pd.DataFrame(json.load(specs_file.open('r')))
    scores = df.groupby('specs')['lda_score'].min()
    sorted_specs = scores.argsort()[::-1].index
    shape_specs = json.loads(sorted_specs[2])
    shape_specs = {k: ShapeSpec(**v) for k, v in shape_specs.items()}
    print("Using Specs:")
    print(shape_specs)

    ixs_of_shape = match_segments_to_shapes(conics=conics, segment_labels=seg_groups, shape_specs=shape_specs, n=n)
    shape_labels = index_groups_to_labels(ixs_of_shape, n=len(segments))
    projs = compute_projections(model, input_vecs, neural_pop,
                                shape_labels=shape_labels, pop_names=[NEURAL_POP.MAJORITY, NEURAL_POP.MINORITY])

    if n_to_draw is not None:
        for shape_name, ixs in ixs_of_shape.items():
            x = input_vecs[ixs]
            mu = x.mean(axis=0)
            dists = np.sum((x - mu) ** 2, axis=1)
            ixs_of_shape[shape_name] = [ixs[i] for i in np.argsort(dists)[:n_to_draw]]

    draw_projs(projs, ixs_of_shape, density_type='kde')
    plt.suptitle(data_mgr.cfg.trials.name)


def seek_specs(model_file, n_pcs: int = 2, n: int = 30):

    assert n_pcs in (2, 3)
    specs_json = str(specs_file) + '.SEEK'

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data, persist=True)
    input_vecs, _ = data_mgr.get_inputs()
    segments = data_mgr.load_segments()
    neural_pop = NeuralPopulation.from_model(model_file)

    conics, scores_df = data_mgr.load_fitted_conics()
    valid_ixs = get_valid_seg_ixs(scores_df)
    seg_groups = calc_stratification_labels(segments, n=n, by='EuSpd', include_ixs=valid_ixs)

    spec_candidates = list(yield_spec_candidates())
    if os.path.isfile(specs_json):
        items = json.load(open(specs_json, 'r'))
        existing_specs = set(item['specs'] for item in items if item['dataset'] == data_mgr.cfg.trials.name)
    else:
        items = []
        existing_specs = set()

    for spec_ix, shape_specs in enumerate(spec_candidates):
        specs_str = json.dumps({k: v.to_dict() for k, v in shape_specs.items()})
        if specs_str in existing_specs:
            continue
        # -----
        ixs_of_shape = match_segments_to_shapes(
            conics=conics, segment_labels=seg_groups, shape_specs=shape_specs, n=n)
        shape_labels = index_groups_to_labels(ixs_of_shape, n=len(segments))
        projs = compute_projections(model, input_vecs, neural_pop,
                                    shape_labels=shape_labels, pop_names=[NEURAL_POP.MINORITY])

        # -----
        item = {'dataset': data_mgr.cfg.trials.name, 'specs': specs_str}
        for proj in projs:
            _, method, _, score = proj
            item[f'{method}_score'] = score
        items.append(item)
        print(f"{spec_ix}/{len(spec_candidates)}", item)
        dump_every = 20
        if (len(items) % dump_every == 0) or (spec_ix == len(spec_candidates) - 1):
            json.dump(items, open(specs_json, 'w'))
            print("DUMPED")
            if len(items) < 2 * dump_every:
                assert len(json.load(open(specs_json, 'r'))) == len(items)
                print("! TESTED")


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
    for monkey in ['RS', 'RJ']:
        model_file = cv_results_mgr.get_chosen_model_file(monkey)
        draw_shape_embeddings(model_file)
    plt.show()
