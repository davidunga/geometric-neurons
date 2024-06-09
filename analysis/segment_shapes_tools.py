import json
import os.path
import numpy as np
import pandas as pd
from itertools import product
import paths
from common.utils.shapes_specs import ShapeSpec
from data_manager import DataMgr
import cv_results_mgr
from neural_population import NeuralPopulation, NEURAL_POP
from analysis.config import DataConfig
from common.utils.conics.conic_fitting import fit_conic_ransac, Conic
from analysis import segment_processing


specs_file = paths.PROCESSED_DIR / 'shape_specs.json'


def get_chosen_shape_specs(by: str = 'lda_score', rank: int = 2, rename: bool = True):
    df = pd.DataFrame(json.load(specs_file.open('r')))
    scores = df.groupby('specs')[by].min()
    sorted_specs = scores.argsort()[::-1].index
    shape_specs = json.loads(sorted_specs[rank])
    shape_specs = {k: ShapeSpec(**v) for k, v in shape_specs.items()}
    return shape_specs


def get_valid_conic_ixs(scores_df: pd.DataFrame) -> list[int]:
    inls_thresh = .9
    mse_thresh = .005
    valid_ixs = scores_df.loc[(scores_df['inl'] > inls_thresh) & (scores_df['mse'] < mse_thresh)]['seg_ix'].tolist()
    return valid_ixs


def match_conics_to_specs(conics: list[Conic], segment_labels: np.ndarray,
                          shape_specs: dict[str, ShapeSpec], n: int = 50) -> dict[str, list[int]]:

    assert segment_labels.max() == n - 1
    valid_ixs = np.nonzero(segment_labels)[0]
    segment_labels = segment_labels[valid_ixs]

    valid_conic_specs = [ShapeSpec.from_conic(conics[i]) for i in valid_ixs]

    dists2 = np.zeros((len(shape_specs), len(valid_ixs)))
    for i, spec in enumerate(shape_specs.values()):
        for j, conic_spec in enumerate(valid_conic_specs):
            dists2[i, j] = spec.dist2(conic_spec)

    ixs_of_shape = {spec_name: [] for spec_name in shape_specs}
    dists_for_dbg = {spec_name: [] for spec_name in shape_specs}
    for stratify_label in range(1, n):
        label_ixs = np.nonzero(segment_labels == stratify_label)[0]
        for i, (name, spec) in enumerate(shape_specs.items()):
            j = label_ixs[np.argmin(dists2[i][label_ixs])]
            if np.isfinite(dists2[i, j]) and dists2[i, j] < 250000:
                dists_for_dbg[name].append(dists2[i, j])
                ixs_of_shape[name].append(valid_ixs[j])
                dists2[:, j] = np.inf

    return ixs_of_shape


def seek_specs(model_file, n_pcs: int = 2, n: int = 30):

    assert n_pcs in (2, 3)
    specs_json = str(specs_file) + '.SEEK'
    dump_every = 20

    def _yield_spec_candidates():
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

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data, persist=True)
    input_vecs, _ = data_mgr.get_inputs()
    segments = data_mgr.load_segments()
    neural_pop = NeuralPopulation.from_model(model_file)

    conics, scores_df = data_mgr.load_fitted_conics()
    valid_ixs = get_valid_conic_ixs(scores_df)
    seg_groups = segment_processing.digitize_segments(segments, n=n, by='EuSpd', include_ixs=valid_ixs)

    spec_candidates = list(_yield_spec_candidates())

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

        ixs_of_shape = match_conics_to_specs(conics=conics, segment_labels=seg_groups, shape_specs=shape_specs, n=n)
        projs = segment_processing.compute_projections(model, input_vecs, neural_pop, groups=ixs_of_shape,
                                                       pop_names=[NEURAL_POP.MAJORITY, NEURAL_POP.MINORITY])

        # -----

        item = {'dataset': data_mgr.cfg.trials.name, 'specs': specs_str}
        for proj in projs:
            _, method, _, score = proj
            item[f'{method}_score'] = score
        items.append(item)

        print(f"{spec_ix}/{len(spec_candidates)}", item)
        if (len(items) % dump_every == 0) or (spec_ix == len(spec_candidates) - 1):
            json.dump(items, open(specs_json, 'w'))
            print("DUMPED")
            if len(items) < 2 * dump_every:
                assert len(json.load(open(specs_json, 'r'))) == len(items)
                print("VERIFIED")


def calc_and_save_conic_fits(model_file):
    from common.utils.devtools import progbar
    from common.utils.conics import get_conic
    from common.utils.conics.conic_fitting import fit_conic_ransac
    from common.utils import hashtools

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data, persist=True)
    segments = data_mgr.load_segments()

    fit_kws = {'max_itrs': 500, 'thresh': .05, 'inlier_p_thresh': 1.1, 'seed': 1, 'n': 7}
    print("Computing conics for " + str(model_file))
    fit_hash = hashtools.calc_hash(fit_kws)[:6]

    conic_fits = []
    for s in progbar(segments, span=20):
        conic, scores = fit_conic_ransac(s.kin.X, **fit_kws)
        conic_fits.append({'seg_ix': s.ix, 'conic': conic.to_json(), 'scores': scores})

    conics_file = paths.PROCESSED_DIR / (data_mgr.cfg.str(DataConfig.SEGMENTS) + f'.{fit_hash}.CONICS.json')
    conics_file.parent.mkdir(exist_ok=True)
    print("Saving to " + str(conics_file))
    json.dump({'fit_kws': fit_kws, 'conic_fits': conic_fits}, conics_file.open('w'))

    # --
    print("Validating...")
    items = json.load(conics_file.open('r'))
    conics = [get_conic(**fit_result['conic']) for fit_result in items['conic_fits']]
    print("Okay. Done.")
