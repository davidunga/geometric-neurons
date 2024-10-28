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
from common.utils import dictools


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
                          shape_specs: dict[str, ShapeSpec], n: int = 50,
                          dist_thresh: float = float('inf')) -> dict[str, list[int]]:

    assert segment_labels.max() == n
    valid_ixs = np.nonzero(segment_labels)[0]
    segment_labels = segment_labels[valid_ixs]

    valid_conic_specs = [ShapeSpec.from_conic(conics[i]) for i in valid_ixs]

    dists2 = np.zeros((len(shape_specs), len(valid_ixs)))
    for i, spec in enumerate(shape_specs.values()):
        for j, conic_spec in enumerate(valid_conic_specs):
            dists2[i, j] = spec.dist2(conic_spec)

    ixs_of_shape = {spec_name: [] for spec_name in shape_specs}
    dists_of_shape = {spec_name: [] for spec_name in shape_specs}
    for stratify_label in range(1, n + 1):
        label_ixs = np.nonzero(segment_labels == stratify_label)[0]
        for i, (name, spec) in enumerate(shape_specs.items()):
            j = label_ixs[np.argmin(dists2[i][label_ixs])]
            if dists2[i, j] < dist_thresh ** 2:
                dists_of_shape[name].append(dists2[i, j] ** .5)
                ixs_of_shape[name].append(valid_ixs[j])
                dists2[:, j] = np.inf

    for shape_name, shape_dists in dists_of_shape.items():
        print(f"{shape_name}: count={len(shape_dists)}, "
              f"dists avg={np.mean(shape_dists):2.1f} sd={np.std(shape_dists):2.1f}")

    return ixs_of_shape


def seek_specs():

    import analysis_optim

    n_pcs = 2
    top_k = 5
    ns = [10, 20]
    dump_every = 20

    popspec_scores = analysis_optim.load_scores()
    model_files = popspec_scores.iloc[:top_k][['RS_file', 'RJ_file']].to_numpy().flatten().tolist()
    model_files = sorted(set(model_files))

    def _get_popspec(model_file, rank_):
        df = popspec_scores.loc[(popspec_scores['RS_file'] == model_file) | (popspec_scores['RJ_file'] == model_file)]
        return df.iloc[rank_]['pop_spec']

    models, cfgs = cv_results_mgr.group_models_by_config(model_files)

    def _yield_spec_candidates():
        bias_grid = np.round(np.linspace(-2, 2, 7), 2)
        e_grid = np.array([0.55, 0.65, 0.7, 0.75, 0.8, 0.85, 0.95])
        for kinds in (['e', 'p', 'p'], ['p', 'e', 'e']):
            if kinds == ['p', 'e', 'e']:
                continue
            es = tuple(e_grid if kind == 'e' else [1] for kind in kinds)
            for b1, b2, b3, e1, e2, e3 in product([0], bias_grid[bias_grid >= 0], bias_grid[bias_grid < 0], *es):
                s1 = ShapeSpec(kind=kinds[0], e=e1, bias=b1)
                s2 = ShapeSpec(kind=kinds[1], e=e2, bias=b2)
                s3 = ShapeSpec(kind=kinds[2], e=e3, bias=b3)
                spec_cand = {s1.name: s1, s2.name: s2, s3.name: s3}
                yield spec_cand

    spec_candidates = list(_yield_spec_candidates())
    total_models_count = sum(len(v) for v in models.values())
    count = 0

    existing_items = []
    if analysis_optim.proj_scores_file.is_file():
        existing_items = json.load(analysis_optim.proj_scores_file.open('r'))
    items = list(existing_items)

    for cfg_id in cfgs:

        cfg = cfgs[cfg_id]
        data_mgr = DataMgr(cfg.data, persist=True)
        input_vecs, _ = data_mgr.get_inputs()
        segments = data_mgr.load_segments()
        conics, scores_df = data_mgr.load_fitted_conics()
        valid_ixs = get_valid_conic_ixs(scores_df)

        ixs_of_shapes = {}
        for n in ns:
            seg_groups = segment_processing.digitize_segments(segments, n=n, by='EuSpd', include_ixs=valid_ixs)
            for spec_ix in range(len(spec_candidates)):
                ixs_of_shapes[(spec_ix, n)] = match_conics_to_specs(
                    conics=conics, segment_labels=seg_groups, shape_specs=spec_candidates[spec_ix], n=n)

        for model_file, model in models[cfg_id]:
            count += 1
            print(f"{count}/{total_models_count} {model_file}")

            for score_rank in [0, 1, 2]:

                try:
                    neural_pop = NeuralPopulation.from_model(model_file, spec=_get_popspec(model_file, score_rank))
                except:
                    continue

                for spec_ix, n in product(list(range(len(spec_candidates))), ns):

                    shape_specs_str = json.dumps({k: v.to_dict() for k, v in spec_candidates[spec_ix].items()})
                    item = {'dataset': data_mgr.cfg.trials.name,
                            'shape_specs': shape_specs_str,
                            'pop_spec': neural_pop.spec_str,
                            'n_projs': n,
                            'model_file': model_file}

                    if any(dictools.is_partially_equal(item, existing_item) for existing_item in existing_items):
                        continue

                    ixs_of_shape = ixs_of_shapes[(spec_ix, n)]

                    projs = segment_processing.compute_projections(
                        model, input_vecs, neural_pop, groups=ixs_of_shape,
                        pop_names=[NEURAL_POP.MAJORITY, NEURAL_POP.MIDMAJ, NEURAL_POP.MINORITY])

                    # -----

                    for proj in projs:
                        pop_label, method, _, score = proj
                        item[str(pop_label) + '_' + method] = score
                    items.append(item)

                    if len(items) % dump_every == 0:
                        json.dump(items, analysis_optim.proj_scores_file.open('w'))
                        print("DUMPED")

    json.dump(items, analysis_optim.proj_scores_file.open('w'))
    print("DONE.")


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


if __name__ == "__main__":
    seek_specs()
