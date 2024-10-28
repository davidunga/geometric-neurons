import json
from common.utils import dictools
import matplotlib.pyplot as plt
from neural_population import NeuralPopulation, NEURAL_POP, get_importance_file
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from common.utils import sigproc
from data_manager import DataMgr
from common.utils.typings import *
import cv_results_mgr
from common.utils import stats
from common.utils import dlutils
from scipy.spatial.distance import squareform, pdist
from enum import Enum
from itertools import product
from common.utils.randtool import Rnd
from common.metric_learning import embedding_eval
from common.utils import strtools
import paths
from glob import glob
from common.utils import hashtools
from collections import defaultdict
from analysis import segment_shapes_tools
from analysis import segment_processing
from collections import defaultdict
from cv_results_mgr import group_models_by_config


scores_file = Path(paths.PROCESSED_DIR / 'scores.json')
proj_scores_file = paths.PROCESSED_DIR / 'proj_scores.json'


def get_projs_file(model_file) -> Path:
    return paths.PROCESSED_DIR / (Path(model_file).stem + '.PROJS.json')


def load_scores():
    items = json.load(scores_file.open('r'))
    #
    # pop_meta = {}
    # for i in range(len(items)):
    #     item = items[i]
    #     if 'pop_spec' not in item:
    #         items[i] = {
    #             'model_file': item['model_file'],
    #             'monkey': 'RJ' if 'TP_RJ' in item['model_file'] else 'RS',
    #             'pop_spec': '',
    #             'pop_label': 'FULL',
    #             'auc': item['auc'],
    #             'n_minority': -1,
    #             'p_minority': -1
    #         }
    #         continue
    #     key = (item['model_file'], item['pop_spec'])
    #     if key not in pop_meta:
    #         neural_pop = NeuralPopulation.from_model(item['model_file'], item['pop_spec'])
    #         n_minority = len(neural_pop.neurons(NEURAL_POP.MINORITY))
    #         pop_meta[key] = {
    #             'n_minority': n_minority,
    #             'p_minority': n_minority / len(neural_pop.neurons(NEURAL_POP.FULL))
    #         }
    #     item.update(**pop_meta[key])
    # json.dump(items, scores_file.open('w'))
    scores = pd.DataFrame(items)
    scores = scores.loc[scores['pop_label'] == str(NEURAL_POP.MINORITY)]
    RS_scores = scores.loc[scores['monkey'] == 'RS']
    RJ_scores = scores.loc[scores['monkey'] == 'RJ']

    def _argbest(scores_, spec):
        ii = np.nonzero(scores_['pop_spec'].to_numpy() == spec)[0]
        ix = ii[np.argmax(scores_['auc'].to_numpy()[ii])]
        return scores_.iloc[ix]

    spec_scores = []
    for spec in set(RS_scores['pop_spec']).intersection(RJ_scores['pop_spec']):
        RS_row = _argbest(RS_scores, spec)
        RJ_row = _argbest(RJ_scores, spec)
        spec_scores.append({'pop_spec': spec,
                            'RS_p_minority': RS_row['p_minority'],
                            'RJ_p_minority': RJ_row['p_minority'],
                            'RS_file': RS_row['model_file'],
                            'RS_auc': RS_row['auc'],
                            'RJ_file': RJ_row['model_file'],
                            'RJ_auc': RJ_row['auc']})
    spec_scores = pd.DataFrame(spec_scores)
    spec_scores['auc'] = spec_scores[['RS_auc', 'RJ_auc']].min(axis=1)

    p_minority_min, p_minority_max = .05, .16
    ii = (spec_scores[['RJ_p_minority', 'RS_p_minority']] >= p_minority_min).all(axis=1)
    ii = ii & (spec_scores[['RJ_p_minority', 'RS_p_minority']] <= p_minority_max).all(axis=1)

    spec_scores = spec_scores.loc[ii]
    spec_scores.sort_values(by='auc', ascending=False, inplace=True)

    proj_scores = pd.DataFrame(json.load(proj_scores_file.open('r')))
    proj_scores['DIFF_lda'] = proj_scores['MINORITY_lda'] - proj_scores['MIDMAJ_lda']
    proj_scores['key'] = proj_scores['shape_specs'] + proj_scores['n_projs'].astype(str) + proj_scores['pop_spec']
    keys = set(proj_scores['key'])

    score_items = []
    for i in range(len(spec_scores)):
        RJ_file = spec_scores.iloc[i]['RJ_file']
        RS_file = spec_scores.iloc[i]['RS_file']
        pop_spec = spec_scores.iloc[i]['pop_spec']
        proj_info = {'projScore': -np.inf}
        for key in keys:
            if not key.endswith(pop_spec):
                continue
            ii = proj_scores['key'] == key
            RS_df = proj_scores.loc[ii & (proj_scores['model_file'] == RS_file)]
            RJ_df = proj_scores.loc[ii & (proj_scores['model_file'] == RJ_file)]
            if not len(RS_df) or not len(RJ_df):
                continue
            assert len(RS_df) == len(RJ_df) == 1
            RS_score = RS_df['DIFF_lda'].item()
            RJ_score = RJ_df['DIFF_lda'].item()
            if min(RJ_score, RS_score) > proj_info['projScore']:
                proj_info.update(**proj_scores.loc[proj_scores['key'] == key, ['shape_specs', 'n_projs']].iloc[0].to_dict())
                proj_info['projScore_RS'] = RS_score
                proj_info['projScore_RJ'] = RJ_score
                proj_info['projScore'] = min(RJ_score, RS_score)

        score_items.append({**spec_scores.iloc[0].to_dict(), **proj_info})
    spec_scores = pd.DataFrame(score_items)

    i = 0
    spec_scores.iloc[i].to_dict()

    return spec_scores


def evaluate_neural_population_specs(model_files=None):
    # -----
    dump_every = 20
    max_n_pairs = 100_000
    specs_grid = {'importance_method': [],
                  'importance_power': [1, 2],
                  'split_method': ['otsu', 'iqr', 'sigmas']}
    # -----

    if model_files is None:
        model_files = glob(str(paths.MODELS_DIR / '*.pth'))
        model_files = [fn for fn in model_files if get_importance_file(fn).is_file()]

    models, cfgs = group_models_by_config(model_files)
    total_models_count = sum(len(v) for v in models.values())

    def _calc_auc_for_embedding(x_, pairs_) -> float:
        embedded_dists2 = -embedding_eval.pairs_dists2(x_, pairs=pairs_)
        return roc_auc_score(y_true=is_same, y_score=embedded_dists2)

    if scores_file.is_file():
        results = [] # json.load(scores_file.open('r'))
    else:
        results = []
    n_existed = len(results)

    def _check_exists(item: dict) -> bool:
        for res in results[:n_existed]:
            if all(res[k] == item[k] for k in item):
                return True
        return False

    last_dump = n_existed
    count = 0
    for cfg_id in cfgs:

        cfg = cfgs[cfg_id]
        monkey = cfg.data.trials.name.split('_')[-1]
        data_mgr = DataMgr(cfg.data)
        pairs_df = data_mgr.load_pairing(n_pairs=max_n_pairs)
        is_same = pairs_df['isSame'].to_numpy(dtype=int)
        pairs = pairs_df[['seg1', 'seg2']].to_numpy()
        input_vecs, inputs_meta = data_mgr.get_inputs()

        for model_file, model in models[cfg_id]:
            count += 1
            print(f"{count}/{total_models_count} {model_file}")

            result = {'model_file': model_file, 'pop': str(NEURAL_POP.FULL), **{k: None for k in specs_grid}}
            if not _check_exists(result):
                result['auc'] = _calc_auc_for_embedding(dlutils.safe_predict(model, input_vecs), pairs)
                results.append(result)

            if not specs_grid['importance_method']:
                importances_dict = json.load(get_importance_file(model_file).open('r'))['importances'][0]
                specs_grid['importance_method'] = sorted(set(importances_dict).difference(['neuron']))

            for neural_pop_spec in dictools.dict_product(specs_grid):
                neural_pop = NeuralPopulation.from_model(model_file, neural_pop_spec)
                for pop_label in [NEURAL_POP.MINORITY, NEURAL_POP.MIDMAJ, NEURAL_POP.MAJORITY]:
                    result = {'model_file': model_file, 'monkey': monkey,
                              'pop_spec': neural_pop.spec_str, 'pop_label': str(pop_label)}
                    if _check_exists(result):
                        continue
                    embedded_vecs = dlutils.safe_predict(model, neural_pop.filter_inputs(input_vecs, include=pop_label))
                    result['auc'] = _calc_auc_for_embedding(embedded_vecs, pairs)
                    results.append(result)

            if len(results) - last_dump >= dump_every:
                last_dump = len(results)
                json.dump(results, scores_file.open('w'))
                print("DUMPED")

        json.dump(results, scores_file.open('w'))
        print("DONE")


if __name__ == "__main__":
    load_scores()
    #evaluate_neural_population_specs()
