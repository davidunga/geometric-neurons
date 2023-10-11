"""
Framework for grid-searching the configuration params to find an "optimal" model
The core functionality here should be viewed as scripts, to be modified manually for each run
"""

from config import get_default_config, make_analysis_name, get_cfg_dataset_name, output_dir
import json
from evaluation import eval_model_file
from common.pairwise.train import train
from utils.utils import hash_id, part2pcnt
from utils import dictools
import os
from copy import deepcopy
import pandas as pd
from glob import glob
import numpy as np
from itertools import compress
import shutil
from datetime import datetime
import stat
from pathlib import Path
from collections import defaultdict
import time

gridsearch_dir = output_dir + '/grid_search'
gridsearch_overview_dir = output_dir + '/gridsearch_overview'
_default_score_col = 'results.test.acc'

# --------------------------------------------------------------
# helper functions:


def _get_hash_from_fn(fname):
    """ Parse a grid file to get the config hash """
    return os.path.basename(fname)[:-5].split('_')[-1]


def validate_and_get_hash(cfg_items, fname):
    """
    Given cfg and its corresponding filename, return the cfg's hash after validating its
        consistent with the hash in the filename
    """
    cfg_hash = _get_hash_from_fn(fname)
    assert cfg_hash == hash_id(cfg_items), f"{cfg_hash} vs {hash_id(cfg_items)} In {fname}"
    return cfg_hash


def _evaluate_cfg(cfg, jsonlike_model_fname=False, dump_json=True):
    """
    Train and evaluate a model based on cfg
    :param cfg: cfg items to use
    :param jsonlike_model_fname: if True, the model filename is changed to be like the grid-file json filename, i.e.
        with the scores and cfg hash. If False (default), the generic model filename is used
    :param dump_json: flag- dump result to grid file
    :return:
        cfg_eval_items - a dict of the cfg used, and the evaluation result
        model_file - file where the model was saved to
    """

    model_file = train(cfg=cfg, tbwrt=False)
    results = eval_model_file(model_file)
    cfg_eval_items = {
        'cfg': cfg,
        'results': {}
    }
    for result in results:
        cfg_eval_items['results'][result.split.name.lower()] = {
            'acc': result.acc,
            'auc': result.auc,
            'rnk_rho': result.rnk_rho,
            'rnk_pvl': result.rnk_pvl,
        }
    assert set(cfg_eval_items['results'].keys()) == {'train', 'test'}
    if dump_json:
        grid_fn = gridsearch_dir + os.path.sep + _make_cfg_eval_fname(cfg_eval_items)
        json.dump(cfg_eval_items, open(grid_fn, 'w'), indent=4)
    if jsonlike_model_fname:
        model_dir = os.path.dirname(model_file)
        jsonlike_model_file = model_dir + os.path.sep + _make_cfg_eval_fname(cfg_eval_items)[:-5] + '.pth'
        shutil.copy(model_file, jsonlike_model_file)
        model_file = jsonlike_model_file
    return cfg_eval_items, model_file


def load_grid_files_to_df(drop_non_grid_cols=True, sort_by=None, filt=None, single_analysis=True):
    """
    Load grid json files to dataframe.
    :param drop_non_grid_cols: drop columns where values are the same for all files
    :param sort_by: column to sort by, default = by the default score column
    :param filt: dictionary of the form {column_name1: column_value1, ...} to filter by
    :param single_analysis: flag, make sure output contains data for exactly one analysis
    :return: dataframe, index = config hash

    example:
        df = load_grid_files_to_df(filt={'cfg.target_ftr': 'TrjAffine_dists', 'dataset': 'RS-200ms'})
    """

    sort_by = _default_score_col if sort_by is None else sort_by
    json_files = glob(gridsearch_dir + '/*.json')
    items_per_json = []
    cfg_hashes = []
    for json_file in json_files:
        items = json.load(open(json_file, 'r'))
        cfg_hash = validate_and_get_hash(items['cfg'], json_file)
        items['cfg_hash'] = cfg_hash
        items['fname'] = json_file
        items['dataset'] = get_cfg_dataset_name(items['cfg'])
        items['analysis_name'] = make_analysis_name(items['cfg'])
        items['mtime'] = time.ctime(Path(json_file).stat().st_mtime)
        items_per_json.append(items)
        cfg_hashes.append(cfg_hash)
    df = pd.json_normalize(items_per_json)
    df = df.set_index('cfg_hash')
    if filt is not None:
        for col, val in filt.items():
            df = df[df[col] == val]

    if single_analysis:
        assert df['analysis_name'].nunique() == 1, "Filters do not uniquely determine analysis."

    df.drop(columns=[col for col in df.columns if col.startswith('_')], inplace=True)

    if drop_non_grid_cols:
        df = df[df.columns[df.nunique() > 1]]

    df.sort_values(by=[sort_by], ascending=False, inplace=True)
    return df


def grid_dict(base_items, grid_items, sep='.'):
    """
        Yield dicts based on grid items.
        base_items: base dictionary.
        grid: flat dictionary of lists, indicating how to modify values of base_items.
            nested keys are separated by ":".
        example:
        grid = {
            "data:rand_seed": [0, 2],
            "data:p_train": [0.5, 0.7],
            "model:params:size":[100, 200, 300]
        }
    """

    for grid_pt in dictools.dict_product(grid_items):
        items = deepcopy(base_items)
        for grid_k in grid_pt:
            dictools.update_nested_dict_(items, grid_k.split(sep), grid_pt[grid_k], allow_new=False)
        yield items


def _make_cfg_eval_fname(grid_eval_items):
    cfg = grid_eval_items['cfg']
    cfg_name = make_analysis_name(cfg)
    cfg_hash = hash_id(cfg)
    res = grid_eval_items['results']['test']
    return "{:s}_acc{:02d}-rho{:02d}_{:s}.json".format(
        cfg_name, part2pcnt(res['acc']), part2pcnt(res['rnk_rho']), cfg_hash
    )


def pack_grid_files():
    json_files = glob(gridsearch_dir + '/*')
    print(f"Preparing grid files package ({len(json_files)} files) ...")
    packed_items = []
    for json_file in json_files:
        file_stat = os.stat(json_file)
        meta = {
            'fname': os.path.basename(json_file),
            'ctime': file_stat[stat.ST_CTIME],
            'mtime': file_stat[stat.ST_MTIME]
        }
        content = json.load(open(json_file, 'r'))
        packed_items.append({'meta': meta, 'content': content})
    fn = gridsearch_overview_dir + '/gridfiles_package.json'
    json.dump(packed_items, open(fn, 'w'), indent=4)
    print("Done. Saved grid files package to " + fn)


def update_summary_file():

    def _format_row(row):
        return f" acc={row['acc']}  rho={row['rho']}  {os.path.basename(row['fname'])}  {row['mtime']}"

    df = load_grid_files_to_df(single_analysis=False, drop_non_grid_cols=False)
    n_top = 20
    title = "Grid Search Summary from " + datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + "\n"
    summary_cols = {'results.test.rnk_rho': 'rho', 'results.test.acc': 'acc', 'fname': 'fname', 'mtime': 'mtime'}
    brief = ""
    detailed = ""
    for analysis_name in df['analysis_name'].unique():
        analysis_df = df[df['analysis_name'] == analysis_name][list(summary_cols.keys())]
        analysis_df = analysis_df.rename(columns=summary_cols)
        analysis_df = analysis_df.round(decimals=2)

        analysis_brief = "{:50s}".format(f"\n{analysis_name} [{len(analysis_df)} files]")
        analysis_brief += " Best acc={:2.2f} rho={:2.2f}".format(analysis_df['acc'].max(), analysis_df['rho'].max())
        analysis_brief += "  | Top-10% acc={:2.2f} rho={:2.2f}".format(
            np.percentile(analysis_df['acc'], 90), np.percentile(analysis_df['rho'], 90))
        brief += analysis_brief

        detailed += analysis_brief

        detailed += "\ntop acc:\n"
        for ix in analysis_df['acc'].argsort()[::-1][:n_top]:
            detailed += _format_row(analysis_df.iloc[ix]) + "\n"

        detailed += "top rho:\n"
        for ix in analysis_df['rho'].argsort()[::-1][:n_top]:
            detailed += _format_row(analysis_df.iloc[ix]) + "\n"

    lines = title + "\n--Brief:" + brief + "\n\n--Details:\n" + detailed

    os.makedirs(gridsearch_overview_dir, exist_ok=True)
    with open(gridsearch_overview_dir + '/summary.txt', 'w') as f:
        f.writelines(lines)

# --------------------------------------------------------------
# Scripts:


def update_grid_files():

    backup_dir = gridsearch_dir + '_backups/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(backup_dir)

    json_files = glob(gridsearch_dir + '/*')
    print(f"Updating {len(json_files)} grid files. Backups saved to: {backup_dir}")

    for json_file in json_files:
        shutil.copy(json_file, backup_dir)

        items = json.load(open(json_file, 'r'))
        new_fn = gridsearch_dir + os.path.sep + _make_cfg_eval_fname(items)

        json.dump(items, open(new_fn, 'w'), indent=4)

        if new_fn != json_file:
            print(json_file, "->", new_fn)
            os.remove(json_file)

    update_summary_file()
    print("Done.")


def _make_randneural_of_best_model():
    df = load_grid_files_to_df(filt={
        'cfg.target_ftr': 'TrjAffine_dists',
        'dataset': 'RS-200ms'}
    )
    fn = df.iloc[0]['fname']
    print(f"Making random neural of cfg: {fn}")
    items = json.load(open(fn, 'r'))
    assert not items['cfg']['data']['random_neural']
    items['cfg']['data']['random_neural'] = True
    eval_items, model_file = _evaluate_cfg(items['cfg'], jsonlike_model_fname=True, dump_json=True)
    print(f"Saved to: {model_file}")


def _eval_and_save_model_of_grid_file(grid_fn):
    if not os.path.isfile(grid_fn):
        df = load_grid_files_to_df(single_analysis=False)
        grid_fn = df.loc[grid_fn]['fname']
    print(f"Training model for cfg: {grid_fn}")
    items = json.load(open(grid_fn, 'r'))
    cfg_hash = validate_and_get_hash(items['cfg'], grid_fn)
    eval_items, model_file = _evaluate_cfg(items['cfg'], jsonlike_model_fname=True, dump_json=False)
    assert eval_items == items
    print(f"Trained model for {cfg_hash}. Saved to: {model_file}")


def find_singlemost_affecting_col():
    df = load_grid_files_to_df()
    score_col = _default_score_col

    cfg_cols = [col for col in df if col.startswith('cfg.')]
    res_cols = [col for col in df if col.startswith('results.')]

    best_idx = df[score_col].idxmax()
    best_cfg = df.loc[best_idx]

    argmax_cfg = None
    argmax_col = None
    max_score_diff = 0
    for i in range(len(df)):
        ii = np.array(df[cfg_cols].iloc[i] != best_cfg[cfg_cols])
        if ii.sum() == 1:
            score_diff = best_cfg[score_col] - df[score_col].iloc[i]
            if score_diff > max_score_diff:
                max_score_diff = score_diff
                argmax_col = cfg_cols[np.nonzero(ii)[0][0]]
                argmax_cfg = df.iloc[i]
    print("For baseline:")
    print(best_cfg[cfg_cols])
    print("Max score diff occurs for col=" + argmax_col)
    print(f"{argmax_col}={best_cfg[argmax_col]} yields score:" + "{:s}={:2.5}".format(score_col, best_cfg[score_col]))
    print(f"{argmax_col}={argmax_cfg[argmax_col]} yields score:" + "{:s}={:2.5}".format(score_col, argmax_cfg[score_col]))
    idxmax = df[df[argmax_col] == argmax_cfg[argmax_col]][score_col].idxmax()
    print(f"The max score where {argmax_col}={argmax_cfg[argmax_col]} is ", df.loc[idxmax][score_col], "For:")
    print(df.loc[idxmax])


def show_best():
    pd.options.display.max_columns = 50
    pd.options.display.width = 200
    score_col = _default_score_col
    score_col = 'results.test.rnk_rho'

    main_flt = {
            'cfg.target_ftr': 'TrjAffine_dists',
            'cfg.data.random_neural': False,
            'cfg.data.ctrls_as_neural': True,
            'dataset': 'RS-200ms',
            'cfg.data.ctrl_kind': 'rel',
            'cfg.data.equalize_ctrls': True,
    }
    specific_flt = {
        #'cfg.model.descriptor_size': 5,
        #'cfg.data.neural_filt_sz': 3,
        #'cfg.model.dropout': 0.5,
        'cfg.data.exclude_reflected': True,
        #'cfg.data.scale_range.min': 0,
    }
    flt = {**specific_flt, **main_flt}

    df = load_grid_files_to_df(sort_by=score_col, filt=flt, single_analysis=False)

    keep_cols = ['mtime']
    df = df.drop(columns=[col for col in df
                          if col not in keep_cols and '.' not in col])
    df['nrm_scr'] = df[score_col] / df[score_col][0]

    PRINT_UNQS = False
    n = 10
    if PRINT_UNQS:
        print("Unique values:")
        for col in df.columns:
            if col.startswith('cfg'):
                print(col, ":", df[col].unique())

    df = df.rename(columns={col: col.split('.')[-1] if col.startswith('cfg') else col.replace('results.', '')
                            for col in df})

    print("\nBest:")
    print(df.iloc[:n])
    print("\nWorst:")
    print(df.iloc[-n:])


def cleanup_grid_files(dryrun=True):

    if not dryrun:
        backup_dir = gridsearch_dir + '_backups/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(backup_dir)

    json_files = glob(gridsearch_dir + '/*.json')

    files_by_hash = defaultdict(list)
    for ix in np.argsort([Path(json_file).stat().st_mtime for json_file in json_files]):
        json_file = json_files[ix]
        items = json.load(open(json_file, 'r'))
        cfg_hash = validate_and_get_hash(items['cfg'], json_file)
        files_by_hash[cfg_hash].append(json_file)

        if not dryrun:
            shutil.copy(json_file, backup_dir)

    for cfg_hash in files_by_hash:
        if len(files_by_hash[cfg_hash]) == 1:
            continue
        files = files_by_hash[cfg_hash]
        print(cfg_hash + ":")
        print(" Keeping:  " + os.path.basename(files[-1]), f"[{time.ctime(Path(files[-1]).stat().st_mtime)}]")
        print(" Removing: ", end="")
        for fn in files[:-1]:
            print(os.path.basename(fn), f"[{time.ctime(Path(fn).stat().st_mtime)}]", end=" ")
            if not dryrun:
                os.remove(fn)
        print("")

    if not dryrun:
        n_before = len(json_files)
        n_after = len(glob(gridsearch_dir + '/*.json'))
        print(f"Removed {n_before - n_after} files [{n_before} -> {n_after}]")
        print("Backed saved to " + backup_dir)
    else:
        print("[DRY RUN, no files removed.]")


def check_unique_cfgs():
    json_files = glob(gridsearch_dir + '/*.json')
    grid_cfgs = []
    grid_cfg_hashs = []
    for json_file in json_files:
        items = json.load(open(json_file, 'r'))
        grid_cfg_hashs.append(validate_and_get_hash(items['cfg'], json_file))
        grid_cfgs.append(items['cfg'])
    n_unq = len(set(grid_cfg_hashs))
    n_tot = len(grid_cfg_hashs)
    print(f"Total hashes=  {n_tot}")
    print(f"Unique hashes= {n_unq}")
    print("All good." if n_unq == n_tot else "Somethings not right.")


def grid_search():
    os.makedirs(gridsearch_dir, exist_ok=True)
    skip_existing = True

    # -------
    # get cfgs to work on:

    grid_base_mode = 'default'
    assert grid_base_mode in ('default', 'modify')

    if grid_base_mode == 'default':

        base_cfg = get_default_config()
        grid = {
            "model.descriptor_size": [3, 5, 7],
            "model.dropout": [0.5, 0.75, 0.8],
            "data.balance": [True],
            "data.neural_filt_sz": [3, 5],
            "data.scale_range.min": [0],
            "data.exclude_reflected": [True, False],
            "data.p_train": [0.7, 0.8],
            "train.optimizer_params.lr": [1e-05, 5e-05, 1e-04],
            "data.equalize_ctrls": [True],
            "data.ctrl_kind": ["rel"],
            "data.ctrls_as_neural": [False]
        }
        grid_cfgs = list(grid_dict(base_cfg, grid))
        print(f"Grid search over {len(grid)} parameters, {len(grid_cfgs)} configurations")

    elif grid_base_mode == 'modify':
        json_files = glob(gridsearch_dir + '/RS-200ms_TrjAffineDists_acc*.json')
        mod_cfgs = []
        current_scores = []
        jsons_to_modify = []
        cfg_hashes = []
        for json_file in json_files:

            # get and validate current
            items = json.load(open(json_file, 'r'))
            validate_and_get_hash(items['cfg'], json_file)

            # modify
            if not items['cfg']['data']['equalize_ctrls']:
                continue
            if items['cfg']['data']['ctrl_kind'] != 'rel':
                continue
            if items['cfg']['data']['ctrls_as_neural']:
                continue
            #items['cfg']['data']['equalize_ctrls'] = True
            #items['cfg']['data']['ctrl_kind'] = 'rel'
            #items['cfg']['data']['ctrls_as_neural'] = True

            cfg_hash = hash_id(items['cfg'])
            if cfg_hash in cfg_hashes:
                print("Two jsons yield same hash after mod:")
                print(os.path.basename(json_file))
                print(os.path.basename(jsons_to_modify[cfg_hashes.index(cfg_hash)]))
                print("quitting..")
                return

            # append
            jsons_to_modify.append(json_file)
            mod_cfgs.append(items['cfg'])
            cfg_hashes.append(cfg_hash)
            current_scores.append(items['results']['test']['acc'])

        grid_cfgs = [mod_cfgs[i] for i in np.argsort(current_scores)[::-1]]
        print(f"Grid search modifying {len(grid_cfgs)} files")

    else:
        raise ValueError

    # -------

    grid_cfg_hashes = [hash_id(cfg) for cfg in grid_cfgs]

    if skip_existing:
        existing_cfg_hashes = load_grid_files_to_df(single_analysis=False).index.tolist()
        keep_ixs = [cfg_hash not in existing_cfg_hashes for cfg_hash in grid_cfg_hashes]
        grid_cfgs = list(compress(grid_cfgs, keep_ixs))
        grid_cfg_hashes = list(compress(grid_cfg_hashes, keep_ixs))

        print(f"Skipped {len(keep_ixs) - len(grid_cfgs)} existing. {len(grid_cfgs)} left to go.")

    assert len(set(grid_cfg_hashes)) == len(grid_cfg_hashes)

    for cfg_ix, cfg in enumerate(grid_cfgs):
        print(f"\nGrid cfgs {cfg_ix}/{len(grid_cfgs) - 1}:")
        cfg_eval_items = _evaluate_cfg(cfg, dump_json=True)
        update_summary_file()


if __name__ == "__main__":
    # df = load_grid_files_to_df(filt={
    #     'cfg.target_ftr': 'TrjAffine_dists',
    #     'cfg.data.random_neural': False,
    #     'dataset': 'RS-200ms'}
    # )
    # print(df.iloc[0]['fname'])
    #_eval_and_save_model_of_grid_file('81a107')
    #_eval_and_save_best_model()
    #_make_randneural_of_best_model()
    #update_grid_files()
    #pack_grid_files()
    #show_best()
    #cleanup_grid_files(dryrun=True)
    #check_unique_cfgs()
    grid_search()
    #update_summary_file()

