"""
Dictionary tools
"""

from copy import deepcopy
from itertools import product, chain
from typing import Callable
import datetime
import numpy as np
import pandas as pd


def unpack_nested_dict(d: dict) -> tuple[list, list]:
    """
    Args:
        d: nest dict
    Returns:
        key_paths: list of key paths
        values: list same length as key_paths, values[i] is the value of key_paths[i]
    Example:
        d = {
            'Alice': {'city': 'LA', 'id': 123},
            'Bob': {'city': 'NY', 'id': 567}
        }
        ->
        key_paths:
            [
                ['Alice', 'city'],
                ['Alice', 'id'],
                ['Bob', 'city'],
                ['Bob', 'id']
            ]
        values:
            ['LA', 123, 'NY', 567]
    """
    key_paths = []
    values = []
    for key, value in d.items():
        if isinstance(value, dict):
            for path, path_value in zip(*unpack_nested_dict(value)):
                key_paths.append([key] + path)
                values.append(path_value)
        else:
            key_paths.append([key])
            values.append(value)
    return key_paths, values


def dict_product_from_grid(grid: dict, base_dict: dict = None):
    """
    Yield product of list values in grid dict.
    All list values are regarded as grid points.
    """

    flat_grid, sep = flatten_dict_autosep(grid)
    for k, v in flat_grid.items():
        flat_grid[k] = [v] if not isinstance(v, list) else v
    for flat_dict in dict_product(flat_grid):
        d = unflatten_dict(flat_dict, sep=sep)
        if base_dict:
            d = deep_merge(deepcopy(base_dict), d)
        yield d


def variance_dicts(dicts: list[dict], flat: bool = True, force_keep: list = None):
    """
        ignore keys where all dicts have the same value
        force_keep: list of (flat) keys to keep even if non-variance
    """
    def _is_unique(s: pd.Series):
        try:
            return len(s.unique()) == 1
        except:
            return len(np.unique(s.to_numpy())) == 1
    flat_dicts = [flatten_dict(d) for d in dicts]
    df = pd.DataFrame.from_records(flat_dicts)
    force_keep = [] if not force_keep else force_keep
    variance_flat_keys = [col for col in df.columns if not _is_unique(df[col]) or col in force_keep]
    variance_dicts = [{k: v for k, v in d.items() if k in variance_flat_keys}
                      for d in flat_dicts]
    if not flat:
        variance_dicts = [unflatten_dict(d) for d in variance_dicts]
    return variance_dicts


def rename(d: dict, **kwargs) -> dict:
    return {kwargs.get(k, k): v for k, v in d.items()}


def inherit_values(d: dict, sep: str = '.', prefix: str = '::'):
    """ replace values of the form '::<key_path>' with d[<key_path>] """
    d_flat = flatten_dict(d, sep=sep)
    for k in d_flat:
        if isinstance(d_flat[k], str) and d_flat[k].startswith(prefix):
            key = d_flat[k][len(prefix):]
            d_flat[k] = d_flat[key]
    return unflatten_dict(d_flat, sep)


def deep_merge(d1: dict, d2: dict, keep_structure: bool = True) -> dict:
    """ recursively overwrite values in d1 from d2, if they exist in d2.
        d2 must be a recursive subset of d1.
    """

    assert set(d2.keys()).issubset(d1.keys()), "modifier is not a subset of base"

    def _get(k):
        if k not in d2:
            return d1[k]
        elif isinstance(d2[k], dict):
            return deep_merge(d1[k], d2[k])
        else:
            if keep_structure:
                assert not isinstance(d1[k], dict)
            return d2[k]

    return {k: _get(k) for k in d1}


def modify_dict(base_dict: dict, copy: bool, exclude: list = None,
                include: list = None, update_dict: dict = None):

    keys = base_dict.keys()
    if include is not None:
        keys = [k for k in keys if k in include]
    if exclude is not None:
        keys = [k for k in keys if k not in exclude]
    if update_dict:
        if not isinstance(keys, list):
            keys = list(keys)
        keys += [k for k in update_dict.keys() if k not in keys]
    else:
        update_dict = {}

    ret = {k: update_dict[k] if k in update_dict else base_dict[k] for k in keys}
    if copy:
        ret = deepcopy(ret)
    return ret


def dict_product(d):
    keys = d.keys()
    for vals in product(*d.values()):
        yield dict(zip(keys, vals))


def is_partially_equal(d1: dict, d2: dict, keys=None):
    if keys is None:
        keys = set(d1).intersection(d2)
    return all(d1[k] == d2[k] for k in keys)


def update_nested_dict(d: dict, keys: list, val, allow_new: bool = False):
    """
    Performs d[keys[0]][keys[1]][..] = val
    Args:
        d: Dictionary
        keys: List of nested keys
        val: Value to assign
        allow_new: Allow creating new keys?
    """
    for key in keys[:-1]:
        if key not in d and allow_new:
            d[key] = {}
        d = d[key]
    key = keys[-1]
    if key not in d and not allow_new:
        raise KeyError(key)
    d[key] = val


def flatten_dict(d: dict, sep='.') -> dict:
    """ Flatten a nested dict by concatenating nested keys """
    flat_dict, _ = flatten_dict_autosep(d, seps=(sep,))
    return flat_dict


def flatten_dict_autosep(d: dict, seps=('.', ':', '|')) -> tuple[dict, str]:
    """ Flatten a nested dict by concatenating nested keys
        Choose first separator which does not collide with current keys
    """
    key_paths, values = unpack_nested_dict(d)
    all_keys_str = "".join(chain(*key_paths))
    sep = None
    for maybe_sep in seps:
        if maybe_sep not in all_keys_str:
            sep = maybe_sep
            break
    assert sep is not None
    flat_keys = [sep.join(key_path) for key_path in key_paths]
    return dict(zip(flat_keys, values)), sep


def unflatten_dict(d: dict, sep='.') -> dict:
    ret = {}
    for k, v in d.items():
        update_nested_dict(ret, k.split(sep), v, allow_new=True)
    return ret


def dict_sort(d: dict):
    return dict_recursive(d, lambda d: dict(sorted(d.items())))


def dict_recursive(d: dict, fn: Callable[[dict], dict]) -> dict:
    """
    recursively apply function to dict.
    Args:
        d: dict
        fn: function dict -> dict
    """
    return {k: v if not isinstance(k, dict) else dict_recursive(v, fn)
            for k, v in fn(d).items()}


def to_json(data):
    """ convert to json-friendly format """
    if isinstance(data, dict):
        return {k: to_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [to_json(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.number):
        return data.item()
    elif isinstance(data, datetime.datetime):
        return data.isoformat()
    elif hasattr(data, '__dict__'):
        return to_json(data.__dict__)
    else:
        return data
