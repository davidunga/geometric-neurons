"""
Dictionary related tools
"""

from itertools import product
from copy import deepcopy
from typing import Callable
import pandas as pd


def dict_product_from_grid(d: dict, grid_suffix: str = '__grid'):
    d = flatten_dict(d)
    new_d = {}
    for k, v in d.items():
        if k.endswith(grid_suffix):
            assert isinstance(v, list)
            new_d[k[:-len(grid_suffix)]] = v
        else:
            new_d[k] = [v]
    for d in dict_product(new_d):
        yield unflatten_dict(d)


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


def update_nested_dict_(d: dict, keys: list, val, allow_new: bool = False):
    """
    Performs d[keys[0]][keys[1]][..] = val
    Args:
        d: Dictionary
        keys: List of nested keys
        val: Value to assign
        allow_new: Allow creating new keys?
    """
    for k in keys[:-1]:
        if allow_new and (k not in d):
            d[k] = {}
        d = d[k]
    if not allow_new:
        assert keys[-1] in d
    d[keys[-1]] = val


def flatten_dict(d, sep='.') -> dict:
    """ Flatten a nest dict by concatenating nested keys """
    return pd.json_normalize(d, sep=sep).to_dict(orient='records')[0]


def unflatten_dict(d, sep='.') -> dict:
    ret = {}
    for k, v in d.items():
        update_nested_dict_(ret, k.split(sep), v, allow_new=True)
    return ret


def find_key_in_dict(d: dict, key):
    """ Return paths (list of lists) to all occurrences of key in dict. """
    sep = '.'
    d_flat = flatten_dict(d, sep=sep)
    paths = []
    for k in d_flat:
        if k.endswith(sep + key):
            paths.append(k.split(sep))
    return paths


def dict_sort(d):
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



if __name__ == "__main__":
    d = {"a": {"aa": [11, 12], "bb__grid": [22, 23]}, "b__grid": [2,3]}
    print(list(dict_product_from_grid(d)))
