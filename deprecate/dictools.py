"""
Dictionary related tools
"""

from itertools import product
from copy import deepcopy
import pandas as pd


def mod_copy_dict(base_dict: dict, mod_dict: dict = None):
    """ get a modified deep-copy of base_dict """
    ret = deepcopy(base_dict)
    if mod_dict is not None:
        ret.update(mod_dict)
    return ret



def dict_product(d):
    """
    For a dictionary of lists: {k0: v0, k1: v1, ..}, (where v0, v1, .. are lists), yields
    dictionaries of all combinations of items from each list:
        {k0: v0[i], k1: v1[j], ..} for all combinations of i, j, ..
    """
    keys = d.keys()
    for vals in product(*d.values()):
        yield dict(zip(keys, vals))


def update_nested_dict_(d, keys, val, allow_new=False):
    """
    Performs: d[keys[0]][keys[1]][..] = val
    :param d: A nested dictionary
    :param keys: List of keys, where keys[i] is a key in the i-th level of the dictionary
    :param val: Value to set
    :param allow_new: allow creating new keys?
    """
    for k in keys[:-1]:
        if allow_new and (k not in d):
            d[k] = {}
        d = d[k]
    if not allow_new:
        assert keys[-1] in d
    d[keys[-1]] = val


def flatten_dict(d, sep='.'):
    """ Flatten a nest dict by concatenating nested keys """
    return pd.json_normalize(d, sep=sep).to_dict(orient='records')[0]


def find_key_in_dict(d, key):
    """ Return paths (list of lists) to all occurrences of key in dict. """
    sep = '.'
    d_flat = flatten_dict(d, sep=sep)
    paths = []
    for k in d_flat:
        if k.endswith(sep + key):
            paths.append(k.split(sep))
    return paths


def dict_sort(d):
    return {key: val if not isinstance(val, dict) else dict_sort(val)
            for key, val in sorted(d.items())}


def slice_dict_array(d_, ixs, ignore_prfx='_'):
    """
    apply slicing on all arrays in dict
    :param d_: dict
    :param ixs: slicing indices
    :param ignore_prfx: ignore keys with this prefix
    :return: dict with sliced arrays
    """
    d = deepcopy(d_)
    for k in d:
        if ignore_prfx is not None and not k.startswith(ignore_prfx):
            d[k] = d[k][ixs]
    return d


class SymDict(dict):

    """
    Symmetric Dict: Dictionary agnostic to ordering within key.
    e.g. key = (0,1,2) is indistinguishable from (0,2,1) / (1,2,0) / (1,0,2)
    """

    def __init__(self, keys, vals):
        super().__init__(zip([frozenset(key) for key in keys], vals))

    def __getitem__(self, key):
        return dict.__getitem__(self, frozenset(key))

    def __setitem__(self, key, value):
        dict.__setitem__(self, frozenset(key), value)

