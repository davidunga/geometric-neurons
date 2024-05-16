from itertools import combinations
from scipy import sparse
from typing import Iterator, Sequence
import numpy as np


def iter_pairs(n: int) -> Iterator[tuple[int, int]]:
    yield from combinations(range(n), 2)


def num_pairs(num_items: int) -> int:
    return num_items * (num_items - 1) // 2


def num_items(num_pairs: int) -> int:
    n = (1 + np.sqrt(1 + 8 * num_pairs)) / 2
    assert n.is_integer()
    return int(n)


def iter_group_pairs(labels: np.ndarray, ignore=(-1, None)) -> Iterator[tuple[int, int]]:
    """ iterate over pairs where both items are of the same group
    Args:
        labels: 1d array, group label for each item
        ignore: labels to ignore
    Returns:
        (i,j) pairs such that labels[i]==labels[j]
    """
    if not hasattr(ignore, '__len__'): ignore = (ignore,)
    labels = np.asarray(labels)
    label_set = sorted(set(labels).difference(ignore))
    for label in label_set:
        yield from combinations(np.nonzero(labels == label)[0], 2)


def to_sparse_matrix(values: Sequence, n: int = None, pairs: Sequence = None) -> sparse.csr_array:
    """
    to_sparse_matrix(values, n, pairs)
        where n is the number of items and values[i] is the value for the i-th pair
        constructs n*n matrix such that mtx[*pair[i]] = values[i]
    to_sparse_matrix(values)
        assumes values is the squareform-array
    """
    if n is None:
        n = num_items(len(values))
    if pairs is None:
        assert len(values) == num_pairs(n)
        pairs = iter_pairs(n)
    else:
        assert len(values) <= num_pairs(n)
        assert len(values) == len(pairs)
    i, j = np.asarray(pairs).T
    mtx = sparse.coo_array((values, (i, j)), (n, n)).tocsr()
    return mtx
