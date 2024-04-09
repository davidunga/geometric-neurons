from itertools import combinations
from typing import Iterator
import numpy as np


def iter_pairs(n: int) -> Iterator[tuple[int, int]]:
    yield from combinations(range(n), 2)


def num_pairs(num_items: int) -> int:
    return num_items * (num_items - 1) // 2


def num_items(num_pairs: int) -> int:
    n = (1 + np.sqrt(1 + 8 * num_pairs)) / 2
    assert n.is_integer()
    return int(n)
