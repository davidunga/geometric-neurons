import numpy as np
import pandas as pd
from dataclasses import dataclass
from common.type_utils import *
from itertools import combinations, chain


def indexes_of_item(k: int, n: int) -> NpVec[int]:
    assert 0 <= k < n
    if k == 0:
        return np.arange(n - 1)
    step_size = np.ones(n - 1, int)
    step_size[0] = 0
    step_size[1:k] = n - 1 - np.arange(1, k)
    if n > k + 1:
        step_size[k] = n - k
    return k - 1 + np.cumsum(step_size)


def iter_pairs(n: int) -> Iterator[tuple[int, int]]:
    return combinations(range(n), 2)


def num_pairs(num_items) -> int:
    return num_items * (num_items - 1) // 2


def num_items(num_pairs) -> int:
    return (1 + np.sqrt(1 + 8 * num_pairs)) // 2


class SymmetricPairs:

    def __init__(self, n: int, participating_indexes: Sequence[int] = None):
        self.n = n
        if participating_indexes is None:
            self.participating_indexes = set(range(num_pairs(self.n)))
        else:
            self.participating_indexes = set(participating_indexes)

    def indexes_of_item(self, item: int) -> list[int]:
        """ index of all pairs which include item """
        return [index for index in indexes_of_item(item, self.n)
                if index in self.participating_indexes]

    def partners_of_item(self, item: int) -> list[int]:
        """ pairing partners of item """
        all_indexes = self.indexes_of_item(item)
        all_partners = chain(range(item), range(item + 1, self.n))
        return [partner for partner, index in zip(all_partners, all_indexes)
                if index in self.participating_indexes]

    def iter_pairs(self) -> Iterator[tuple[int, int]]:
        for index, pair in enumerate(iter_pairs(self.n)):
            if index in self.participating_indexes:
                yield pair

    def __len__(self):
        return len(self.participating_indexes)


class SymmetricPairsData(SymmetricPairs):
    """
    Wrapper for a dataframe of symmetric-pairwise relations.
    """

    def __init__(self, data: pd.DataFrame, n: int):
        """
        Args:
            data: dataframe, indexed according to pairs.
            n: total number of items pairing was computed over.
        Examples:
            See example() below
        """
        self._data = data
        super().__init__(n, list(data.index))
        assert len(self) == self.shape[0]

    def __setstate__(self, state) -> None:
        self._data = state['data']
        self.n = state['n']
        self.participating_indexes = set(list(self.data.index))
        assert len(self) == self.shape[0]

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def data_of_item(self, item: int) -> pd.DataFrame:
        return self._data.loc[self.indexes_of_item(item)]

    def __getstate__(self) -> dict:
        return {'data': self.data, 'n': self.n}

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape


def example():

    def _make_mock_dists_data(num_pts):
        rng = np.random.default_rng(0)
        X = rng.random((num_pts, 2))
        dist_data = []
        for i in range(num_pts - 1):
            for j in range(i + 1, num_pts):
                dist_data.append({'i': i, 'j': j, 'Dist': np.linalg.norm((X[i] - X[j]))})
        return pd.DataFrame.from_records(dist_data)

    def _report(symm_pairs_data):
        print("Rows that contain item 0:")
        print(symm_pairs_data.data_of_item(0).to_string())
        print("\nRows that contain item 2:")
        print(symm_pairs_data.data_of_item(2).to_string())
        print("\nOthers of item 0:", symm_pairs_data.partners_of_item(0))
        print("Others of item 2:", symm_pairs_data.partners_of_item(2))
        print("Unravel:", list(symm_pairs_data.iter_pairs()))

    num_pts = 5
    data = _make_mock_dists_data(num_pts=num_pts)

    print("\n ======= All pairs: ======= ")
    _report(SymmetricPairsData(data=data, n=num_pts))

    print("\n ======= Partial pairs: ======= ")
    _report(SymmetricPairsData(data=data[::2], n=num_pts))


if __name__ == "__main__":
    example()
