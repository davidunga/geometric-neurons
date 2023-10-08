import numpy as np
import pandas as pd
from common.type_utils import *
from itertools import combinations, chain


def iter_indexes_of_item(k: int, n: int) -> Iterator[int]:
    assert 0 <= k < n
    if k == 0:
        item = 0
    else:
        item = k - 1
        yield item
        for step_size in range(n - 2, n - k - 1, -1):
            item += step_size
            yield item
        item += n - k
    yield from range(item, item + n - 1 - k)


def iter_pairs(n: int) -> Iterator[tuple[int, int]]:
    return combinations(range(n), 2)


def num_pairs(num_items: int) -> int:
    return int(num_items * (num_items - 1) / 2)


def num_items(num_pairs: int) -> int:
    n = (1 + np.sqrt(1 + 8 * num_pairs)) / 2
    assert n.is_integer()
    return int(n)


class SymmetricPairsData:

    _n_limit = 2 ** 16
    _no_group = "__noGroup__"

    def __init__(self, n: int, data: pd.DataFrame = None, group_by: (str, pd.Series) = None):

        self.n = n
        self.data = data
        self._group_by = group_by

        default = self._no_group if data is None and group_by is None else None
        self.pairs = pd.DataFrame(dict(group=[default] * num_pairs(self.n)))

        if isinstance(group_by, str):
            self.pairs['group'][self.data.index] = self.data[group_by]
        elif isinstance(group_by, pd.Series):
            self.pairs['group'][group_by.index] = group_by.values
        elif group_by is None and data is not None:
            self.pairs['group'][self.data.index] = self._no_group
        else:
            raise ValueError()

        self.pairs[['item1', 'item2']] = np.fromiter(iter_pairs(self.n), dtype=np.dtype((int, 2)))
        self._len = sum(~self.pairs['group'].isna())

        self._validate()

    def __setstate__(self, state) -> None:
        self.__init__(**state)

    def __getstate__(self) -> dict:
        return {'n': self.n, 'data': self.data, 'group_by': self._group_by}

    def _validate(self):
        assert self.n < self._n_limit
        assert self.pairs.index.max() < num_pairs(self.n)

    def indexes_of_item(self, item: int, label=None) -> NpVec[int]:
        labels = self.pairs['group'][iter_indexes_of_item(item, self.n)]
        if label is None:
            mask = labels.notna()
        else:
            mask = labels == label
        return mask.index[mask]

    def partners_of_item(self, item: int, label=None) -> NpVec[int]:
        indexes = self.indexes_of_item(item, label)
        partners = self.pairs.loc[indexes][['item1', 'item2']].to_numpy().flatten()
        return partners[partners != item]

    def labeled_pairs(self) -> pd.DataFrame:
        return self.pairs.loc[self.pairs['group'].notna()]

    def data_of_item(self, item, label=None) -> pd.DataFrame:
        return self.data.loc[self.indexes_of_item(item, label)]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self._len


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
        print("\nOthers of item 0:", list(symm_pairs_data.partners_of_item(0)))
        print("Others of item 2:", list(symm_pairs_data.partners_of_item(2)))
        print("Unravel:\n", symm_pairs_data.labeled_pairs().to_string())

    num_pts = 5
    data = _make_mock_dists_data(num_pts=num_pts)

    print("\n ======= All pairs: ======= ")
    _report(SymmetricPairsData(data=data, n=num_pts))

    print("\n ======= Partial pairs: ======= ")
    _report(SymmetricPairsData(data=data[::2], n=num_pts))


if __name__ == "__main__":
    example()
