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

    def __init__(self, n: int, data: pd.DataFrame = None, group_by: (str, pd.Series) = None):
        self.n = n
        self.data = data
        self._group_by = group_by
        self._group_labels: pd.Series = None
        self._participating_indexes: set = None
        self._validate()

    def __setstate__(self, state) -> None:
        self.__init__(**state)

    def __getstate__(self) -> dict:
        return {'n': self.n, 'data': self.data, 'group_by': self._group_by}

    def _validate(self):
        assert self.n < self._n_limit
        assert self.group_labels.index.max() < num_pairs(self.n)

    @property
    def group_labels(self):
        if self._group_labels is None:
            if self._group_by is None:
                self._group_labels = pd.Series([None] * num_pairs(self.n))
            elif isinstance(self._group_by, str):
                self._group_labels = self.data[self._group_by]
                assert not np.any(self._group_labels.isna()), "NA labels are not allowed"
            else:
                assert isinstance(self._group_by, pd.Series)
                self._group_labels = self._group_by
                assert not np.any(self._group_labels.isna()), "NA labels are not allowed"
        return self._group_labels

    def _group_of(self, index):
        """ group label of index.
            raises KeyError if index is not participating.
            returns None if no groups were specified (=all indexes are participating)
        """
        return self.group_labels[index]

    def _is_matching(self, index, label) -> bool:
        """
        if label is None - returns True iff index is participating
        if label is not None - returns True iff index's label matches given label
        """
        try:
            group = self._group_of(index)
        except KeyError:
            # index is not participating
            return False
        return label is None or group == label

    def iter_indexes_of_item(self, item: int, label=None) -> Iterator[int]:
        for index in iter_indexes_of_item(item, self.n):
            if self._is_matching(index, label):
                yield index

    def iter_partners_of_item(self, item: int, label=None) -> Iterator[int]:
        indexes = self.iter_indexes_of_item(item)
        partners = chain(range(item), range(item + 1, self.n))
        for partner, index in zip(partners, indexes):
            if self._is_matching(index, label):
                yield partner

    def iter_pairs(self, label=None) -> Iterator[tuple[int, int]]:
        for index, pair in enumerate(iter_pairs(self.n)):
            if self._is_matching(index, label):
                yield pair

    def iter_labeled_pairs(self) -> Iterator[tuple[int, int, Any]]:
        for index, pair in enumerate(iter_pairs(self.n)):
            try:
                yield pair, self._group_of(index)
            except KeyError:
                continue

    def data_of_item(self, item, label=None) -> pd.DataFrame:
        return self.data.loc[self.iter_indexes_of_item(item, label)]

    def __getitem__(self, item):
        return self.data[item]

    @property
    def participating_indexes(self):
        if self._participating_indexes is None:
            self._participating_indexes = set(list(self.group_labels.index))
        return self._participating_indexes

    def __len__(self):
        return len(self.group_labels)


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
        print("\nOthers of item 0:", list(symm_pairs_data.iter_partners_of_item(0)))
        print("Others of item 2:", list(symm_pairs_data.iter_partners_of_item(2)))
        print("Unravel:", list(symm_pairs_data.iter_pairs()))

    num_pts = 5
    data = _make_mock_dists_data(num_pts=num_pts)

    print("\n ======= All pairs: ======= ")
    _report(SymmetricPairsData(data=data, n=num_pts))

    print("\n ======= Partial pairs: ======= ")
    _report(SymmetricPairsData(data=data[::2], n=num_pts))


if __name__ == "__main__":
    example()
