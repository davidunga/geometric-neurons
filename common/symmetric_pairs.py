import numpy as np
import pandas as pd
from common.utils.typings import *
from itertools import combinations


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
    """
    pairwise information is held a dataframe (self.pairs) with columns: [item1, item2, group]
    self.pairs is full rank, i.e. has indexes 0,1,..,n(n-1)/2 - 1, n=number of items.
    if an index is missing, self.pairs.group[index] is None
    otherwise, self.pairs.group[index] if the group of the index, or self._no_group - if no groups were specified.
    """

    _n_limit = 2 ** 16
    _no_group = "__noGroup__"

    def __init__(self, n: int, data: pd.DataFrame = None, group_by: (str, pd.Series) = None,
                 pairs: pd.DataFrame = None):

        self.n = n
        self.data = data
        self._group_by = group_by
        self._label_set: set
        self._item_counts = None

        if pairs is not None:
            self.pairs = pairs
        else:
            self.pairs = pd.DataFrame(data=iter_pairs(self.n), columns=['item1', 'item2'])
            self.pairs['group'] = None
            if group_by is None:
                if data is None:
                    self.pairs['group'] = self._no_group
                else:
                    self.pairs.loc[self.data.index, 'group'] = self._no_group
            elif isinstance(group_by, pd.Series):
                self.pairs.loc[group_by.index, 'group'] = group_by.values
            else:
                assert isinstance(group_by, str)
                assert self.data is not None
                self.pairs.loc[self.data.index, 'group'] = self.data[group_by]

        self._label_set = set(self.pairs['group'].values) - {None}
        self._len = sum(self.pairs['group'].notna())
        self._num_items = None
        self._validate()

    def __setstate__(self, state) -> None:
        self.__init__(**state)

    def __getstate__(self) -> dict:
        return {'n': self.n, 'data': self.data, 'group_by': self._group_by}

    def _validate(self):
        assert self.n < self._n_limit
        assert np.all(self.pairs.index == pd.RangeIndex(num_pairs(self.n)))

    def indexes_of_item(self, item: int, label=None) -> pd.Index:
        labels = self.pairs.loc[iter_indexes_of_item(item, self.n), 'group']
        mask = labels.notna() if label is None else labels == label
        indexes = mask.index[mask]
        assert len(indexes), f"Item {item} does not appear in group={label}"
        return indexes

    def partners_of_item(self, item: int):
        indexes = self.indexes_of_item(item)
        partners = self.pairs.loc[indexes]
        ret = {}
        for label in self._label_set:
            ret[label] = partners.loc[partners['group'] == label, ['item1', 'item2']].to_numpy().flatten()
            ret[label] = ret[label][ret[label] != item]
        return ret if self.is_grouped else ret[self._no_group]

    def item_pairs(self, label=None) -> NpPairs[int]:
        indexes = self.pairs.group.notna() if label is None else self.pairs.group == label
        return self.pairs.loc[indexes, ['item1', 'item2']].values

    def is_grouped(self):
        return self._label_set != {self._no_group}

    def item_set(self, label=None) -> set[int]:
        return set(self.item_pairs(label).flatten())

    def labeled_pairs(self) -> pd.DataFrame:
        return self.pairs.loc[self.pairs['group'].notna()]

    def data_of_item(self, item, label=None) -> pd.DataFrame:
        return self.data.loc[self.indexes_of_item(item, label)]

    @property
    def data_columns(self):
        return self.data.columns

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self._len

    def num_items(self) -> int:
        if self._num_items is None:
            self._num_items = len(self.item_set())
        return self._num_items

    def mutex_indexes(self, split_items: Container[int]) -> NpVec[int]:
        """ returns a labels vector,
            1 = both pair items are members of split_items
            2 = both pair items are not members of split_items
            0 = pair is mixed
        """
        split1_counts = self.pairs[['item1', 'item2']].isin(split_items).sum(axis=1)
        split_labels = np.zeros(len(split1_counts), int)
        split_labels[split1_counts == 2] = 1
        split_labels[split1_counts == 0] = 2
        return split_labels


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
