import numpy as np
import pandas as pd
from dataclasses import dataclass


class symmetric_pairs:

    @staticmethod
    def pair_indexes(k: int, n: int) -> np.ndarray[int]:
        """
        In a symmetric pairing of n items (e.g. flat pairwise distance array, excluding the diagonal), returns the
        index of pairs that include the k-th item.
        """

        assert 0 <= k < n

        if k == 0:
            return np.arange(n - 1)

        step_size = np.ones(n - 1, int)
        step_size[0] = 0
        step_size[1:k] = n - 1 - np.arange(1, k)

        if n > k + 1:
            step_size[k] = n - k

        ixs = k - 1 + np.cumsum(step_size)

        return ixs

    @staticmethod
    def unravel(n: int, indexes: list[int] = None) -> dict[int, (int, int)]:
        """
        index -> (i, j)
        Args:
            n: number of items
            indexes: list of indices
        Returns:
            dict, mapping from each index to its corresponding (i, j) pair.
        """

        if indexes is None:
            indexes = list(range(symmetric_pairs.num_pairs(n)))

        mapping = dict(zip(indexes, [None] * len(indexes)))
        collected_count = 0
        index = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if index in mapping:
                    mapping[index] = (i, j)
                    collected_count += 1
                    if collected_count == len(mapping):
                        return mapping
                index += 1
        raise ValueError("Did not find all specified indexes.")

    @staticmethod
    def ravel(n: int, pairs: list[tuple[int, int]]) -> dict[(int, int), int]:
        """
        (i, j) -> index
        Args:
            n: number of items
            pairs: list of item pairs
        Returns:
            dict, mapping from each pair (i, j) to its corresponding index.
        """

        mapping = dict(zip(pairs, [None] * len(pairs)))
        collected_count = 0
        index = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if (i, j) in mapping:
                    mapping[(i, j)] = index
                    collected_count += 1
                    if collected_count == len(mapping):
                        return mapping
                index += 1
        raise ValueError("Did not find all specified pairs.")

    @staticmethod
    def num_pairs(num_items) -> int:
        return num_items * (num_items - 1) // 2

    @staticmethod
    def num_items(num_pairs) -> int:
        return (1 + np.sqrt(1 + 8 * num_pairs)) // 2


@dataclass(init=False)
class SymmetricPairsData:
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
        self._data = None
        self._n = None
        self._data_indices = None
        self._is_partial = None
        self.__setstate__({'data': data, 'n': n})

    def __setstate__(self, state) -> None:
        self._data = state['data']
        self._n = state['n']
        self._data_indices = None
        self._is_partial = symmetric_pairs.num_pairs(self.n) > len(self)
        self._validate()

    @property
    def n(self) -> int:
        return self._n

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def is_partial(self) -> bool:
        return self._is_partial

    def get_pair_ixs(self, item_ix: int) -> np.ndarray[int]:
        """ index of all pairs which include item_ix """
        ixs = symmetric_pairs.pair_indexes(item_ix, self.n)
        if self.is_partial:
            if self._data_indices is None:
                self._data_indices = set(list(self._data.index))
            ixs = np.array([ix for ix in ixs if ix in self._data_indices], ixs.dtype)
        return ixs

    def get_item_pairs(self, item_ix: int) -> pd.DataFrame:
        pair_ixs = self.get_pair_ixs(item_ix)
        return self._data.loc[pair_ixs]

    def __getstate__(self) -> dict:
        return {'data': self.data, 'n': self.n}

    def _validate(self) -> None:
        data_pair_count = int(self._data.index.max()) + 1
        assert data_pair_count <= symmetric_pairs.num_pairs(self.n)

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    def __len__(self) -> int:
        return self.shape[0]


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
        print("Is Partial=", symm_pairs_data.is_partial)
        print("Rows that contain item 0:")
        print(symm_pairs_data.get_item_pairs(0).to_string())
        print("\nRows that contain item 2:")
        print(symm_pairs_data.get_item_pairs(2).to_string())

    num_pts = 5
    data = _make_mock_dists_data(num_pts=num_pts)

    print("\nAll pairs:")
    _report(SymmetricPairsData(data=data, n=num_pts))

    print("\nPartial pairs:")
    _report(SymmetricPairsData(data=data[::2], n=num_pts))

    pairs = [(0, 2), (0, 3), (7, 6)]
    indexes = symmetric_pairs.ravel(num_pts, pairs)
    print(indexes)
    print(symmetric_pairs.unravel(num_pts, list(indexes.values())))

if __name__ == "__main__":
    example()
