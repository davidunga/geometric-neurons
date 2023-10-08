from typing import Sequence
import pandas as pd
import numpy as np
from motorneural.uniformly_sampled import UniformGrid
from common.sigproc import reduce_rows

class NpDataFrame:

    __slots__ = ['_df', '_t', '_aliases', '_meta']

    def __init__(self, df: pd.DataFrame, aliases: dict = None, meta: dict = None, t: Sequence[float] = None):
        self._df = df
        self._t = np.asarray(t if t is not None else [])
        self._aliases = {} if aliases is None else aliases
        self._meta = {} if meta is None else meta
        self._validate()

    def _validate(self):
        assert 't' not in self._df.columns
        if self._t is not None:
            assert len(self._t) == self._df.shape[0]

    def __getstate__(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self._validate()

    def get_slice(self, *args):
        slc = slice(*args)
        return type(self)(self._df.iloc[slc], aliases=self._aliases, meta=self._meta,
                          t=self._t[slc] if self._t is not None else None)

    def get_flat_series(self) -> pd.Series:
        s = self._df.stack()
        s.index = [f'{col_name}.{index}' for index, col_name in s.index]
        return s

    def get_binned(self, *args, win_sz: int = None, bin_sz: float = None):
        assert (win_sz is None) ^ (bin_sz is None)
        if bin_sz is not None:
            win_sz = int(.5 + bin_sz / self.bin_size)
            assert abs(win_sz - bin_sz / self.bin_size) < 1e-6, "bin_sz must be an integer multiply of current bin size"
        values = reduce_rows(self[:], win_sz, 'mean')
        t = None if self._t is None else reduce_rows(self._t, win_sz, 'mean')
        return type(self)(pd.DataFrame(values, columns=self.columns), aliases=self._aliases, meta=self._meta, t=t)

    @property
    def columns(self):
        return self._df.columns

    @property
    def shape(self) -> tuple[int, int]:
        return self._df.shape

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, items: (str, list[str])) -> np.typing.NDArray:
        if isinstance(items, slice):
            assert items == slice(None, None, None)
            colnames = self.columns
        else:
            colnames = []
            for item in (items if isinstance(items, list) else [items]):
                col = self._aliases.get(item, item)
                colnames += col if isinstance(col, list) else [col]
        return self._df[colnames].to_numpy().squeeze()

    def __getattr__(self, item):
        return self.__getitem__(item)

    # ---------
    # time-dependant functionality:

    @property
    def t(self) -> np.typing.NDArray:
        assert self._t is not None
        return self._t

    @property
    def duration(self) -> float:
        return float(np.diff(self.t[[0, -1]]))

    @property
    def bin_size(self) -> float:
        return UniformGrid.from_samples(self.t).dt

    def time2index(self, tm: float) -> int:
        return np.searchsorted(self.t, tm)
