from typing import Sequence

import numpy as np
import numpy.typing
import pandas as pd

from common.utils.sigproc import reduce_rows


class NpDataFrame:

    __slots__ = ['_df', '_t', '_aliases', '_meta', '_events']

    def __init__(self, df: pd.DataFrame, aliases: dict = None, meta: dict = None, t: Sequence[float] = None,
                 events: dict[str, int] = None):
        self._df = df
        self._events = events if events else {}
        self._t = np.asarray(t if t is not None else [])
        self._aliases = {} if aliases is None else aliases
        self._meta = {} if meta is None else meta
        # --

        self._validate()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def _validate(self):
        assert 't' not in self._df.columns
        assert all(0 <= i < self._df.shape[0] for i in self._events.values())
        if self._t is not None:
            assert len(self._t) == self._df.shape[0]

    def __getstate__(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self._validate()

    @property
    def events(self) -> dict[str, int]:
        return self._events

    @events.setter
    def events(self, events_dict: dict[str, int], reorder: bool = True):
        if reorder:
            self._events = {k: v for k, v in sorted(events_dict.items(), key=lambda kv: kv[1])}
        else:
            self._events = events_dict

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, meta_dict):
        self._meta = meta_dict

    def get_slice(self, *args):
        slc = slice(*args)
        events = {name: i - slc.start for name, i in self._events.items() if slc.start <= i < slc.stop}
        return type(self)(self._df.iloc[slc], aliases=self._aliases, meta=self._meta,
                          t=self._t[slc] if self._t is not None else None, events=events)

    def get_flat_series(self) -> pd.Series:
        s = self._df.stack()
        s.index = [f'{col_name}.{index}' for index, col_name in s.index]
        return s

    def get_binned(self, bin_sz: float = None, factor: float = None):
        if bin_sz is None and factor is None:
            return self
        if bin_sz is not None:
            assert factor is None
            factor = bin_sz / self.bin_size
        win_sz = int(round(factor))
        values, sampled_ixs = reduce_rows(self[:], win_sz)
        new_len = len(self) // win_sz
        assert len(values) == new_len
        t = None if self._t is None else self._t[sampled_ixs]
        events = {name: i // win_sz for name, i in self._events.items() if i < new_len * win_sz}
        return type(self)(pd.DataFrame(values, columns=self.columns), aliases=self._aliases,
                          meta=self._meta, t=t, events=events)

    @property
    def columns(self):
        return self._df.columns

    @property
    def shape(self) -> tuple[int, int]:
        return self._df.shape

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, items) -> np.typing.NDArray:
        if isinstance(items, slice) or (isinstance(items, tuple) and isinstance(items[0], slice)):
            return self._df[self.columns].to_numpy().squeeze()[items]
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
        dt = (self.t[-1] - self.t[0]) / (len(self.t) - 1)

        # -- validate uniformity:
        dts = np.diff(self.t)
        assert np.abs(dt - dts).max() < 1e-5 * dt
        # --

        return dt

    def time2index(self, tm: float) -> int:
        return min(np.searchsorted(self.t, tm), len(self) - 1)

    @property
    def event_times(self) -> dict[str, float]:
        bin_size = self.bin_size
        return {name: i * bin_size for name, i in self._events.items()}

