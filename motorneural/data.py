from common.utils.devtools import verbolize
from common.utils import strtools
from motorneural.typetools import *
from motorneural.neural import NeuralData
from motorneural.motor import KinData
from dataclasses import dataclass, field
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


@dataclass
class DataSlice:

    ix: int
    kin: KinData
    neural: NeuralData
    neural_pc: NeuralData = None
    parent: str = None
    properties: dict = None

    _lag: float = None

    def __str__(self):
        name = self.__class__.__name__
        return f"{name} {self.uid} dur={self.duration:2.3f} len={len(self)} bin={self.bin_size:2.3f} " \
               f"neurons={self.neural.num_neurons} lag={self.lag:2.3f}"

    def __repr__(self):
        return str(self)

    def __post_init__(self):
        self._lag = float(np.median(self.kin.t - self.neural.t))
        self.properties = {} if not self.properties else self.properties
        self._validate()

    def get_segment(self, ifrom: int, ito: int, ix: int = None, parent: str = None):
        kin = self.kin.get_slice(ifrom, ito)
        neural = self.neural.get_slice(ifrom, ito)
        neural_pc = self.neural_pc.get_slice(ifrom, ito)
        ix = ifrom if ix is None else ix
        parent = self.uid if parent is None else parent
        return Segment(kin=kin, neural=neural, neural_pc=neural_pc, ix=ix,
                       parent=parent, properties=self.properties)

    def get_binned(self, **kwargs):
        kin = self.kin.get_binned(**kwargs)
        neural = self.neural.get_binned(**kwargs)
        neural_pc = self.neural_pc.get_binned(**kwargs)
        return self.__class__(kin=kin, neural=neural, neural_pc=neural_pc, ix=self.ix,
                              parent=self.parent, properties=self.properties)

    def __len__(self):
        return len(self.kin)

    @property
    def uid(self) -> str:
        return f"{self.ix}/{self.parent}"

    @property
    def dataset(self) -> str:
        return self.parent.split('/')[-1]

    @property
    def bin_size(self) -> float:
        return self.kin.bin_size

    @property
    def lag(self) -> float:
        return self._lag

    @property
    def duration(self) -> float:
        return self.kin.duration

    @property
    def events(self) -> dict[str, int]:
        return {**self.kin.events, **self.neural.events, **self.neural_pc.events}

    def __getitem__(self, item: str):
        if '.' in item:
            attr, field = item.split('.')
            ret = getattr(self, attr)[field]
        else:
            ret = getattr(self, item)
        return ret

    def _validate(self):
        def _assert_close(obj1, obj2):
            assert np.isclose(obj1.bin_size, obj2.bin_size)
            assert np.isclose(obj1.duration, obj2.duration)
        _assert_close(self.kin, self.neural)
        if self.neural_pc:
            _assert_close(self.kin, self.neural_pc)
        lags = self.kin.t - self.neural.t
        assert np.max(np.abs(self._lag - lags)) < 1e-3 * self._lag


class Trial(DataSlice):
    """ Highest level data slice, direct child of dataset """
    pass


class Segment(DataSlice):
    """ A slice of trial (usually), or of a larger segment """
    pass


def postprocess_data_slices(
        data_slices: list[DataSlice],
        variable: str = 'neural',
        normalize: bool = True,
        drop_zero_variance: bool = True,
        new_bin_sz: float = None,
        inplace: bool = False):

    orig_bin_size = data_slices[0].bin_size

    slice_indexes = []
    def _register_index(df: pd.DataFrame, idx: int):
        slice_indexes.append([idx] * len(df))
        return df

    df = pd.concat((_register_index(s[variable].get_binned(bin_sz=new_bin_sz).df, idx)
                    for idx, s in enumerate(data_slices)), axis=0)

    slice_indexes = np.concatenate(slice_indexes)

    if new_bin_sz is not None:
        verbolize.inform(f"Time re-binned {orig_bin_size} -> {new_bin_sz} ="
                         f" (x{int(.5 + new_bin_sz / orig_bin_size)} decimation)")

    _eps = 1e-6
    if drop_zero_variance:
        is_zero_var_col = df.std() < _eps
        df.drop(columns=df.columns[is_zero_var_col], inplace=True)
        verbolize.inform(f"Dropped {strtools.part(is_zero_var_col, pr=1)} zero variance units")

    if normalize:
        df = (df - df.mean()) / (df.std() + _eps)

    if inplace:
        assert new_bin_sz is None, "Setting resampled inplace currently not supported"
        for i, s in enumerate(data_slices):
            s[variable]._df = df.iloc[slice_indexes == i, :]

    return df, slice_indexes


def set_neural_pcs_inplace(data_slices: list[DataSlice], num_pcs: int = 10):
    pca = PCA(n_components=num_pcs)
    spikecounts, _ = postprocess_data_slices(data_slices, 'neural', drop_zero_variance=False, inplace=False)
    pca.fit(spikecounts)
    meta = {'explained_variance_ratio': pca.explained_variance_ratio_,
            'explained_variance': pca.explained_variance_}
    pc_names = [f'pc{i + 1}' for i in range(pca.n_components)]
    for s in data_slices:
        df = pd.DataFrame(data=pca.transform(s.neural[:]), columns=pc_names)
        s.neural_pc = NeuralData(df=df, meta=meta, t=s.neural.t.copy())


def validate_data_slices(data_slices: list[DataSlice], same_len: bool = False,
                         normalized_neural: bool = False):
    """
    Verifies that properties that should be the same for all slices within a group (e.g. bin size)
    are indeed the same, and properties that should be unique (e.g. index) are indeed unique.
    Args:
        data_slices:
        same_len: if True, checks that all slices have the same length
        normalized_neural: if True, checks that neural data is normalized
    """

    def _unpack(attr: str | Callable):
        if isinstance(attr, str):
            return [getattr(s, attr) for s in data_slices]
        else:
            return [attr(s) for s in data_slices]

    def _all_close(attr):
        v = _unpack(attr)
        return np.min(v) > .99 * np.max(v)

    def _all_same(attr):
        v = _unpack(attr)
        return all([v[0] == vv for vv in v])

    def _all_unique(attr):
        v = _unpack(attr)
        return len(set(v)) == len(v)

    assert _all_unique('ix')
    assert _all_close('lag')
    assert _all_close('bin_size')
    assert _all_same(type)
    if same_len:
        assert _all_same(len)

    if isinstance(data_slices[0], Trial):
        assert _all_same('parent')
    elif isinstance(data_slices[0], Segment):
        pass
    else:
        raise TypeError()

    if normalized_neural:
        tol = 1e-3
        neurals = np.concatenate([s.neural[:] for s in data_slices], axis=0)
        assert np.max(np.abs(np.mean(neurals, axis=0))) < tol
        assert np.max(np.abs(1 - np.std(neurals, axis=0))) < tol


def postprocess_trials_inplace(trials: list[Trial], dataset: str, process_neural: bool = True):
    """
    - sets trials' parent
    - normalizes neural
    - sets neural PCs
    Args:
        trials:
        dataset: name of dataset, to set as trials' parent
    """
    for tr in trials:
        tr.parent = dataset
    if process_neural:
        postprocess_data_slices(trials, variable='neural', inplace=True,
                                drop_zero_variance=True, normalize=True)
    set_neural_pcs_inplace(trials, num_pcs=10)
    validate_data_slices(trials, normalized_neural=process_neural)

