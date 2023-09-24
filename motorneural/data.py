
from motorneural.typetools import *
from motorneural.neural import NeuralData
from motorneural.motor import KinData
from dataclasses import dataclass, field
import numpy as np

# ----------------------------------


@dataclass
class DatasetMeta:
    """ Dataset metadata """
    name: str
    task: str
    monkey: str
    sites: Set[str]
    file: str

#
# @dataclass
# class DataDef:
#
#     name: str
#     bin_sz: float
#     lag: float
#     seg_dur: float = None
#     seg_radcurv_bounds: tuple[float, float] = None
#     proc_kind: str = None
#
#     @property
#     def level(self) -> str:  # 'data'/'segments'/'dists'
#         lvl = int(self.seg_dur is not None) + int(self.proc_kind is not None)
#         return ('data', 'segments', 'dists')[lvl]
#
#     def __str__(self) -> str:
#         return getattr(self, f'{self.level}_str')()
#
#     def data_str(self) -> str:
#         return f"{self.name} bin{round(1000 * self.bin_sz):d} lag{round(1000 * self.lag):d}"
#
#     def segments_str(self) -> str:
#         return f"{self.data_str()} dur{round(1000 * self.seg_dur)}"
#
#     def dists_str(self) -> str:
#         return f"{self.segments_str()} proc{self.proc_kind.capitalize()}"
#
#     def __eq__(self, other):
#         return self.__dict__ == other.__dict__


@dataclass
class Event:
    """ Neural/kinematic event time and index """
    name: str
    tm: float
    ix: int
    is_neural: bool

# ----------------------------------


@dataclass
class Segment:

    trial_ix: int = None
    kin: KinData = None
    neural: NeuralData = None
    uid: str = 'auto'

    def __post_init__(self):
        assert len(self.kin) == len(self.neural)
        assert np.isclose(self.kin.duration, self.neural.duration)
        if self.uid == 'auto':
            self.uid = f"{self.trial_ix:03d}-{self.kin.t.mean():06.2f}"

    def get_slice(self, *args):
        kin = self.kin.get_slice(*args)
        neural = self.neural.get_slice(*args)
        return Segment(trial_ix=self.trial_ix, kin=kin, neural=neural)

    def __len__(self):
        return len(self.kin)

    @property
    def duration(self) -> float:
        return self.kin.duration


@dataclass
class Trial:
    """ Single trial data """

    dataset: str
    ix: int
    lag: float
    bin_sz: float
    kin: KinData = None
    neural: NeuralData = None

    _properties: dict[str, Any] = field(default_factory=dict)
    _events: dict[str, Event] = field(default_factory=dict)

    def get_segment(self, *args):
        kin = self.kin.get_slice(*args)
        neural = self.neural.get_slice(*args)
        return Segment(trial_ix=self.ix, kin=kin, neural=neural)

    @property
    def duration(self):
        if not ("end" in self and "st" in self):
            raise AssertionError("Start and end trial events are not defined")
        return self.end - self.st

    def __len__(self):
        return len(self.kin)

    @property
    def base_data_params(self) -> dict:
        return {'name': self.dataset, 'lag': self.lag, 'bin_sz': self.bin_sz}

    @property
    def properties(self) -> dict[str, Any]:
        return self._properties

    @property
    def events(self) -> dict[str, Event]:
        return self._events

    def add_events(self, event_tms: dict[str, float], is_neural=False):
        if (is_neural and self.neural is None) or (not is_neural and self.kin is None):
            raise AssertionError("Cannot set event before its prospective data is initialized")
        for name, tm in event_tms.items():
            if name in self._events or name in self._properties:
                raise AssertionError("Event or property already exists: " + name)
            ix = self.neural.time2index(tm) if is_neural else self.kin.time2index(tm)
            self._events[name] = Event(name=name, tm=tm, ix=ix, is_neural=is_neural)

    def add_properties(self, properties: dict[str, Any]):
        for name in properties:
            if name in self._events or name in self._properties:
                raise AssertionError("Event or property already exists: " + name)
            self._properties[name] = properties[name]

    def __getitem__(self, item):
        if item in self._events:
            return self._events[item].ix
        elif item in self._properties:
            return self._properties[item]
        else:
            raise AttributeError(f"Unknown event or property: " + item)
        pass


# ----------------------------------

@dataclass
class Data:

    """ Highest level data container
        Behaves as a Trial iterator, and provides access to dataset-level attributes
    """

    def __init__(self, trials: list[Trial], meta: DatasetMeta):
        self._trials = trials
        self._meta = meta
        self._validate()

    def __getstate__(self) -> dict:
        return {'trials': self._trials, 'meta': self._meta}

    def __setstate__(self, state):
        self._trials = state['trials']
        self._meta = state['meta']
        self._validate()

    def _validate(self):
        if not len(self._trials):
            raise ValueError("Cannot initialize data with empty trial list")
        # all trials should have the same lag and in size:
        assert all([tr.lag == self[0].lag for tr in self])
        assert all([tr.bin_sz == self[0].bin_sz for tr in self])
        assert all([tr.base_data_params == self[0].base_data_params for tr in self])
        # all trials should have the same events and properties:
        assert all([tr.properties.keys() == self[0].properties.keys() for tr in self])
        assert all([tr.events.keys() == self[0].events.keys() for tr in self])

    def set_trials(self, trials: list[Trial]):
        self._trials = trials
        self._validate()

    @property
    def meta(self) -> DatasetMeta:
        return self._meta
    #
    # @property
    # def datadef(self) -> DataDef:
    #     return self._trials[0].datadef

    @property
    def lag(self) -> float:
        return self._trials[0].lag

    @property
    def bin_sz(self) -> float:
        return self._trials[0].bin_sz

    @property
    def num_neurons(self) -> int:
        return self._trials[0].neural.num_neurons

    def __str__(self):
        return f"{self.name}: {len(self)} trials, {self.num_neurons:d} neurons, " \
               f"lag={self.lag:2.2f}s, bin={self.bin_sz:2.2f}s"

    def __iter__(self):
        yield from self._trials

    def __next__(self):
        return self.__iter__().__next__()

    def __len__(self):
        return len(self._trials)

    def __getitem__(self, item):
        if isinstance(item, np.ndarray):
            if item.dtype == bool:
                item = np.nonzero(item)[0]
            assert item.dtype == int
            return [self._trials[ix] for ix in item]
        return self._trials[item]

    def __getattr__(self, item):
        return self._meta.__getattribute__(item)

