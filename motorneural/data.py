
from src.motorneural.typetools import *
from src.motorneural.neural import NeuralData
from src.motorneural.motor import KinData
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


@dataclass
class DataSummary:
    name: str
    bin_sz: float
    lag: float

    def __str__(self):
        return f"{self.name} bin{int(.5 + 1000 * self.bin_sz):d} lag{int(.5 + 1000 * self.lag):d}"


@dataclass
class Event:
    """ Neural/kinematic event time and index """
    name: str
    tm: float
    ix: int
    is_neural: bool

# ----------------------------------


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

    @property
    def duration(self):
        if not ("end" in self and "st" in self):
            raise AssertionError("Start and end trial events are not defined")
        return self.end - self.st

    @property
    def data_summary(self):
        return DataSummary(self.dataset, bin_sz=self.bin_sz, lag=self.lag)

    @property
    def num_samples(self):
        assert self.kin.num_samples == self.neural.num_samples
        return self.kin.num_samples

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
            ix = self.neural.index(tm) if is_neural else self.kin.index(tm)
            self._events[name] = Event(name=name, tm=tm, ix=ix, is_neural=is_neural)

    def add_properties(self, properties: dict[str, Any]):
        for name in properties:
            if name in self._events or name in self._properties:
                raise AssertionError("Event or property already exists: " + name)
            self._properties[name] = properties[name]

    def __getattr__(self, item):
        return self.__getitem__(item)

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

    def _validate(self):
        if not len(self._trials):
            raise ValueError("Cannot initialize data with empty trial list")
        # all trials should have the same lag and in size:
        assert all([tr.lag == self[0].lag for tr in self])
        assert all([tr.bin_sz == self[0].bin_sz for tr in self])
        assert all([tr.data_summary == self[0].data_summary for tr in self])
        # all trials should have the same events and properties:
        assert all([tr.properties.keys() == self[0].properties.keys() for tr in self])
        assert all([tr.events.keys() == self[0].events.keys() for tr in self])

    def set_trials(self, trials: list[Trial]):
        self._trials = trials
        self._validate()

    @property
    def meta(self) -> DatasetMeta:
        return self._meta

    @property
    def summary(self) -> DataSummary:
        return self._trials[0].data_summary

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

