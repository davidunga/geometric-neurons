"""
Handles processing and storing neural information
"""

from copy import deepcopy

import pandas as pd

from motorneural.typetools import *
from motorneural.uniformly_sampled import make_time_bins
import numpy as np


class PopulationSpikeTimes:
    """ Stores neuron spike times in a flat time-sorted array
        Optimized for population-level analysis
    """

    def __init__(self, neuron_spktimes: dict[str, NpVec[float]] = None):
        """
        Args:
            neuron_spktimes: dict: <neuron_name>:<neuron spike times>
        """

        self._tms = None    # flat sorted array of all neurons' spike times
        self._ixs = None    # same size as tms, ixs[i] is the index of the neuron that spiked at tms[i]
        self._names = None  # names[ix] is the name of neuron with index ix
        if neuron_spktimes is not None:
            ixs = np.concatenate([np.tile(ix, len(tms)) for ix, tms in enumerate(neuron_spktimes.values())])
            tms = np.concatenate(list(neuron_spktimes.values()))
            si = np.argsort(tms)
            self._tms = tms[si]
            self._ixs = ixs[si]
            self._names = np.array(list(neuron_spktimes.keys()))
            self._validate()

    def _validate(self):
        assert max(self._ixs) < self.num_neurons
        assert self._tms.ndim == self._ixs.ndim == 1
        assert len(self._tms) == len(self._ixs)
        assert is_sorted(self._tms)

    def population_spktimes(self, tlims: Pair[float] = None) -> Tuple[NpVec[float], NpVec]:
        """
        Args:
            tlims: time limits, default = all
        Returns:
            tms - sorted array of all neurons' spike times,
            ixs - ixs[i] is the index of the neuron that spiked at tms[i]
        """
        ifm, ito = 0, len(self._ixs)
        if tlims is not None:
            ifm, ito = np.searchsorted(self._tms, tlims)
        return self._tms[ifm: ito], self._ixs[ifm: ito]

    @property
    def names(self):
        """ Neuron names """
        return self._names

    @property
    def num_neurons(self):
        return len(self._names)

    def get_time_slice(self, tlims: Pair[float]):
        sliced = PopulationSpikeTimes()
        sliced._tms, sliced._ixs = self.population_spktimes(tlims)
        sliced._names = self._names
        sliced._validate()
        return sliced

    def spike_times(self, tlims: Pair[float] = None) -> dict[str, NpVec[float]]:
        """
        Get spike times per neuron
        Returns:
            result - dict: neuron name -> vector of spike times
        """
        result = {name: np.array([], float) for name in self.names}
        for tm, ix in zip(*self.population_spktimes(tlims)):
            result[self.names[ix]] = np.append(result[self.names[ix]], tm)
        return result

    def spike_counts(self, bin_edges) -> dict[str, NpVec[float]]:
        """
        Returns:
            dict of spike histograms per neuron
        """
        return {name: np.histogram(tms, bins=bin_edges)[0]
                for name, tms in self.spike_times().items()}


# -------------------------------------------
from motorneural.npdataframe import NpDataFrame


class NeuralData(NpDataFrame):

    @classmethod
    def from_spike_times(cls,
                         fs: float,
                         spktimes: PopulationSpikeTimes,
                         tlims: tuple[float, float],
                         neuron_info: dict[str, dict] = None):
        if tlims is None:
            tms = spktimes.population_spktimes()[0]
            tlims = (tms[0] - 1 / fs, tms[-1] + 1 / fs)
        data = spktimes.spike_counts(bin_edges=make_time_bins(fs, tlims))
        df = pd.DataFrame.from_dict(data)
        t = make_time_bins(fs, tlims, margin=False)[:-1]
        return cls(df, meta=neuron_info, t=t)

    @property
    def neuron_info(self) -> dict:
        return self._meta

    @property
    def num_neurons(self) -> int:
        return self.shape[1]


#
# class NeuralData2(UniformlySampled):
#
#     """ Neural data: spike times, spike counts, and additional info on source neurons """
#
#     def __init__(self, *args, neuron_info: dict = None, **kwargs):
#         """
#         Args:
#             spktimes: population spike times
#             fs: desired spike counts sampling rate (1 / bin_sz)
#             tlims: time limits
#             neuron_info: a dictionary of info per neuron. all neurons' dictionaries must have the same keys
#         """
#
#         super().__init__(*args, **kwargs)
#
#         self._neuron_info = None
#         if neuron_info is not None:
#             assert set(self.names) == set(neuron_info.keys())
#             self._neuron_info = {name: neuron_info[name] for name in self.names}
#
#     @classmethod
#     def from_spike_times(cls,
#                          fs: float,
#                          spktimes: PopulationSpikeTimes,
#                          tlims: tuple[float, float] = None,
#                          neuron_info: dict[str, dict] = None):
#
#         if tlims is None:
#             tms = spktimes.population_spktimes()[0]
#             tlims = (tms[0] - 1 / fs, tms[-1] + 1 / fs)
#         t0 = tlims[0]
#         neuron_spkcounts = spktimes.neuron_spkcounts(bin_edges=make_time_bins(fs, tlims))
#         return cls(fs,
#                    t0,
#                    neuron_info=neuron_info,
#                    keys=list(neuron_spkcounts.keys()),
#                    vals=list(neuron_spkcounts.values()))
#
#     def get_slice(self, slc: slice):
#         return NeuralData(self._fs, self._t[slc.start], keys=self._keys, vals=self._vals[:, slc],
#                           neuron_info=self._neuron_info)
#
#     def to_json(self) -> dict:
#         jsn = super().to_json()
#         jsn['neuron_info'] = self._neuron_info
#         return jsn
#
#     @property
#     def names(self):
#         return self._keys
#
#     @property
#     def num_neurons(self):
#         return len(self.names)
#
#     @property
#     def spkcounts(self):
#         return self._as_array()
#
#     @spkcounts.setter
#     def spkcounts(self, s):
#         self._set_array(s)
#
#     def neuron_spkcounts(self) -> dict:
#         return dict(zip(self._keys, self._vals))
#
#     def neuron_info(self, item: str = None) -> list[Any]:
#         """ List of info items per neuron, ordered by neuron names """
#         if item is None:
#             return self._neuron_info
#         else:
#             return [self._neuron_info[neuron][item] for neuron in self.names]
#
#     def set_neuron_info(self, neuron_info: dict = None):
#         if neuron_info is None:
#             self._neuron_info = None
#         else:
#             assert set(self.names) == set(neuron_info.keys())
#             self._neuron_info = {name: neuron_info[name] for name in self.names}
#
#     def __str__(self):
#         return "NeuralData:" + super()._base_str()
#
#     def __repr__(self):
#         return str(self)
