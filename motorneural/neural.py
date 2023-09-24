"""
Handles processing and storing neural information
"""

from src.motorneural.typetools import *
from src.motorneural.uniformly_sampled import UniformlySampled, make_time_bins
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
            (tms, ixs) where:
                tms is a sorted array of all neurons' spike times,
                ixs[i] is the index of the neuron that spiked at tms[i]
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

    def TimeSlice(self, tlims: Pair[float]):
        """ Get a time-sliced copy of object """
        sliced = PopulationSpikeTimes()
        sliced._tms, sliced._ixs = self.population_spktimes(tlims)
        sliced._names = self._names
        sliced._validate()
        return sliced

    def neuron_spktimes(self, tlims: Pair[float] = None) -> dict[str, NpVec[float]]:
        """
        Get spike-times per neuron
        Returns:
            results dict such that results[<neuron>] is a list of spike times
        """
        result = {name: np.array([], float) for name in self.names}
        for tm, ix in zip(*self.population_spktimes(tlims)):
            result[self.names[ix]] = np.append(result[self.names[ix]], tm)
        return result

    def neuron_spkcounts(self, bin_edges) -> dict[str, NpVec[float]]:
        """
        Get spike-counts per neuron
        Returns:
            results dict such that results[<neuron>] is the spikes histogram for <neuron> within <bin_edges>
        """
        return {name: np.histogram(tms, bins=bin_edges)[0] for name, tms in self.neuron_spktimes().items()}


# -------------------------------------------


class NeuralData(UniformlySampled):

    """ Neural data: spike times, spike counts, and additional info on source neurons """

    def __init__(self,
                 spktimes: PopulationSpikeTimes,
                 fs: float,
                 tlims: Pair = None,
                 neuron_info: dict[str, dict] = None):
        """
        Args:
            spktimes: population spike times
            fs: desired spike counts sampling rate (1 / bin_sz)
            tlims: time limits
            neuron_info: a dictionary of info per neuron. all neurons' dictionaries must have the same keys
        """
        if tlims is None:
            tms = spktimes.population_spktimes()[0]
            tlims = [tms[0] - 1 / fs, tms[-1] + 1 / fs]
        super().__init__(fs, tlims[0], spktimes.neuron_spkcounts(bin_edges=make_time_bins(fs, tlims)))
        if neuron_info is not None:
            assert set(spktimes.names) == set(neuron_info.keys())
            self._neuron_info = {name: neuron_info[name] for name in spktimes.names}
        else:
            self._neuron_info = None

    @property
    def names(self):
        return self._keys

    @property
    def num_neurons(self):
        return len(self.names)

    @property
    def spkcounts(self):
        return self._as_array()

    @spkcounts.setter
    def spkcounts(self, s):
        self._set_array(s)

    def neuron_info(self, item: str = None) -> list[Any]:
        """ List of info items per neuron, ordered by neuron names """
        if item is None:
            return self._neuron_info
        return [self._neuron_info[neuron][item] for neuron in self.names]
