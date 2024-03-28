"""
Handles processing and storing neural information
"""
from motorneural.npdataframe import NpDataFrame
from collections import Counter
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from motorneural.typetools import *
import numpy as np


class PopulationSpikeTimes:

    def __init__(self, *args, tlims: Pair[float] = None, **kwargs):
        """
        Either
            PopulationSpikeTimes(neuron_spktimes <,tlims=..>)
                where neuron_spktimes is a dict <neuron_name>:<neuron spike times>
        Or
            PopulationSpikeTimes(tms=.., ixs=.., names=.. <,tlims=..>)
                tms a is flat sorted array of all neurons' spike times
                ixs is an array same size as tms, ixs[i] is the index of the neuron that spiked at tms[i]
                names[ix] is the name of neuron with index ix

        tlims defines the [min, max] time range, default = [min, max] of all spike times.
        """

        if args:
            assert len(args) == 1 and isinstance(args[0], dict)
            neuron_spktimes = args[0]
            ixs = np.concatenate([[neuron_ix for _ in neuron_tms]
                                  for neuron_ix, neuron_tms in enumerate(neuron_spktimes.values())])
            tms = np.concatenate(list(neuron_spktimes.values()))
            names = np.array(list(neuron_spktimes.keys()))

            si = np.argsort(tms)
            tms = tms[si]
            ixs = ixs[si]

        else:
            assert len(kwargs) == 3
            tms = kwargs['tms']
            ixs = kwargs['ixs']
            names = kwargs['names']

        self._tms = np.asarray(tms, float)  # flat sorted array of all neurons' spike times
        self._ixs = np.asarray(ixs, int)    # same size as tms, ixs[i] is the index of the neuron that spiked at tms[i]
        self._names = np.asarray(names)     # names[ix] is the name of neuron with index ix
        self._tlims = tlims if tlims is not None else self._tms[[0, -1]]
        self._validate()

    def _validate(self):
        assert max(self._ixs) < self.num_neurons
        assert self._tms.ndim == self._ixs.ndim == 1
        assert len(self._tms) == len(self._ixs)
        assert len(self._tlims) == 2
        assert self._tms[0] >= self._tlims[0]
        assert self._tms[-1] <= self._tlims[1]
        assert is_sorted(self._tms)

    @property
    def names(self):
        """ Neuron names """
        return self._names

    @property
    def tms(self) -> NpVec[float]:
        """ All spike times """
        return self._tms

    @property
    def ixs(self) -> NpVec[int]:
        return self._ixs

    @property
    def num_neurons(self):
        return len(self._names)

    @property
    def tlims(self) -> Pair[float]:
        return self._tlims

    def get_time_slice(self, tlims: Pair[float]):
        ifm, ito = np.searchsorted(self._tms, tlims)
        sliced = PopulationSpikeTimes(ixs=self._ixs[ifm: ito], tms=self._tms[ifm: ito], names=self._names, tlims=tlims)
        return sliced

    def spike_times(self) -> dict[str, NpVec[float]]:
        """
        Get a dict of spike times per neuron
        """
        total_spikes_per_neuron = Counter(self._ixs)
        result = {name: np.zeros(total_spikes_per_neuron[ix], float)
                  for ix, name in enumerate(self.names)}
        visit_count = np.zeros(len(self.names), int)
        for ix, tm in zip(self._ixs, self._tms):
            name = self.names[ix]
            pos = visit_count[ix]
            result[name][pos] = tm
            visit_count[ix] += 1
        assert all(total_spikes_per_neuron[ix] == visit_count[ix] for ix in range(len(self.names)))
        return result

    def get_spike_counts(self, bin_edges) -> dict[str, NpVec[float]]:
        """
        Get dict of spike counts per neuron
        For uniform bins, get_raster() might be a better option
        """
        return {neuron_name: np.histogram(neuron_tms, bins=bin_edges)[0]
                for neuron_name, neuron_tms in self.spike_times().items()}

    def get_raster(self, bin_size: float, smooth_dur: float = 0, normalize: bool = False) -> [pd.DataFrame, NpVec]:
        """
        Get spikes raster dataframe
        Args:
            bin_size: bin duration
            smooth_dur: temporal sigma size of gaussian smoothing
            normalize: z-score normalize each neuron?
        Returns:
            df: dataframe with columns = neuron names, rows = spike counts
            bin_edges: time of row df.iloc[i] is between [bin_edges[i], bin_edges[i+1]]
        """
        t0 = int(self.tlims[0] / bin_size) * bin_size
        tf = int(1 + self.tlims[1] / bin_size) * bin_size
        bin_edges = np.arange(t0, tf, bin_size)
        spike_counts = self.get_spike_counts(bin_edges=bin_edges)
        if smooth_dur:
            sigma = smooth_dur / bin_size
            spike_counts = {name: gaussian_filter1d(counts, sigma=sigma, axis=0, mode='mirror') for
                            name, counts in spike_counts.items()}
        df = pd.DataFrame.from_dict(spike_counts)
        if normalize:
            df = (df - df.mean()) / df.std()
        return df, bin_edges

    def __str__(self):
        tmin, tmax = self.tms[[0, -1]]
        return f"PopulationSpikeTimes for {self.num_neurons} neurons, times {tmin} -> {tmax}"

    def __repr__(self):
        return str(self)

# -------------------------------------------


class NeuralData(NpDataFrame):
    """ Container for spikes raster """

    @classmethod
    def from_spike_times(
            cls, bin_size: float, spktimes: PopulationSpikeTimes,
            neuron_info: dict[str, dict] = None, smooth_dur: float = 0):
        raster_df, bin_edges = spktimes.get_raster(bin_size=bin_size, smooth_dur=smooth_dur)
        return cls(raster_df, meta=neuron_info, t=bin_edges[:-1])

    @property
    def neuron_info(self) -> dict:
        return self._meta

    @property
    def sites(self) -> list[str]:
        return list(set(v['site'] for v in self.meta.values()))

    @property
    def num_neurons(self) -> int:
        return self.shape[1]

    def __str__(self):
        return f"NeuralData: {self.num_neurons} neurons, {len(self)} bins"

    def __repr__(self):
        return str(self)


def _test():

    bin_size = .1
    rng = np.random.default_rng(1)

    def _simulate_spike_times(spike_counts):
        tms = []
        for bin_ix, count in enumerate(spike_counts):
            t0 = bin_ix * bin_size
            tms += list(t0 + rng.random(size=count) * bin_size)
        return np.asarray(tms)

    gt_spikes_counts_per_neuron = {
        'nA': [0, 3, 1, 5],
        'nB': [1, 1, 0, 2],
        'nC': [0, 0, 0, 0],
        'nD': [2, 0, 2, 1]
    }

    gt_spikes_counts_per_neuron = {k: np.asarray(v) for k, v in gt_spikes_counts_per_neuron.items()}
    gt_spike_times_per_neuron = {neuron: _simulate_spike_times(spike_counts)
                                 for neuron, spike_counts in gt_spikes_counts_per_neuron.items()}

    min_time = np.min(list(min(v) if len(v) else 10_000 for v in gt_spike_times_per_neuron.values()))
    max_time = np.max(list(max(v) if len(v) else -1 for v in gt_spike_times_per_neuron.values()))
    max_bins = np.max(list(len(v) for v in gt_spikes_counts_per_neuron.values()))
    bin_edges = np.arange(max_bins + 1) * bin_size

    assert bin_edges[0] <= min_time
    assert bin_edges[-1] > max_time

    pst = PopulationSpikeTimes(gt_spike_times_per_neuron)
    counts = pst.get_spike_counts(bin_edges)

    assert set(counts.keys()) == set(gt_spikes_counts_per_neuron.keys())
    for neuron in counts:
        assert np.all(counts[neuron] == gt_spikes_counts_per_neuron[neuron])
        print(neuron, "ok")




if __name__ == "__main__":
    _test()

