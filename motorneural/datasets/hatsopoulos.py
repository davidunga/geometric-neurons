"""
Interface to Hatsopoulos's 2007 data
Motor-neural (M1/PMd) and kinematic data of 2 macaque monkeys performing
Random Target Pursuit & Center Out tasks.
More details: Hatsopoulos 2007, monkeys RS & RJ.

Overview:

    monkey1 = RockStar = RS:
        Performed TP & 2*COs, recordings only from M1
    monkey2 = Raju = RJ:
        Performed only TP, recordings are from M1 & PMd

    i.e. available data:
        monkey=RS task=TP region=M1     (100 neurons)
        monkey=RS task=CO region=M1     (141 neurons)
        monkey=RS task=CO region=M1     (68 neurons)
        monkey=RJ task=TP region=M1     (54 neurons)
        monkey=RJ task=TP region=PMd    (50 neurons)

    kinematics were sampled at 500 Hz


Structure:
- Fields of raw CenterOut Data:
'ans', 'cpl_0deg', 'cpl_0deg_endmv', 'cpl_0deg_go', 'cpl_0deg_instr',
'cpl_0deg_stmv', 'cpl_135deg', 'cpl_135deg_endmv', 'cpl_135deg_go',
'cpl_135deg_instr', 'cpl_135deg_stmv', 'cpl_180deg', 'cpl_180deg_endmv',
'cpl_180deg_go', 'cpl_180deg_instr', 'cpl_180deg_stmv', 'cpl_225deg',
'cpl_225deg_endmv', 'cpl_225deg_go', 'cpl_225deg_instr', 'cpl_225deg_stmv',
'cpl_270deg', 'cpl_270deg_endmv', 'cpl_270deg_go', 'cpl_270deg_instr',
'cpl_270deg_stmv', 'cpl_315deg', 'cpl_315deg_endmv', 'cpl_315deg_go',
'cpl_315deg_instr', 'cpl_315deg_stmv', 'cpl_45deg', 'cpl_45deg_endmv',
'cpl_45deg_go', 'cpl_45deg_instr', 'cpl_45deg_stmv', 'cpl_90deg',
'cpl_90deg_endmv', 'cpl_90deg_go', 'cpl_90deg_instr',
'cpl_90deg_stmv', 'cpl_st_trial', 'endmv', 'go_cue', 'instruction', 'reward',
'st_trial', 'stmv', 'spikes', 'chans', 'MIchans', 'instr_cell', 'go_cell',
'stmv_cell', 'endmv_cell', 'x', 'y', 'MIchan2rc'

- Fields of raw RTP Data:
'PositionX', 'PositionY', 'endmv', 'reward', 'st_trial', 'target_hit',
'Digital', 'y', 'x', 'spikes', 'chans', 'MIchans', 'PMdchans',
'cpl_st_trial_rew', 'PMdchan2rc', 'MIchan2rc', 'trial', 'monkey',
'force_x', 'force_y', 'shoulder', 'elbow', 't_sh', 't_elb', 'hit_target'

"""

# -----------------------

import numpy as np
from src.motorneural.data import Trial, Data, DatasetMeta
from src.motorneural.neural import NeuralData, PopulationSpikeTimes
from src.motorneural.motor import KinData, kinematics
from scipy.io import loadmat
import os
import re
from src.motorneural.typetools import Callable

# -----------------------

_DATASETS = {
    "TP_RS": {"file": "rs1050211_clean_spikes_SNRgt4.mat", "task": "TP", "monkey": "RS"},
    "TP_RJ": {"file": "r1031206_PMd_MI_modified_clean_spikesSNRgt4.mat", "task": "TP", "monkey": "RJ"},
    "CO_01": {"file": "rs1050225_clean_SNRgt4.mat", "task": "CO", "monkey": "RS"},
    "CO_02": {"file": "rs1051013_clean_SNRgt4.mat", "task": "CO", "monkey": "RS"}
}

# -----------------------


class HatsoData(Data):

    def __init__(self, trials, meta):
        super().__init__(trials, meta)

    @classmethod
    def make(cls, data_dir: str, dataset: str, lag: float, bin_sz: float, kin_fnc: Callable[None, KinData] = None, max_trials: int = None):
        return _load_data(data_dir, dataset, lag, bin_sz, kin_fnc, max_trials)

# -----------------------


def _load_data(data_dir: str, dataset: str, lag: float, bin_sz: float,
               kin_fnc: Callable[None, KinData] = None, max_trials: int = None) -> HatsoData:

    if kin_fnc is None:
        kin_fnc = kinematics

    assert np.abs(lag) <= 1, f"Extreme lag value: {lag}. Make sure its in seconds."
    assert 0 < bin_sz <= 1, f"Extreme bin size value: {bin_sz}. Make sure its in seconds."

    # ----------
    # helper functions:

    def _get_TP_events_and_properies(raw):
        """ Get Target Pursuit trial events and properties """
        st = np.real(raw['cpl_st_trial_rew'])[:, 0]
        end = np.real(raw['cpl_st_trial_rew'])[:, 1]
        mv_end = raw['endmv'].flatten()
        mv_end = mv_end[np.searchsorted(mv_end, st[0]):]
        assert len(mv_end) == len(st)
        event_tms = [{"st": st[ix], "end": end[ix], "mv_end": mv_end[ix]} for ix in range(len(st))]
        return event_tms, [{} for _ in range(len(st))]

    def _get_CO_events_and_properies(raw):
        """ Get Center Out trial events and properties """
        st = np.concatenate([raw[f'cpl_{ang}deg'].flatten() for ang in range(0, 360, 45)])
        si = np.argsort(st)
        st = st[si]
        angs = np.concatenate([np.tile(ang, raw[f'cpl_{ang}deg'].size) for ang in range(0, 360, 45)])[si]
        go = np.concatenate([raw[f'cpl_{ang}deg_go'].flatten() for ang in range(0, 360, 45)])[si]
        instr = np.concatenate([raw[f'cpl_{ang}deg_instr'].flatten() for ang in range(0, 360, 45)])[si]
        mv_st = np.concatenate([raw[f'cpl_{ang}deg_stmv'].flatten() for ang in range(0, 360, 45)])[si]
        mv_end = np.concatenate([raw[f'cpl_{ang}deg_endmv'].flatten() for ang in range(0, 360, 45)])[si]
        end = raw['reward'].flatten()
        event_tms = [{"st": st[ix], "end": end[ix], "instr": instr[ix], "go": go[ix],
                      "mv_st": mv_st[ix], "mv_end": mv_end[ix]} for ix in range(len(st))]
        properties = [{"ang": int(angs[ix])} for ix in range(len(angs))]
        return event_tms, properties

    def _get_neural_data(raw):
        """ Get neuron spikes times and info """
        chans = list(raw['MIchans'].squeeze())
        if 'PMdchans' in raw:
            chans += list(raw['PMdchans'].squeeze())
        neuron_spktimes = {}
        neuron_info = {}
        for k in sorted(list(k for k in raw.keys() if k.startswith('Chan'))):
            assert re.match("Chan[0-9]{3}[a-z]{1}", k) is not None
            neuron_name = k[4:]
            neuron_chan = int(k[4:-1])
            neuron_spktimes[neuron_name] = np.real(raw[k].flatten())
            neuron_info[neuron_name] = {'site': 'm1' if neuron_chan in raw['MIchans'] else 'pmd'}
        assert len(neuron_spktimes) > 0
        return PopulationSpikeTimes(neuron_spktimes), neuron_info

    # ----------
    # core:

    raw_ = loadmat(os.path.expanduser(data_dir) + "/" + _DATASETS[dataset]["file"])

    # full kinematics:
    X = np.stack([raw_['x'][:, 1], raw_['y'][:, 1]], axis=1)
    t = .5 * (raw_['x'][:, 0] + raw_['y'][:, 0])

    # full neural:
    population_spktimes, neuron_info = _get_neural_data(raw_)

    # get events and properties:
    events_tms, properties = (_get_CO_events_and_properies(raw_) if 'cpl_0deg' in raw_ else
                              _get_TP_events_and_properies(raw_))

    trials = []
    for ix, (tr_event_tms, tr_properties) in enumerate(zip(events_tms, properties)):

        if max_trials is not None and len(trials) == max_trials:
            break

        # trial skeleton:
        tr = Trial(dataset=dataset, ix=ix, lag=lag, bin_sz=bin_sz)

        # add neural data:
        st = np.ceil(tr_event_tms["st"] / bin_sz) * bin_sz
        end = np.floor((tr_event_tms["end"] - lag) / bin_sz) * bin_sz
        tr.neural = NeuralData(spktimes=population_spktimes.TimeSlice([st, end]),
                               fs=1 / bin_sz, tlims=[st, end], neuron_info=neuron_info)

        # add kinematic data:
        ifm, ito = np.searchsorted(t, [st + lag, end + lag])
        ifm, ito = max(0, ifm - 1), min(len(t), ito + 1)
        tr.kin = kin_fnc(X[ifm: ito], t[ifm: ito], dst_t=tr.neural.t + lag, dx=.5)

        # add events and properties:
        tr_event_tms["max_spd"] = tr.kin.t[np.argmax(tr.kin["spd2"])]
        tr.add_events(tr_event_tms, is_neural=False)
        tr.add_properties(tr_properties)

        assert tr.kin.num_samples == tr.neural.num_samples, (tr.kin.num_samples, tr.neural.num_samples)
        assert np.max(np.abs((tr.kin.t - tr.neural.t) - lag)) < 1e-6

        trials.append(tr)

    # normalize spike counts per neuron:
    all_spkcounts = np.concatenate([tr.neural.spkcounts for tr in trials], axis=1)
    assert all_spkcounts.shape[0] == trials[0].neural.num_neurons
    mu = np.mean(all_spkcounts, axis=1)
    sd = np.maximum(np.std(all_spkcounts, axis=1), np.finfo(float).eps)
    for tr in trials:
        tr.neural.spkcounts = (tr.neural.spkcounts - mu[:, None]) / sd[:, None]

    # ---
    # sanity-
    # properties that are suppose to be the same for all trials:
    assert all([tr.neural.neuron_info() == trials[0].neural.neuron_info() for tr in trials])
    assert all([tr.dataset == trials[0].dataset for tr in trials])
    assert all([tr.lag == trials[0].lag for tr in trials])
    assert all([tr.bin_sz == trials[0].bin_sz for tr in trials])
    # verify indices:
    assert [tr.ix for tr in trials] == list(range(len(trials)))
    # verify equal number of neural and kinematic samples, in each trial:
    assert all([tr.neural.num_samples == tr.kin.num_samples for tr in trials])
    # ---

    meta = DatasetMeta(**{**{"name": dataset, "sites": set(trials[0].neural.neuron_info("site"))}, **_DATASETS[dataset]})

    return HatsoData(trials, meta)


if __name__ == "__main__":
    data = HatsoData.make("~/data/hatsopoulos", "TP_RS", lag=.1, bin_sz=.01)

