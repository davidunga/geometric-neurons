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

import re
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from ..data import Trial, postprocess_trials_inplace, validate_data_slices
from ..motor import KinData, kinematics
from ..neural import NeuralData, PopulationSpikeTimes
from typing import Callable
from copy import deepcopy

# -----------------------

HATSO_DATASET_SPECS = {
    "TP_RS": {
        "file": "rs1050211_clean_spikes_SNRgt4.mat",
        "task": "TP",
        "monkey": "RS",
        "brain_sites": ["m1"],
        "total_neurons": 100,
        "trials_blacklist": [2, 92, 151, 167, 180, 212, 244, 256,
                             325, 415, 457, 508, 571, 662, 686, 748]
    },
    "TP_RJ": {
        "file": "r1031206_PMd_MI_modified_clean_spikesSNRgt4.mat",
        "task": "TP",
        "monkey": "RJ",
        "brain_sites": ["m1"],
        "total_neurons": 54,
        "trials_blacklist": [4, 10, 30, 43, 44, 46, 53, 66, 71, 78, 79, 84, 85, 91, 106,
                             107, 118, 128, 141, 142, 145, 146, 163, 165, 172, 173, 180,
                             185, 203, 209, 210, 245, 254, 260, 267, 270, 275, 278, 281,
                             283, 288, 289, 302, 313, 314, 321, 326, 340, 350, 363, 364,
                             366, 383, 385, 386, 390, 391]
    },
    "TP_RJ_PMD": {"_placeholder_"},
    "CO_RS_01": {
        "file": "rs1050225_clean_SNRgt4.mat",
        "task": "CO",
        "monkey": "RS",
        "brain_sites": ["m1"],
        "total_neurons": None,  # either 141 or 68 - need to check
        "trials_blacklist": []
    },
    "CO_RS_02": {
        "file": "rs1051013_clean_SNRgt4.mat",
        "task": "CO",
        "monkey": "RS",
        "brain_sites": ["m1"],
        "total_neurons": None,  # either 141 or 68 - need to check
        "trials_blacklist": []
    }
}
HATSO_DATASET_SPECS["TP_RJ_PMD"] = deepcopy(HATSO_DATASET_SPECS["TP_RJ"])
HATSO_DATASET_SPECS["TP_RJ_PMD"].update({"brain_sites": ["pmd"], "total_neurons": 50})


def get_hatso_datasets(**kwargs):
    filters = {k: [v] if isinstance(v, (str, int, float)) else v for k, v in kwargs.items()}
    datasets = []
    for dataset, spec in HATSO_DATASET_SPECS.items():
        match = True
        for k, v in filters.items():
            if spec[k] not in v:
                match = False
                break
        if match:
            datasets.append(dataset)
    return datasets


def get_metadata(dataset: str) -> dict:
    exclude = ["trials_blacklist"]
    meta = {k: v for k, v in HATSO_DATASET_SPECS[dataset].items() if k not in exclude}
    meta['dataset'] = dataset
    return meta

# -----------------------


def make_hatso_data(data_dir: Path, dataset: str, lag: float, bin_sz: float) -> tuple[list[Trial], dict]:
    """
    Args:
        data_dir: path to data folder
        dataset: name of dataset as appears in HATSO_DATASET_SPECS
        lag: lag between neural and kinematics
        bin_sz: bin duration, for both neural and kinematics
    """
    print(f"Making hatsopoulos data for {dataset}")
    specs = HATSO_DATASET_SPECS[dataset]
    trials = construct_hatso_trials(file=str(data_dir / specs["file"]), brain_sites=specs["brain_sites"],
                                    lag=lag, bin_sz=bin_sz, kin_fnc=kinematics,
                                    trials_blacklist=specs["trials_blacklist"])
    postprocess_trials_inplace(trials, dataset, process_neural=True)
    validate_data_slices(trials, normalized_neural=True)
    meta = get_metadata(dataset)
    return trials, meta


def construct_hatso_trials(
        file: str, brain_sites: list[str], lag: float, bin_sz: float,
        smooth_dur: (str, float) = 'auto', kin_fnc: Callable[..., KinData] = None,
        trials_blacklist: list[int] = None, max_trials: int = None) -> list[Trial]:

    if kin_fnc is None:
        kin_fnc = kinematics

    if smooth_dur == 'auto':
        smooth_dur = bin_sz

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
        properties = [{"CO_ang": int(ang)} for ang in angs]
        return event_tms, properties

    def _get_neural_spiketimes_and_info(raw):
        """ Get neuron spikes times and info """

        site_of_chan = {}  # dict channel index -> site name
        for chan_ in raw.get('MIchans', np.array([])).squeeze():
            site_of_chan[chan_] = 'm1'
        for chan_ in raw.get('PMdchans', np.array([])).squeeze():
            site_of_chan[chan_] = 'pmd'

        neuron_spktimes = {}
        neuron_info = {}
        for k in sorted(list(k for k in raw.keys() if k.startswith('Chan'))):
            assert re.match("Chan[0-9]{3}[a-z]{1}", k) is not None
            neuron_chan = int(k[4:-1])
            site = site_of_chan[neuron_chan]
            if site not in brain_sites:
                continue
            neuron_name = k[4:]
            neuron_spktimes[neuron_name] = np.real(raw[k].flatten())
            neuron_info[neuron_name] = {'site': site}
        assert len(neuron_spktimes) > 0
        return PopulationSpikeTimes(neuron_spktimes), neuron_info

    # ----------
    # core:

    raw_ = loadmat(file)

    # full kinematics (x,y,t):
    X = np.stack([raw_['x'][:, 1], raw_['y'][:, 1]], axis=1)
    t = .5 * (raw_['x'][:, 0] + raw_['y'][:, 0])

    # full neural:
    population_spktimes, neuron_info = _get_neural_spiketimes_and_info(raw_)

    # get events and properties per trial:
    events_tms, properties = (_get_CO_events_and_properies(raw_) if 'cpl_0deg' in raw_ else
                              _get_TP_events_and_properies(raw_))

    print(f"Constructing trials with bin_size={bin_sz:2.3f}, lag={lag:2.3f}, "
          f"num_neurons={population_spktimes.num_neurons} ...")
    print(f"{len(trials_blacklist)}/{len(events_tms)} trials are black listed")

    trials = []
    for raw_ix, (tr_event_tms, tr_properties) in enumerate(zip(events_tms, properties)):

        if raw_ix % 100 == 0:
            print(f"{raw_ix}/{len(events_tms)}")

        if raw_ix in trials_blacklist:
            continue

        if max_trials is not None and len(trials) == max_trials:
            break

        # neural:
        st = np.ceil(tr_event_tms["st"] / bin_sz) * bin_sz
        end = np.floor((tr_event_tms["end"] - lag) / bin_sz) * bin_sz
        neural = NeuralData.from_spike_times(
            spktimes=population_spktimes.get_time_slice([st, end]), bin_size=bin_sz,
            neuron_info=neuron_info, smooth_dur=smooth_dur)

        # kinematic:
        kin_t = neural.t + lag
        ifm, ito = np.searchsorted(t, kin_t[[0, -1]])
        ifm, ito = max(0, ifm - 1), min(len(t), ito + 1)
        kin = kin_fnc(X[ifm: ito], t[ifm: ito], dst_t=kin_t, dx=.5, smooth_dur=smooth_dur)

        kin_event_bins = {event_name: kin.time2index(event_time)
                          for event_name, event_time in tr_event_tms.items()}
        kin_event_bins['maxSpd'] = int(np.argmax(kin['EuSpd']))
        kin_event_bins['maxAcc'] = int(np.argmax(kin['EuAcc']))
        assert kin_event_bins['st'] == 0
        assert kin_event_bins['end'] == len(kin) - 1
        assert max(kin_event_bins.values()) <= kin_event_bins['end']
        assert min(kin_event_bins.values()) == 0

        kin.events = kin_event_bins
        trials.append(Trial(neural=neural, kin=kin, ix=len(trials), properties=tr_properties))

    return trials
