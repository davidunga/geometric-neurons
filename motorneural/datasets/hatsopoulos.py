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

from pathlib import Path
import numpy as np
from motorneural.data import Trial, Data, DatasetMeta, postprocess_neural
from motorneural.neural import NeuralData, PopulationSpikeTimes
from motorneural.motor import KinData, kinematics
from scipy.io import loadmat
import re
from motorneural.typetools import Callable

# -----------------------

HATSO_DATASET_SPECS = {
    "TP_RS": {
        "file": "rs1050211_clean_spikes_SNRgt4.mat",
        "task": "TP",
        "monkey": "RS",
        "sites": ["m1"],
        "trials_blacklist": [2, 92, 151, 167, 180, 212, 244, 256,
                             325, 415, 457, 508, 571, 662, 686, 748]
    },
    "TP_RS_PMD": "auto",
    "TP_RJ": {
        "file": "r1031206_PMd_MI_modified_clean_spikesSNRgt4.mat",
        "task": "TP",
        "monkey": "RJ",
        "sites": ["m1"],
        "trials_blacklist": [4, 10, 30, 43, 44, 46, 53, 66, 71, 78, 79, 84, 85, 91, 106,
                             107, 118, 128, 141, 142, 145, 146, 163, 165, 172, 173, 180,
                             185, 203, 209, 210, 245, 254, 260, 267, 270, 275, 278, 281,
                             283, 288, 289, 302, 313, 314, 321, 326, 340, 350, 363, 364,
                             366, 383, 385, 386, 390, 391]
    },
    "CO_RS_01": {
        "file": "rs1050225_clean_SNRgt4.mat",
        "task": "CO",
        "monkey": "RS",
        "sites": ["m1"],
        "trials_blacklist": []
    },
    "CO_RS_02": {
        "file": "rs1051013_clean_SNRgt4.mat",
        "task": "CO",
        "monkey": "RS",
        "sites": ["m1"],
        "trials_blacklist": []
    }
}

HATSO_DATASET_SPECS["TP_RS_PMD"] = HATSO_DATASET_SPECS["TP_RS"].copy()
HATSO_DATASET_SPECS["TP_RS_PMD"]["sites"] = ["pmd"]

# -----------------------


def make_hatso_data(data_dir: Path, dataset: str, lag: float, bin_sz: float) -> Data:
    specs = HATSO_DATASET_SPECS[dataset]
    meta = DatasetMeta(
        file=str(data_dir / specs["file"]),
        name=dataset,
        task=specs["task"],
        monkey=specs["monkey"],
        sites=set(specs["sites"])
    )
    trials = load_trials(meta=meta, lag=lag, bin_sz=bin_sz, kin_fnc=kinematics,
                         trials_blacklist=specs["trials_blacklist"])
    trials = postprocess_neural(trials)
    data = Data(trials=trials, meta=meta)
    return data


def load_trials(meta: DatasetMeta,
                lag: float,
                bin_sz: float, smooth_dur: (str, float) = 'auto',
                kin_fnc: Callable[None, KinData] = None,
                trials_blacklist: list[int] = None,
                max_trials: int = None) -> list[Trial]:

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
        properties = [{"ang": int(angs[ix])} for ix in range(len(angs))]
        return event_tms, properties

    def _get_neural_data(raw):
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
            if site not in meta.sites:
                continue
            neuron_name = k[4:]
            neuron_spktimes[neuron_name] = np.real(raw[k].flatten())
            neuron_info[neuron_name] = {'site': site}
        assert len(neuron_spktimes) > 0
        return PopulationSpikeTimes(neuron_spktimes), neuron_info

    # ----------
    # core:

    raw_ = loadmat(meta.file)

    # full kinematics:
    X = np.stack([raw_['x'][:, 1], raw_['y'][:, 1]], axis=1)
    t = .5 * (raw_['x'][:, 0] + raw_['y'][:, 0])

    # full neural:
    population_spktimes, neuron_info = _get_neural_data(raw_)

    # get events and properties:
    events_tms, properties = (_get_CO_events_and_properies(raw_) if 'cpl_0deg' in raw_ else
                              _get_TP_events_and_properies(raw_))

    trials = []
    for raw_ix, (tr_event_tms, tr_properties) in enumerate(zip(events_tms, properties)):

        if raw_ix in trials_blacklist:
            continue

        if max_trials is not None and len(trials) == max_trials:
            break

        st = np.ceil(tr_event_tms["st"] / bin_sz) * bin_sz
        end = np.floor((tr_event_tms["end"] - lag) / bin_sz) * bin_sz

        # trial skeleton:
        tr = Trial(dataset=meta.name, ix=len(trials), lag=lag, bin_sz=bin_sz)

        # add neural data:
        st = np.ceil(tr_event_tms["st"] / bin_sz) * bin_sz
        end = np.floor((tr_event_tms["end"] - lag) / bin_sz) * bin_sz
        tr.neural = NeuralData.from_spike_times(
            spktimes=population_spktimes.get_time_slice([st, end]), fs=1 / bin_sz, tlims=(st, end),
            neuron_info=neuron_info, smooth_dur=smooth_dur)

        # add kinematic data:
        ifm, ito = np.searchsorted(t, [st + lag, end + lag])
        ifm, ito = max(0, ifm - 1), min(len(t), ito + 1)
        tr.kin = kin_fnc(X[ifm: ito], t[ifm: ito], dst_t=tr.neural.t + lag, dx=.5, smooth_dur=smooth_dur)

        # add events and properties:
        tr_event_tms["max_spd"] = tr.kin.t[np.argmax(tr.kin["EuSpd"])]
        tr.add_events(tr_event_tms, is_neural=False)
        tr.add_properties(tr_properties)

        assert len(tr.kin) == len(tr.neural)
        assert np.max(np.abs((tr.kin.t - tr.neural.t) - lag)) < 1e-6

        trials.append(tr)

    sites = set([v['site'] for v in trials[0].neural.neuron_info.values()])
    return trials


if __name__ == "__main__":
    name = "TP_RS"
    specs = DATASET_SPECS[name]
    meta = DatasetMeta(
        file = str(Path("~/data/hatsopoulos") / specs["file"]),
        name = name,
        task = specs["task"],
        monkey = specs["monkey"],
        sites = set(specs["sites"])
    )
    trials_blacklist = specs["trials_blacklist"]
    data = HatsoDataFactory(meta=meta, lag=.01, bin_sz=.05, kin_fnc=kinematics,
                            trials_blacklist=trials_blacklist).make()

