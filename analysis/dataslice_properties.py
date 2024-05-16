import numpy as np
from common.utils import stats
from motorneural.data import Segment
from common.utils import conics


def ang_bins(segments: list[Segment], bins: stats.BinSpec) -> np.ndarray[int]:
    return stats.safe_digitize([s.kin.ang for s in segments], bins=bins)[0]


def arclen_bins(segments: list[Segment], bins: stats.BinSpec) -> np.ndarray[int]:
    return stats.safe_digitize([s.kin.arclen for s in segments], bins=bins)[0]


def shape_types(segments: list[Segment], **kwargs) -> np.ndarray:
    ret = np.array([conics.estimate_conic_type(s.kin.X, **kwargs) for s in segments])

    for i, r in enumerate(ret):
        if r != '_':
            circ_snr = (np.mean(segments[i]['EuCrv']) ** 2) / np.var(segments[i]['EuCrv'])
            if circ_snr > 2:
                ret[i] = 'c'
            elif ret[i] == 'c':
                ret[i] = 'e'
    return ret
