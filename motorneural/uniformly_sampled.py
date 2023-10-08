from motorneural.typetools import *
import numpy as np


class UniformGrid:
    """ 1d uniform grid """

    def __init__(self, dt: float):
        self.dt = dt
        self.fs = 1 / self.dt

    @classmethod
    def from_samples(cls, t: Sequence, tol: float = 1e-6):
        dt = (t[-1] - t[0]) / (len(t) - 1)
        ret = cls(dt)
        ret.snap(t, tol)  # trigger assertion
        return ret

    def snap(self, t: Sequence[float], tol: float = 1e-6):
        """ snap values to grid ticks """
        tt = np.round(np.asarray(t) * self.fs) * self.dt
        if tol is not None:
            error = max(abs(t_ - tt_) for (t_, tt_) in zip(t, tt))
            assert error < tol * self.dt, f"Error={error}, Relative={error / self.dt}"
        return tt

    def get_ticks(self, lims: Pair, margin: bool = True, tol: float = 1e-6) -> Sequence[float]:
        lims = self.snap(lims, tol)
        ticks = np.linspace(lims[0], lims[1], int(.5 + np.diff(lims) * self.fs))
        if margin:
            ticks[0] -= self.dt * tol / 2
            ticks[-1] += self.dt * tol / 2
        return ticks
