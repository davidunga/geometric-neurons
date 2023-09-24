import numpy as np


class shapesbank:

    @staticmethod
    def parabola(t0: float = -1, t1: float = 1, n: int = 50):
        t = np.linspace(t0, t1, n)
        return np.c_[t, t ** 2]
