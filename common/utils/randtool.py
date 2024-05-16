import numpy as np


class Rnd:

    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)
        self._init_state = self.get_state()

    def get_state(self):
        return self._rng.__getstate__()

    def get_init_state(self):
        return self._init_state

    def set_state(self, state):
        return self._rng.__setstate__(state)

    def reset_state(self):
        self.set_state(self.get_init_state())

    def shuffle(self, a):
        return self._rng.permutation(a)

    def subset(self, a, n: int | float, shuff: bool = False):
        a = np.asarray(a)
        ret = a[self.split_mask(len(a), n)]
        if shuff:
            ret = self.shuffle(ret)
        return ret

    def split_mask(self, shape, n: int | float) -> np.ndarray[bool]:
        size = np.prod(shape)
        if n < 1:
            n = int(round(n * size))
        mask = np.zeros(np.prod(shape), bool)
        mask[self._rng.permutation(size)[:n]] = True
        return mask.reshape(shape)

    def integers(self, low, high=None, shape=None) -> np.ndarray[int]:
        return self._rng.integers(low, high, shape)
