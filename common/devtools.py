from time import time

import numpy as np


class TicToc:
    def __init__(self):
        self._times = []

    def __call__(self):
        self._times.append(time())
        if len(self) > 1:
            print(f"timed {len(self) - 1}:", self._times[-1] - self._times[-2])

    def __len__(self):
        return len(self._times)

