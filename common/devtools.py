import os.path
from time import time
import numpy as np
import inspect
from dataclasses import dataclass
from collections import defaultdict


class TicToc:

    @dataclass
    class Item:
        t: time
        file: str
        fn: str
        line: int

        def __str__(self):
            return f"{os.path.basename(self.file)}:{self.fn}:{self.line:3d}"

    def __init__(self, verbose: int = 1):
        self.items = []
        self.verbose = verbose
        self()

    def __call__(self):
        toc = time()
        _, file, line, fn, _, _ = inspect.stack()[1]
        self.items.append(TicToc.Item(t=toc, file=file, fn=fn, line=line))
        if self.verbose and len(self) > 1:
            print(f"timed {self.items[-1]}:", self.items[-1].t - self.items[-2].t)

    def __len__(self):
        return len(self.items)

    def report(self):
        line_times = defaultdict(list)
        for i in range(1, len(self)):
            dt = self.items[i].t - self.items[i - 1].t
            line_times[str(self.items[i])].append(dt)

        for name, times in line_times.items():
            print(f"{name} mean= {np.mean(times):2.4}s, count={len(times)}, total= {np.sum(times):2.4}s")
