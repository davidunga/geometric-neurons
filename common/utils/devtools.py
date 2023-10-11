import os.path
from time import time
import numpy as np
import inspect
from dataclasses import dataclass
from collections import defaultdict
import logging
import functools


class Verbolize:

    def __init__(self):
        self.level = 0
        self.indent_size = 2
        self.times_stack = []

    def __call__(self, *args, **kwargs):
        return self._verbolize(*args, **kwargs)

    def _verbolize(self, verbose: int = 1):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if verbose > 0:
                    self.times_stack.append(time())
                    print(f"{self.level * self.indent_size * ' '}{func.__name__} start")
                self.level += 1
                result = func(*args, **kwargs)
                self.level -= 1
                if verbose > 0:
                    dt = time() - self.times_stack[-1]
                    self.times_stack.pop()
                    print(f"{self.level * self.indent_size * ' '}{func.__name__} done. [{dt:2.4f}s]")
                return result
            return wrapper
        return decorator


verbolize = Verbolize()


class _Printer:

    def __init__(self):
        self.level = 'info'
        self.tstamp = True
        self._last_t = time()

    def __call__(self, *args, **kwargs):
        print(*args, **kwargs)

    def dbg(self, *args, **kwargs):
        t = time()
        if self.level == 'info':
            return
        prefix = 'debug'
        if self.tstamp:
            prefix = f'{prefix} [dt={t - self._last_t:2.4f}s]'
        print(prefix + ':', *args, **kwargs)
        self._last_t = t


printer = _Printer()



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
