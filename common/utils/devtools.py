import functools
import inspect
import os.path
from collections import defaultdict
from dataclasses import dataclass
from time import time

import numpy as np

from common.utils.typings import *


class progbar:

    def __init__(self, x: int | Iterable, n: int = None, prefix: str = "", suffix: str = "",
                 span: int = 10, enum: bool = False, leave: str = "none", refresh_frq: float = 1.):

        """
        progress bar.
        Args:
            x: iterable, or range span (int)
            n: number of iterations, default = len(x) if x is sized
            prefix: prefix string
            suffix: suffix string
            span: length of bar, in chars. default = 10
            enum: flag - add enumeration
            leave: fields to leave after completion. either comma-separated string, "all", or "none".
                e.g., "prefix,bar,suffix"
            refresh_frq: number of seconds between refreshes
        """

        self.x = range(x) if isinstance(x, int) else x
        self.n = len(self.x) if n is None and hasattr(self.x, '__len__') else n
        self.prefix = prefix
        self.suffix = suffix
        self.span = span
        self.enum = enum
        self.count = 0
        self.leave = leave
        self.s = ""
        self._is_sized = self.n is not None
        self._t_start = None
        self._t_last_update = None
        self.refresh_frq = refresh_frq
        self.print = functools.partial(print, flush=True)  # avoid printing buffer

    def update(self, **kwargs):
        """
        re-print based on current state, optional override fields.
        e.g.,
            update() - re-print based on state and specifications
            update(prefix='myPrefix', suffix='mySuffix') - override prefix and suffix
        """

        if self._t_start is None:
            self._t_start = time()

        t = time()
        if self._t_last_update and (t - self._t_last_update) < self.refresh_frq:
            return

        elements = {}
        elements['prefix'] = self.prefix
        elements['bar'] = self._make_bar()
        elements['count'] = self._make_count()
        elements['percent'] = self._make_percent()
        elements['time'] = self._make_time(t)
        elements['suffix'] = self.suffix

        for k in kwargs:
            assert k in elements, f"Unknown field {k}"
            elements[k] = kwargs[k]

        if self.count == 1:
            self._filter_leave_elements(elements)  # dryrun to detect problems at the start
        elif self.count == self.n:
            elements = self._filter_leave_elements(elements)

        self.clear()
        sp = "|"
        self.s = sp.join([v for v in elements.values() if len(v)]) + (sp if len(elements) > 1 else "")
        self.print(self.s, end="")
        self._t_last_update = t

    def clear(self):
        self.print("\b" * len(self.s), end="")

    def __iter__(self):
        for self.count, xx in enumerate(self.x, start=1):
            self.update()
            if self.enum:
                yield self.count - 1, xx
            else:
                yield xx
            if self.count == self.n:
                break

    def _make_bar(self) -> str:
        _full_element = u"\u2588"
        _part_elements = [u'\u258F', u'\u258E', u'\u258D', u'\u258C',
                          u'\u258B', u'\u258A', u'\u2589']
        _dot_element = u"\u00B7"
        if self._is_sized:
            p = self.count / self.n
            b = int(p * self.span) * [_full_element]
            part_ix = int(((p * self.span) - int(p * self.span)) * 8)
            if part_ix > 0:
                b += _part_elements[part_ix - 1]
            b += (self.span - len(b)) * [" "]
        else:
            b = [_dot_element] * self.span
            b[self.count % self.span] = _full_element
        return "".join(b)

    def _make_time(self, t) -> str:
        t_elapsed = t - self._t_start
        t_avg = t_elapsed / self.count
        remain_str = f"<{t_avg * (self.n - self.count):2.3f}s" if self._is_sized else ""
        return f"{t_elapsed:2.3f}s{remain_str},{t_avg:2.3f}s"

    def _make_count(self) -> str:
        n_digits = len(str(self.n)) if self._is_sized else 6
        return f"{self.count:{n_digits}}" + (f"/{self.n}" if self._is_sized else "")

    def _make_percent(self) -> str:
        return f"{int(round(100 * self.count / self.n)):3d}%" if self._is_sized else ""

    def _filter_leave_elements(self, elements):
        if self.leave == "none":
            leave_fields = set()
        elif self.leave == "all":
            leave_fields = set(elements.keys())
        else:
            leave_fields = set(fld.strip() for fld in self.leave.split(","))
        unknowns = leave_fields.difference(elements.keys())
        if len(unknowns):
            raise ValueError(f"Unknown fields to leave: {unknowns}")
        return {k: v for k, v in elements.items() if k in leave_fields}


class Verbolize:

    def __init__(self):
        self.level = 0
        self.indent_size = 2
        self.stack = []
        self.verbose = 1

    def __call__(self, *args, **kwargs):
        return self._verbolize(*args, **kwargs)

    def _verbolize(self, verbose: int = 1):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.verbose = verbose
                self.open(name=func.__name__)
                result = func(*args, **kwargs)
                self.close()
                return result
            return wrapper
        return decorator

    def indent_print(self, *args, **kwargs):
        print(self.level * self.indent_size * ' ', end='')
        print(*args, **kwargs)

    def open(self, name: str, inline: bool = False):
        self.stack.append((name, time(), inline))
        if self.verbose:
            if not inline:
                self.indent_print(name + " start")
            else:
                self.indent_print(name, end="... ")
        self.level += 1

    def close(self):
        t = time()
        name, start_t, inline = self.stack.pop()
        self.level -= 1
        if self.verbose:
            dt = t - start_t
            if not inline:
                self.indent_print(f"{name} done. [{dt:2.4f}s]")
            else:
                print(f"done. [{dt:2.4f}s]")

    def inform(self, *args, **kwargs):
        self.indent_print(*args, **kwargs)

    def alert(self, *args, **kwargs):
        self.indent_print("!", *args, **kwargs)


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



if __name__ == "__main__":
    def g() -> Generator:
        for _ in range(10_000):
            yield time()
    from time import sleep
    z = np.arange(10) + 500
    pbar = progbar(z, leave="all", enum=True)
    for i, zz in pbar:
        pbar.update(prefix=str(i), suffix=str(zz))
        sleep(np.random.random())

