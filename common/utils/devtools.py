import os.path
from time import time
import numpy as np
import inspect
from dataclasses import dataclass
from collections import defaultdict
from common.utils.typings import *
import logging
import functools
#import time


class progbar:

    _full_element = u"\u2588"
    _half_element = u"\u258c"
    _dot_element = u"\u00B7"
    _empty_element = " "

    def __init__(self, x: int | Iterable, n: int = None, prefix: str = "", suffix: str = "",
                 span: int = 10, enum: bool = False, leave: str = "none"):

        """
        progress bar.
        Args:
            x: iterable, or range span (int)
            n: number of iterations. default = len(x) if x is sized
            prefix: prefix string
            suffix: suffix string
            span: length of bar, in chars
            enum: add enumeration
            leave: "all", "none", or comma-separated string of fields to leave after completion,
                e.g., "prefix,bar"
        """

        if isinstance(x, int):
            self.n = x
            x = range(x)
        if n is None and isinstance(x, Sized):
            self.n = len(x)

        self.x = x
        self.prefix = prefix
        self.suffix = suffix
        self.span = span
        self.enum = enum
        self.leave = leave
        self.s = ""
        self._is_sized = self.n is not None
        self._t_start = None
        pass

    def update_field(self, name: str, value: str):
        """ update field for current iteration. e.g., update_field('suffix', 'my suffix')  """
        items = self._items
        items[name] = value
        self._update(items)

    def clear(self):
        print("\b" * len(self.s), end="")

    def __iter__(self):

        if self._t_start is None:
            self._t_start = time()

        items = {}

        for count, xx in enumerate(self.x, start=1):

            items['prefix'] = self.prefix
            items['bar'] = self._make_bar(count)
            items['count'] = self._make_count(count)
            if self._is_sized:
                items['percent'] = f"{int(round(100 * count / self.n)):3d}%"
            items['time'] = self._make_time(count)
            items['suffix'] = self.suffix

            if count == 1:
                self._filter_leave_items(items)  # dryrun to detect problems at the start
            elif count == self.n:
                items = self._filter_leave_items(items)

            self._update(items)

            if self.enum:
                yield count - 1, xx
            else:
                yield xx

            if count == self.n:
                break

    def _make_bar(self, count: int) -> str:
        if self._is_sized:
            p = count / self.n
            b = int(p * self.span) * [self._full_element]
            b += (round(p * self.span) - int(p * self.span)) * [self._half_element]
            b += (self.span - len(b)) * [self._empty_element]
        else:
            b = [self._dot_element] * self.span
            b[count % self.span] = self._full_element
        return "".join(b)

    def _make_time(self, count: int) -> str:
        t_elapsed = time() - self._t_start
        t_avg = t_elapsed / count
        remain_str = f"<{t_avg * (self.n - count):2.3f}s" if self._is_sized else ""
        return f"{t_elapsed:2.3f}s{remain_str},{t_avg:2.3f}s"

    def _make_count(self, count: int) -> str:
        n_digits = len(str(self.n)) if self._is_sized else 6
        return f"{count:{n_digits}}" + (f"/{self.n}" if self._is_sized else "")

    def _update(self, items):
        self._items = items
        self.clear()
        self.s = "|".join([v for v in items.values() if len(v)]) + ("|" if len(items) > 1 else "")
        print(self.s, end="")

    def _filter_leave_items(self, items):
        if self.leave == "none":
            leave_items = []
        elif self.leave == "all":
            leave_items = list(items.keys())
        else:
            leave_items = self.leave.split(",")
        if len(set(leave_items).difference(items.keys())):
            raise ValueError(f"Unknown items to leave: {set(leave_items).difference(items.keys())}")
        return {k: v for k, v in items.items() if k in leave_items}


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



if __name__ == "__main__":
    def g() -> Generator:
        for _ in range(10_000):
            yield time()
    from time import sleep
    z = np.arange(10)
    pbar = progbar(z, leave="prefix,bar", prefix="ff ")
    for i in pbar:
        sleep(1)
