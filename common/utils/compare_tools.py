from typings import *
import numpy as np


def unpack(objs: Iterable, attr: str | Callable | None):
    if attr is None:
        return objs
    if isinstance(attr, str):
        return (getattr(s, attr) for s in objs)
    else:
        return (attr(s) for s in objs)


def all_close(a, attr=None, rtol: float = .01):
    v = unpack(a, attr)
    return np.min(a) > (1 - rtol) * np.max(a)


def all_same(a, attr=None):
    v = unpack(a, attr)
    v0 = next(v)
    return all(v0 == vv for vv in v)


def all_unique(a, attr=None):
    v = list(unpack(a, attr))
    return len(set(v)) == len(v)
