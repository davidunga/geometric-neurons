import numpy as np
from collections import Counter

def get_float_format(f: int | float | str = '2.3'):
    prefix = '.' if isinstance(f, int) else ''
    suffix = '' if isinstance(f, str) and f.endswith('f') else 'f'
    return f'{prefix}{f}{suffix}'


def to_str(x, f: int | float | str = 2.3, keys='auto'):
    if isinstance(x, float):
        return f"{x:{get_float_format(f)}}"
    elif isinstance(x, (list, tuple, np.ndarray)):
        s = ', '.join([to_str(xx, f=f) for xx in x])
        return f'({s})' if isinstance(x, tuple) else f'[{s}]'
    elif isinstance(x, dict):
        if isinstance(keys, str):
            if keys == 'auto':
                keys = [k for k in x.keys() if not k.startswith('_')]
            elif keys == 'all':
                keys = x.keys()
            else:
                raise ValueError('Unknown keys argument')
        s = ', '.join([f"{key}:{to_str(x[key], f=f)}" for key in keys])
        return s
    else:
        return str(x)


def attribs_string(obj, f='2.3', dl='\n') -> str:
    if hasattr(obj, '__dict__'):
        obj = obj.__dict__
    s = [f"{k}: {to_str(v, f=f)}" for k, v in obj.items()]
    s = dl.join(s)
    return s


def parts(**kwargs) -> str:
    """
    e.g. parts(heads=4, tails=6) -> 'heads=4 (40.00%), tails=6 (60.00%), [total=10]'
    """
    if len(kwargs) == 1:
        key = next(iter(kwargs))
        val = kwargs[key]
        if hasattr(val, '__len__') and not isinstance(val, Counter):
            val = Counter(val)
        if isinstance(val, Counter):
            kwargs = {f'{key}[{k}]': count for k, count in val.items()}
    total = sum(kwargs.values())
    tokens = []
    for name, count in kwargs.items():
        tokens.append(f"{name}={count} ({count / total:3.2%})")
    tokens.append(f'[total={total}]')
    return ", ".join(tokens)


def part(p, q=None, show_tot: bool = True, pr: int = 2) -> str:

    if q is None:
        q = len(p)
        p = int(np.sum(p))

    pcnt = f"{p / q * 100:{get_fmt(3, pr)}}%"
    if show_tot:
        return f"{p}/{q} ({pcnt})"
    else:
        return f"{p} ({pcnt})"


def get_fmt(m: int, p: int) -> str:
    return f"{m}.{p}f"
