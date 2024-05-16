import numpy as np


def to_str(x, f='2.3'):
    if isinstance(x, float):
        return f"{x:{f + 'f'}}"
    else:
        return str(x)


def attribs_string(obj, f='2.3', dl='\n') -> str:
    if hasattr(obj, '__dict__'):
        obj = obj.__dict__
    s = [f"{k}: {to_str(v, f=f)}" for k, v in obj.items()]
    s = dl.join(s)
    return s


def parts(**kwargs) -> str:
    total = sum(kwargs.values())
    tokens = [f"Total={total}"]
    for name, count in kwargs.items():
        tokens.append(f"{name}={count} ({count / total:3.2%})")
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
