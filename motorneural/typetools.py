from typing import Sequence, Literal, Annotated, TypeVar, Callable
import numpy as np
from numpy.typing import NDArray

_T = TypeVar('_T')

Pair = Annotated[Sequence[_T], "Shape[1,2]"]
Vec = Annotated[Sequence[_T], "Shape[1,*]"]

NpPair = Annotated[NDArray[_T], "Shape[1,2]"]
NpVec = Annotated[NDArray[_T], "Shape[1,*]"]
NpPoints = Annotated[NDArray[_T], "Shape[*,*]"]
NpMatrix = Annotated[NDArray[_T], "Shape[*,*]"]
NpImage = Annotated[NDArray[_T], "Shape[*,*,*]"]


def is_sorted(x: NpVec[float], order: Literal['a', 'd'] = 'a'):
    if order == 'a':
        return np.all(x[:-1] <= x[1:])
    return np.all(x[:-1] >= x[1:])
