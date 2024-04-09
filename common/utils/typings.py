from pathlib import Path
from typing import (Generator, Callable, Sequence, Any, Iterable, Iterator, Literal)

from numpy.typing import NDArray

NpVec = NDArray
NpMatrix = NDArray

NpPoints = NDArray
NpPoint = NDArray

NpPair = NDArray
NpPairs = NDArray

Vec = list | NpVec

PathLike = Path | str

RandState = Generator | int | None
