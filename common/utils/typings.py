from numpy.typing import NDArray, ArrayLike
from numpy.random import Generator
from pathlib import Path
from typing import (Any, Sequence, Generator, Iterator, Optional, Container,
                    Iterable, Collection, Sized, Callable, Literal, Hashable)

NpVec = NDArray
NpMatrix = NDArray

NpPoints = NDArray
NpPoint = NDArray

NpPair = NDArray
NpPairs = NDArray

Vec = list | NpVec

PathLike = Path | str

RandState = Generator | int | None
