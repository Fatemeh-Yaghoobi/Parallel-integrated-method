import itertools
from typing import NamedTuple, Callable, Any, Union


class MVNStandard(NamedTuple):
    mean: Any
    cov: Any


class LinearIntegrated(NamedTuple):
    F: Any
    G: Any
    b: Any
    cov: Any