import itertools
from typing import NamedTuple, Callable, Any, Union


class MVNStandard(NamedTuple):
    mean: Any
    cov: Any


class LinearIntegrated(NamedTuple):
    A: Any
    A_bar: Any
    B_bar: Any
    b_bar: Any
    cov: Any

class LinearIntegratedObs(NamedTuple):
    C: Any
    cov: Any