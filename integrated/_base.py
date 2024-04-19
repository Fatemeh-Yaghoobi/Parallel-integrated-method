import itertools
from typing import NamedTuple, Callable, Any, Union


class MVNStandard(NamedTuple):
    mean: Any
    cov: Any

class LinearTran(NamedTuple):
    # Equation 8_a: A x + B u + w, w ~ N(0, Q)
    A: Any
    B: Any
    u: Any
    Q: Any


class LinearObs(NamedTuple):
    # Equation 8_c: C x + v, v ~ N(0, R)
    C: Any
    R: Any


class SlowRateIntegratedParams(NamedTuple):
    # Equations 9 and 10: h_{k, l} = A_bar x_{k-1, l} + B_bar u_bar_{k, l} + G_bar w_bar_{k},   G_bar w_bar_{k} ~ N(0, Q_bar)
    A_bar: Any
    B_bar: Any
    u_bar: Any
    G_bar: Any


class FastRateIntegratedParams(NamedTuple):
    pass
