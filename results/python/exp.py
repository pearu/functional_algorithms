# This file is generated using functional_algorithms tool (0.15.1.dev3+ge93b47e.d20250113), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def exp_0(z: complex) -> complex:
    x: float = (z).real
    e: float = math.exp(x)
    eq_e_constant_posinf: bool = (e) == (math.inf)
    e2: float = math.exp((x) * (0.5))
    y: float = (z).imag
    cs: float = math.cos(y)
    zero: float = 0.0
    sn: float = math.sin(y)
    return complex(
        (((e2) * (cs)) * (e2)) if (eq_e_constant_posinf) else ((e) * (cs)),
        (zero) if ((y) == (zero)) else ((((e2) * (sn)) * (e2)) if (eq_e_constant_posinf) else ((e) * (sn))),
    )
