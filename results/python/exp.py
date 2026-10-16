# This file is generated using functional_algorithms tool (0.15.1.dev20+g7dbbfd7db.d20261016), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def exp_0(z: complex) -> complex:
    y: float = (z).imag
    y_is_not_finite: bool = not (math.isfinite(y))
    x: float = (z).real
    constant_posinf: float = math.inf
    eq_x_constant_neginf: bool = (x) == (-math.inf)
    e: float = math.exp(x)
    eq_e_constant_posinf: bool = (e) == (constant_posinf)
    e2: float = math.exp((x) * (0.5))
    cs: float = math.cos(y)
    zero: float = 0.0
    sn: float = math.sin(y)
    return complex(
        (
            (e)
            if ((y_is_not_finite) and (((x) == (constant_posinf)) or (eq_x_constant_neginf)))
            else ((((e2) * (cs)) * (e2)) if (eq_e_constant_posinf) else ((e) * (cs)))
        ),
        (
            ((((e2) * (sn)) * (e2)) if (eq_e_constant_posinf) else ((e) * (sn)))
            if ((not ((eq_x_constant_neginf) and (y_is_not_finite))) and ((y) != (zero)))
            else (zero)
        ),
    )
