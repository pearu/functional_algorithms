# This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def absolute_0(z: complex) -> float:
    x: float = (z).real
    abs_x: float = abs(x)
    abs_y: float = abs((z).imag)
    mx: float = max(abs_x, abs_y)
    mn: float = min(abs_x, abs_y)
    constant_f1: float = 1.0
    mn_over_mx: float = (mn) / (mx)
    r: float = (mn_over_mx) * (mn_over_mx)
    sqa: float = math.sqrt((constant_f1) + (r))
    return (
        ((1.4142135623730951) * (mx))
        if ((mx) == (mn))
        else (((mx) + (((mx) * (r)) / (2.0))) if (((sqa) == (constant_f1)) and ((r) > (0.0))) else ((mx) * (sqa)))
    )
