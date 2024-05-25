# This file is generated using functional_algorithms tool (0.1.2.dev2+g1428951.d20240525), see
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
    mn_over_mx: float = (mn) / (mx)
    r: float = (mn_over_mx) * (mn_over_mx)
    sqa: float = math.sqrt((1) + (r))
    return (
        ((math.sqrt(2)) * (mx))
        if ((mx) == (mn))
        else (((mx) + (((mx) * (r)) / (2))) if (((sqa) == (1)) and ((r) > (0))) else ((mx) * (sqa)))
    )
