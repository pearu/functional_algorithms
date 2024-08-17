# This file is generated using functional_algorithms tool (0.4.0), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def log1p_0(z: float) -> float:
    return math.log1p(z)


def log1p_1(z: complex) -> complex:
    x: float = (z).real
    one: float = 1
    xp1: float = (x) + (one)
    y: float = (z).imag
    ay: float = abs(y)
    axp1: float = abs(xp1)
    mx: float = max(axp1, ay)
    mn: float = min(axp1, ay)
    r: float = (mn) / (mx)
    return complex(
        (math.log1p((x) if ((xp1) >= (ay)) else ((mx) - (one))))
        + ((0.5) * (math.log1p((one) if ((mn) == (mx)) else ((r) * (r))))),
        math.atan2(y, xp1),
    )
