# This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def square_0(z: float) -> float:
    return (z) * (z)


def square_1(z: complex) -> complex:
    x: float = (z).real
    y: float = (z).imag
    return complex(
        (0.0) if ((math.isfinite(x)) and ((abs(x)) == (abs(y)))) else (((x) - (y)) * ((x) + (y))), (2.0) * ((x) * (y))
    )
