# This file is generated using functional_algorithms tool (0.1.2.dev7+g332df57.d20240604), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def square_0(z: float) -> float:
    return (z) * (z)


def square_1(z: complex) -> complex:
    x: float = (z).real
    y: float = (z).imag
    return complex((0) if ((abs(x)) == (abs(y))) else (((x) - (y)) * ((x) + (y))), (2) * ((x) * (y)))
