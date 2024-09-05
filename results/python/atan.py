# This file is generated using functional_algorithms tool (0.4.0), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def atan_0(z: complex) -> complex:
    w: complex = math.atanh(complex(-((z).imag), (z).real))
    return complex((w).imag, -((w).real))


def atan_1(z: float) -> float:
    return math.atan(z)