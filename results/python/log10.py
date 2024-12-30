# This file is generated using functional_algorithms tool (0.13.3.dev1+g8d134ad.d20241230), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def log10_0(z: complex) -> complex:
    lnz: complex = math.log(z)
    x: float = (lnz).real
    ln10: float = math.log(10.0)
    return complex((x) / (ln10), ((lnz).imag) / (ln10))
