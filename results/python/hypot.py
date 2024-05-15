# This file is generated using functional_algorithms tool (0.1.dev6+g8b58236), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def hypot_0(x: float, y: float) -> float:
    abs_x: float = abs(x)
    abs_y: float = abs(y)
    mx: float = max(abs_x, abs_y)
    mn: float = min(abs_x, abs_y)
    z: float = (mn) / (mx)
    return ((mx) * (math.sqrt(2))) if ((mx) == (mn)) else ((mx) * (math.sqrt(((z) * (z)) + (1))))
