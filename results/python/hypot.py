# This file is generated using functional_algorithms tool (0.1.2.dev7+g332df57.d20240604), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def hypot_0(x: float, y: float) -> float:
    abs_x: float = abs(x)
    abs_y: float = abs(y)
    mx: float = max(abs_x, abs_y)
    mn: float = min(abs_x, abs_y)
    constant_2: float = 2
    constant_1: float = 1
    mn_over_mx: float = (mn) / (mx)
    r: float = (mn_over_mx) * (mn_over_mx)
    sqa: float = math.sqrt((constant_1) + (r))
    return (
        ((math.sqrt(constant_2)) * (mx))
        if ((mx) == (mn))
        else (((mx) + (((mx) * (r)) / (constant_2))) if (((sqa) == (constant_1)) and ((r) > (0))) else ((mx) * (sqa)))
    )
