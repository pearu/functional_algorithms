# This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def atanh_0(z: complex) -> complex:
    x: float = (z).real
    zero: float = 0.0
    one: float = 1.0
    constant_fneg1: float = -1.0
    ax: float = abs(x)
    largest: float = sys.float_info.max
    inv_negeps: float = (9007199254740994.0) if ((largest) > (1e308)) else ((16777218.0) if ((largest) > (1e38)) else (2050.0))
    safe_max: float = (inv_negeps) * (inv_negeps)
    y: float = (z).imag
    ay: float = abs(y)
    in_safe_region: bool = ((ax) < (safe_max)) and ((ay) < (safe_max))
    naxm1: float = (one) - (ax)
    y2: float = (y) * (y)
    constant_posinf: float = math.inf
    constant_neginf: float = -math.inf
    return complex(
        (
            ((one) if ((x) >= (zero)) else (constant_fneg1))
            * (
                math.log1p(
                    (4.0)
                    * (
                        ((ax) / (((naxm1) * (naxm1)) + (y2)))
                        if (in_safe_region)
                        else (
                            ((one) / (ax))
                            if (((ay) * (inv_negeps)) < (ax))
                            else (
                                (zero)
                                if (
                                    (((x) == (constant_posinf)) or ((x) == (constant_neginf)))
                                    or (((y) == (constant_posinf)) or ((y) == (constant_neginf)))
                                )
                                else (((one) / (((ax) / (y)) + ((y) / (ax)))) / (y))
                            )
                        )
                    )
                )
            )
        )
        * (0.25),
        (
            (math.atan2((y) + (y), ((naxm1) * ((one) + (ax))) - (y2)))
            if (in_safe_region)
            else (((one) if ((y) >= (0.0)) else (constant_fneg1)) * (math.pi))
        )
        * (0.5),
    )
