# This file is generated using functional_algorithms tool (0.1.dev6+g8b58236), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def asin_0(z: complex) -> complex:
    signed_x: float = (z).real
    x: float = abs(signed_x)
    signed_y: float = (z).imag
    y: float = abs(signed_y)
    largest: float = sys.float_info.max
    safe_max: float = (math.sqrt(largest)) / (8)
    xp1: float = (x) + (1)
    abs_xp1: float = abs(xp1)
    _hypot_1_mx: float = max(abs_xp1, y)
    mn: float = min(abs_xp1, y)
    sqrt_two: float = math.sqrt(2)
    _square_1_z: float = (mn) / (_hypot_1_mx)
    r: float = (
        ((_hypot_1_mx) * (sqrt_two))
        if ((_hypot_1_mx) == (mn))
        else ((_hypot_1_mx) * (math.sqrt(((_square_1_z) * (_square_1_z)) + (1))))
    )
    xm1: float = (x) - (1)
    abs_xm1: float = abs(xm1)
    _hypot_2_mx: float = max(abs_xm1, y)
    _hypot_2_mn: float = min(abs_xm1, y)
    _square_2_z: float = (_hypot_2_mn) / (_hypot_2_mx)
    s: float = (
        ((_hypot_2_mx) * (sqrt_two))
        if ((_hypot_2_mx) == (_hypot_2_mn))
        else ((_hypot_2_mx) * (math.sqrt(((_square_2_z) * (_square_2_z)) + (1))))
    )
    a: float = (0.5) * ((r) + (s))
    half_apx: float = (0.5) * ((a) + (x))
    yy: float = (y) * (y)
    rpxp1: float = (r) + (xp1)
    smxm1: float = (s) - (xm1)
    spxm1: float = (s) + (xm1)
    real: float = math.atan2(
        signed_x,
        (
            (y)
            if ((max(x, y)) >= (safe_max))
            else (
                (math.sqrt((half_apx) * (((yy) / (rpxp1)) + (smxm1))))
                if ((x) <= (1))
                else ((y) * (math.sqrt(((half_apx) / (rpxp1)) + ((half_apx) / (spxm1)))))
            )
        ),
    )
    safe_max_opt: float = ((safe_max) * (1e-06)) if ((x) < ((safe_max) * (1000000000000.0))) else ((safe_max) * (100.0))
    y_gt_safe_max_opt: bool = (y) >= (safe_max_opt)
    mx: float = (y) if (y_gt_safe_max_opt) else (x)
    posinf: float = math.inf
    xoy: float = ((x) / (y)) if ((y_gt_safe_max_opt) and (not ((y) == (posinf)))) else (0)
    smallest: float = sys.float_info.min
    logical_and_lt_y_safe_min_lt_x_one: bool = ((y) < ((math.sqrt(smallest)) * (4))) and ((x) < (1))
    ap1: float = (a) + (1)
    half_yy: float = (0.5) * (yy)
    divide_half_yy_rpxp1: float = (half_yy) / (rpxp1)
    x_ge_1_or_not: float = (
        ((divide_half_yy_rpxp1) + ((0.5) * (spxm1)))
        if ((x) >= (1))
        else (((divide_half_yy_rpxp1) + ((half_yy) / (smxm1))) if ((a) <= (1.5)) else ((a) - (1)))
    )
    am1: float = (-(((xp1) * (xm1)) / (ap1))) if (logical_and_lt_y_safe_min_lt_x_one) else (x_ge_1_or_not)
    sq: float = math.sqrt((am1) * (ap1))
    imag: float = (
        (((math.log(2)) + (math.log(mx))) + ((0.5) * (math.log1p((xoy) * (xoy)))))
        if ((mx) >= ((safe_max_opt) if (y_gt_safe_max_opt) else (safe_max)))
        else (((y) / (sq)) if (logical_and_lt_y_safe_min_lt_x_one) else (math.log1p((am1) + (sq))))
    )
    return complex(real, (-(imag)) if ((signed_y) < (0)) else (imag))
