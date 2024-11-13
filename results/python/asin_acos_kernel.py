# This file is generated using functional_algorithms tool (0.11.0), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def asin_acos_kernel_0(z: complex) -> complex:
    signed_x: float = (z).real
    x: float = abs(signed_x)
    signed_y: float = (z).imag
    y: float = abs(signed_y)
    safe_max: float = (math.sqrt(sys.float_info.max)) / (8.0)
    one: float = 1.0
    half: float = 0.5
    xp1: float = (x) + (one)
    abs_xp1: float = abs(xp1)
    mx: float = max(abs_xp1, y)
    mn: float = min(abs_xp1, y)
    sqrt_two: float = 1.4142135623730951
    mn_over_mx: float = (mn) / (mx)
    _r_0_: float = (mn_over_mx) * (mn_over_mx)
    sqa: float = math.sqrt((one) + (_r_0_))
    zero: float = 0.0
    two: float = 2.0
    r: float = (
        ((sqrt_two) * (mx))
        if ((mx) == (mn))
        else (((mx) + (((mx) * (_r_0_)) / (two))) if (((sqa) == (one)) and ((_r_0_) > (zero))) else ((mx) * (sqa)))
    )
    xm1: float = (x) - (one)
    abs_xm1: float = abs(xm1)
    _mx_0_: float = max(abs_xm1, y)
    _mn_0_: float = min(abs_xm1, y)
    _mn_over_mx_0_: float = (_mn_0_) / (_mx_0_)
    _r_1_: float = (_mn_over_mx_0_) * (_mn_over_mx_0_)
    _sqa_0_: float = math.sqrt((one) + (_r_1_))
    s: float = (
        ((sqrt_two) * (_mx_0_))
        if ((_mx_0_) == (_mn_0_))
        else (
            ((_mx_0_) + (((_mx_0_) * (_r_1_)) / (two)))
            if (((_sqa_0_) == (one)) and ((_r_1_) > (zero)))
            else ((_mx_0_) * (_sqa_0_))
        )
    )
    a: float = (half) * ((r) + (s))
    half_apx: float = (half) * ((a) + (x))
    yy: float = (y) * (y)
    rpxp1: float = (r) + (xp1)
    smxm1: float = (s) - (xm1)
    spxm1: float = (s) + (xm1)
    safe_max_opt: float = ((safe_max) * (1e-06)) if ((x) < ((safe_max) * (1000000000000.0))) else ((safe_max) * (100.0))
    y_gt_safe_max_opt: bool = (y) >= (safe_max_opt)
    _mx_1_: float = (y) if (y_gt_safe_max_opt) else (x)
    xoy: float = ((x) / (y)) if ((y_gt_safe_max_opt) and (not ((y) == (math.inf)))) else (zero)
    logical_and_lt_y_safe_min_lt_x_one: bool = ((y) < ((math.sqrt(sys.float_info.min)) * (4.0))) and ((x) < (one))
    ap1: float = (a) + (one)
    half_yy: float = (half) * (yy)
    divide_half_yy_rpxp1: float = (half_yy) / (rpxp1)
    x_ge_1_or_not: float = (
        ((divide_half_yy_rpxp1) + ((half) * (spxm1)))
        if ((x) >= (one))
        else (((divide_half_yy_rpxp1) + ((half_yy) / (smxm1))) if ((a) <= (1.5)) else ((a) - (one)))
    )
    am1: float = (-(((xp1) * (xm1)) / (ap1))) if (logical_and_lt_y_safe_min_lt_x_one) else (x_ge_1_or_not)
    sq: float = math.sqrt((am1) * (ap1))
    return complex(
        (
            (y)
            if ((max(x, y)) >= (safe_max))
            else (
                (math.sqrt((half_apx) * (((yy) / (rpxp1)) + (smxm1))))
                if ((x) <= (one))
                else ((y) * (math.sqrt(((half_apx) / (rpxp1)) + ((half_apx) / (spxm1)))))
            )
        ),
        (
            (((math.log(two)) + (math.log(_mx_1_))) + ((half) * (math.log1p((xoy) * (xoy)))))
            if ((_mx_1_) >= ((safe_max_opt) if (y_gt_safe_max_opt) else (safe_max)))
            else (((y) / (sq)) if (logical_and_lt_y_safe_min_lt_x_one) else (math.log1p((am1) + (sq))))
        ),
    )
