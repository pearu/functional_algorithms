# This file is generated using functional_algorithms tool (0.11.0), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def acosh_0(z: complex) -> complex:
    signed_y: float = (z).imag
    y: float = abs(signed_y)
    signed_x: float = (z).real
    x: float = abs(signed_x)
    safe_max: float = (math.sqrt(sys.float_info.max)) / (8.0)
    safe_max_opt: float = ((safe_max) * (1e-06)) if ((x) < ((safe_max) * (1000000000000.0))) else ((safe_max) * (100.0))
    y_gt_safe_max_opt: bool = (y) >= (safe_max_opt)
    mx: float = (y) if (y_gt_safe_max_opt) else (x)
    two: float = 2.0
    half: float = 0.5
    zero: float = 0.0
    xoy: float = ((x) / (y)) if ((y_gt_safe_max_opt) and (not ((y) == (math.inf)))) else (zero)
    one: float = 1.0
    logical_and_lt_y_safe_min_lt_x_one: bool = ((y) < ((math.sqrt(sys.float_info.min)) * (4.0))) and ((x) < (one))
    xp1: float = (x) + (one)
    xm1: float = (x) - (one)
    abs_xp1: float = abs(xp1)
    _mx_0_: float = max(abs_xp1, y)
    mn: float = min(abs_xp1, y)
    sqrt_two: float = 1.4142135623730951
    mn_over_mx: float = (mn) / (_mx_0_)
    _r_0_: float = (mn_over_mx) * (mn_over_mx)
    sqa: float = math.sqrt((one) + (_r_0_))
    r: float = (
        ((sqrt_two) * (_mx_0_))
        if ((_mx_0_) == (mn))
        else (((_mx_0_) + (((_mx_0_) * (_r_0_)) / (two))) if (((sqa) == (one)) and ((_r_0_) > (zero))) else ((_mx_0_) * (sqa)))
    )
    abs_xm1: float = abs(xm1)
    _mx_1_: float = max(abs_xm1, y)
    _mn_0_: float = min(abs_xm1, y)
    _mn_over_mx_0_: float = (_mn_0_) / (_mx_1_)
    _r_1_: float = (_mn_over_mx_0_) * (_mn_over_mx_0_)
    _sqa_0_: float = math.sqrt((one) + (_r_1_))
    s: float = (
        ((sqrt_two) * (_mx_1_))
        if ((_mx_1_) == (_mn_0_))
        else (
            ((_mx_1_) + (((_mx_1_) * (_r_1_)) / (two)))
            if (((_sqa_0_) == (one)) and ((_r_1_) > (zero)))
            else ((_mx_1_) * (_sqa_0_))
        )
    )
    a: float = (half) * ((r) + (s))
    ap1: float = (a) + (one)
    yy: float = (y) * (y)
    half_yy: float = (half) * (yy)
    rpxp1: float = (r) + (xp1)
    divide_half_yy_rpxp1: float = (half_yy) / (rpxp1)
    spxm1: float = (s) + (xm1)
    smxm1: float = (s) - (xm1)
    x_ge_1_or_not: float = (
        ((divide_half_yy_rpxp1) + ((half) * (spxm1)))
        if ((x) >= (one))
        else (((divide_half_yy_rpxp1) + ((half_yy) / (smxm1))) if ((a) <= (1.5)) else ((a) - (one)))
    )
    am1: float = (-(((xp1) * (xm1)) / (ap1))) if (logical_and_lt_y_safe_min_lt_x_one) else (x_ge_1_or_not)
    sq: float = math.sqrt((am1) * (ap1))
    half_apx: float = (half) * ((a) + (x))
    _imag_0_: float = math.atan2(
        (
            (y)
            if ((max(x, y)) >= (safe_max))
            else (
                (math.sqrt((half_apx) * (((yy) / (rpxp1)) + (smxm1))))
                if ((x) <= (one))
                else ((y) * (math.sqrt(((half_apx) / (rpxp1)) + ((half_apx) / (spxm1)))))
            )
        ),
        signed_x,
    )
    return complex(
        (
            (((math.log(two)) + (math.log(mx))) + ((half) * (math.log1p((xoy) * (xoy)))))
            if ((mx) >= ((safe_max_opt) if (y_gt_safe_max_opt) else (safe_max)))
            else (((y) / (sq)) if (logical_and_lt_y_safe_min_lt_x_one) else (math.log1p((am1) + (sq))))
        ),
        (-(_imag_0_)) if ((signed_y) < (0.0)) else (_imag_0_),
    )


def acosh_1(z: float) -> float:
    constant_f2: float = 2.0
    constant_f1: float = 1.0
    sqrt_subtract_z_constant_f1: float = math.sqrt((z) - (constant_f1))
    return (
        ((math.log(constant_f2)) + (math.log(z)))
        if ((z) >= ((sys.float_info.max) / (constant_f2)))
        else (math.log1p((sqrt_subtract_z_constant_f1) * ((math.sqrt((z) + (constant_f1))) + (sqrt_subtract_z_constant_f1))))
    )
