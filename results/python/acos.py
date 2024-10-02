# This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def acos_0(z: complex) -> complex:
    signed_x: float = (z).real
    x: float = abs(signed_x)
    signed_y: float = (z).imag
    y: float = abs(signed_y)
    safe_max: float = (math.sqrt(sys.float_info.max)) / (8)
    one: float = 1
    half: float = 0.5
    xp1: float = (x) + (one)
    abs_xp1: float = abs(xp1)
    mx: float = max(abs_xp1, y)
    mn: float = min(abs_xp1, y)
    two: float = 2
    sqrt_two: float = math.sqrt(two)
    mn_over_mx: float = (mn) / (mx)
    r__0: float = (mn_over_mx) * (mn_over_mx)
    sqa: float = math.sqrt((one) + (r__0))
    zero: float = 0
    r: float = (
        ((sqrt_two) * (mx))
        if ((mx) == (mn))
        else (((mx) + (((mx) * (r__0)) / (two))) if (((sqa) == (one)) and ((r__0) > (zero))) else ((mx) * (sqa)))
    )
    xm1: float = (x) - (one)
    abs_xm1: float = abs(xm1)
    mx__0: float = max(abs_xm1, y)
    mn__0: float = min(abs_xm1, y)
    mn_over_mx__0: float = (mn__0) / (mx__0)
    r__1: float = (mn_over_mx__0) * (mn_over_mx__0)
    sqa__0: float = math.sqrt((one) + (r__1))
    s: float = (
        ((sqrt_two) * (mx__0))
        if ((mx__0) == (mn__0))
        else (
            ((mx__0) + (((mx__0) * (r__1)) / (two))) if (((sqa__0) == (one)) and ((r__1) > (zero))) else ((mx__0) * (sqa__0))
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
    _asin_acos_kernel_1_mx__0: float = (y) if (y_gt_safe_max_opt) else (x)
    xoy: float = ((x) / (y)) if ((y_gt_safe_max_opt) and (not ((y) == (math.inf)))) else (zero)
    logical_and_lt_y_safe_min_lt_x_one: bool = ((y) < ((math.sqrt(sys.float_info.min)) * (4))) and ((x) < (one))
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
    imag: float = (
        (((math.log(two)) + (math.log(_asin_acos_kernel_1_mx__0))) + ((half) * (math.log1p((xoy) * (xoy)))))
        if ((_asin_acos_kernel_1_mx__0) >= ((safe_max_opt) if (y_gt_safe_max_opt) else (safe_max)))
        else (((y) / (sq)) if (logical_and_lt_y_safe_min_lt_x_one) else (math.log1p((am1) + (sq))))
    )
    return complex(
        math.atan2(
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
        ),
        (imag) if ((signed_y) < (0)) else (-(imag)),
    )


def acos_1(z: float) -> float:
    constant_1: float = 1
    return math.atan2(math.sqrt(((constant_1) - (z)) * ((constant_1) + (z))), z)
