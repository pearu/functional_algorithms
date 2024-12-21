# This file is generated using functional_algorithms tool (0.11.1), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def sqrt_0(z: complex) -> complex:
    x: float = (z).real
    constant_f0: float = 0.0
    ax: float = abs(x)
    y: float = (z).imag
    ay: float = abs(y)
    eq_ax_ay: bool = (ax) == (ay)
    sq_ax: float = math.sqrt(ax)
    sq_2: float = 1.4142135623730951
    mx: float = max(ax, ay)
    mn: float = min(ax, ay)
    one: float = 1.0
    mn_over_mx: float = (mn) / (mx)
    r: float = (mn_over_mx) * (mn_over_mx)
    sqa: float = math.sqrt((one) + (r))
    two: float = 2.0
    u_general: float = math.sqrt(
        (
            (
                ((sq_2) * (mx))
                if ((mx) == (mn))
                else (((mx) + (((mx) * (r)) / (two))) if (((sqa) == (one)) and ((r) > (constant_f0))) else ((mx) * (sqa)))
            )
            / (two)
        )
        + ((ax) / (two))
    )
    logical_or_eq_u_general_constant_f0_eq_u_general_constant_posinf: bool = ((u_general) == (constant_f0)) or (
        (u_general) == (math.inf)
    )
    gt_ax_ay: bool = (ax) > (ay)
    lt_ax_ay: bool = (ax) < (ay)
    _r_0_: float = (one) if (eq_ax_ay) else (((ax) / (ay)) if (lt_ax_ay) else ((ay) / (ax)))
    abs__r_0_: float = abs(_r_0_)
    _mx_0_: float = max(one, abs__r_0_)
    _mn_0_: float = min(one, abs__r_0_)
    _mn_over_mx_0_: float = (_mn_0_) / (_mx_0_)
    _r_1_: float = (_mn_over_mx_0_) * (_mn_over_mx_0_)
    _sqa_0_: float = math.sqrt((one) + (_r_1_))
    h: float = (
        ((sq_2) * (_mx_0_))
        if ((_mx_0_) == (_mn_0_))
        else (
            ((_mx_0_) + (((_mx_0_) * (_r_1_)) / (two)))
            if (((_sqa_0_) == (one)) and ((_r_1_) > (constant_f0)))
            else ((_mx_0_) * (_sqa_0_))
        )
    )
    sq_1h: float = math.sqrt((one) + (h))
    sq_ay: float = math.sqrt(ay)
    sq_rh: float = math.sqrt((_r_0_) + (h))
    u: float = (
        (((sq_ax) * (1.5537739740300374)) / (sq_2))
        if (eq_ax_ay)
        else (
            (((sq_ax) * ((sq_1h) / (sq_2))) if (gt_ax_ay) else ((sq_ay) * ((sq_rh) / (sq_2))))
            if (logical_or_eq_u_general_constant_f0_eq_u_general_constant_posinf)
            else (u_general)
        )
    )
    ay_div_u: float = (
        ((sq_ay) / (2.19736822693562))
        if (eq_ax_ay)
        else (
            (
                (
                    ((sq_ay) * ((one) if (eq_ax_ay) else (((sq_ax) / (sq_ay)) if (lt_ax_ay) else ((sq_ay) / (sq_ax)))))
                    / ((sq_1h) * (sq_2))
                )
                if (gt_ax_ay)
                else ((sq_ay) / ((sq_rh) * (sq_2)))
            )
            if (logical_or_eq_u_general_constant_f0_eq_u_general_constant_posinf)
            else ((ay) / ((u_general) * (two)))
        )
    )
    lt_y_constant_f0: bool = (y) < (constant_f0)
    return complex(
        (u) if ((x) >= (constant_f0)) else (ay_div_u),
        (
            ((-(u)) if (lt_y_constant_f0) else (u))
            if ((x) < (constant_f0))
            else ((-(ay_div_u)) if (lt_y_constant_f0) else (ay_div_u))
        ),
    )
