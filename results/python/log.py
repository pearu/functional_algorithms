# This file is generated using functional_algorithms tool (0.14.1.dev0+ge22be68.d20241231), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def log_0(z: complex) -> complex:
    constant_fneg1: float = -1.0
    y: float = (z).imag
    square_dekker_high: float = (y) * (y)
    x: float = (z).real
    _square_dekker_high_0_: float = (x) * (x)
    gt_square_dekker_high__square_dekker_high_0_: bool = (square_dekker_high) > (_square_dekker_high_0_)
    mxh: float = (square_dekker_high) if (gt_square_dekker_high__square_dekker_high_0_) else (_square_dekker_high_0_)
    _add_fast2sum_high_2_: float = (constant_fneg1) + (mxh)
    mnh: float = (_square_dekker_high_0_) if (gt_square_dekker_high__square_dekker_high_0_) else (square_dekker_high)
    _add_fast2sum_high_1_: float = (_add_fast2sum_high_2_) + (mnh)
    largest: float = sys.float_info.max
    veltkamp_splitter_constant: float = (
        (134217729.0) if ((largest) > (1e308)) else ((4097.0) if ((largest) > (1e38)) else (65.0))
    )
    multiply_veltkamp_splitter_constant_y: float = (veltkamp_splitter_constant) * (y)
    yh: float = (multiply_veltkamp_splitter_constant_y) + ((y) - (multiply_veltkamp_splitter_constant_y))
    yl: float = (y) - (yh)
    multiply_yh_yl: float = (yh) * (yl)
    square_dekker_low: float = ((((-(square_dekker_high)) + ((yh) * (yh))) + (multiply_yh_yl)) + (multiply_yh_yl)) + (
        (yl) * (yl)
    )
    _add_fast2sum_high_0_: float = (_add_fast2sum_high_1_) + (square_dekker_low)
    multiply_veltkamp_splitter_constant_x: float = (veltkamp_splitter_constant) * (x)
    xh: float = (multiply_veltkamp_splitter_constant_x) + ((x) - (multiply_veltkamp_splitter_constant_x))
    xl: float = (x) - (xh)
    multiply_xh_xl: float = (xh) * (xl)
    _square_dekker_low_0_: float = ((((-(_square_dekker_high_0_)) + ((xh) * (xh))) + (multiply_xh_xl)) + (multiply_xh_xl)) + (
        (xl) * (xl)
    )
    add_fast2sum_high: float = (_add_fast2sum_high_0_) + (_square_dekker_low_0_)
    add_fast2sum_low: float = (mxh) - ((_add_fast2sum_high_2_) - (constant_fneg1))
    _add_fast2sum_low_0_: float = (mnh) - ((_add_fast2sum_high_1_) - (_add_fast2sum_high_2_))
    _add_fast2sum_low_1_: float = (square_dekker_low) - ((_add_fast2sum_high_0_) - (_add_fast2sum_high_1_))
    _add_fast2sum_low_2_: float = (_square_dekker_low_0_) - ((add_fast2sum_high) - (_add_fast2sum_high_0_))
    sum_fast2sum_high: float = (add_fast2sum_high) + (
        (((add_fast2sum_low) + (_add_fast2sum_low_0_)) + (_add_fast2sum_low_1_)) + (_add_fast2sum_low_2_)
    )
    half: float = 0.5
    abs_x: float = abs(x)
    abs_y: float = abs(y)
    mx: float = max(abs_x, abs_y)
    mn: float = min(abs_x, abs_y)
    mn_over_mx: float = (1.0) if ((mn) == (mx)) else ((mn) / (mx))
    return complex(
        (
            ((half) * (math.log1p(sum_fast2sum_high)))
            if ((abs(sum_fast2sum_high)) < (half))
            else ((math.log(mx)) + ((half) * (math.log1p((mn_over_mx) * (mn_over_mx)))))
        ),
        math.atan2(y, x),
    )
