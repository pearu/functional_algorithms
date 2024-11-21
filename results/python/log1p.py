# This file is generated using functional_algorithms tool (0.11.1), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import math
import sys


def log1p_1(z: complex) -> complex:
    x: float = (z).real
    ax: float = abs(x)
    y: float = (z).imag
    ay: float = abs(y)
    mx: float = max(ax, ay)
    largest: float = sys.float_info.max
    half: float = 0.5
    mn: float = min(ax, ay)
    one: float = 1.0
    r: float = (mn) / (mx)
    xp1: float = (x) + (one)
    square_dekker_high: float = (y) * (y)
    x2h: float = (x) + (x)
    _add_2sum_high_2_: float = (x2h) + (square_dekker_high)
    _square_dekker_high_0_: float = (x) * (x)
    _add_2sum_high_1_: float = (_add_2sum_high_2_) + (_square_dekker_high_0_)
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
    _add_2sum_high_0_: float = (_add_2sum_high_1_) + (square_dekker_low)
    multiply_veltkamp_splitter_constant_x: float = (veltkamp_splitter_constant) * (x)
    xh: float = (multiply_veltkamp_splitter_constant_x) + ((x) - (multiply_veltkamp_splitter_constant_x))
    xl: float = (x) - (xh)
    multiply_xh_xl: float = (xh) * (xl)
    _square_dekker_low_0_: float = ((((-(_square_dekker_high_0_)) + ((xh) * (xh))) + (multiply_xh_xl)) + (multiply_xh_xl)) + (
        (xl) * (xl)
    )
    add_2sum_high: float = (_add_2sum_high_0_) + (_square_dekker_low_0_)
    subtract__add_2sum_high_2__x2h: float = (_add_2sum_high_2_) - (x2h)
    add_2sum_low: float = ((x2h) - ((_add_2sum_high_2_) - (subtract__add_2sum_high_2__x2h))) + (
        (square_dekker_high) - (subtract__add_2sum_high_2__x2h)
    )
    subtract__add_2sum_high_1___add_2sum_high_2_: float = (_add_2sum_high_1_) - (_add_2sum_high_2_)
    _add_2sum_low_0_: float = (
        (_add_2sum_high_2_) - ((_add_2sum_high_1_) - (subtract__add_2sum_high_1___add_2sum_high_2_))
    ) + ((_square_dekker_high_0_) - (subtract__add_2sum_high_1___add_2sum_high_2_))
    subtract__add_2sum_high_0___add_2sum_high_1_: float = (_add_2sum_high_0_) - (_add_2sum_high_1_)
    _add_2sum_low_1_: float = (
        (_add_2sum_high_1_) - ((_add_2sum_high_0_) - (subtract__add_2sum_high_0___add_2sum_high_1_))
    ) + ((square_dekker_low) - (subtract__add_2sum_high_0___add_2sum_high_1_))
    subtract_add_2sum_high__add_2sum_high_0_: float = (add_2sum_high) - (_add_2sum_high_0_)
    _add_2sum_low_2_: float = ((_add_2sum_high_0_) - ((add_2sum_high) - (subtract_add_2sum_high__add_2sum_high_0_))) + (
        (_square_dekker_low_0_) - (subtract_add_2sum_high__add_2sum_high_0_)
    )
    sum_2sum_high: float = (add_2sum_high) + (
        (((add_2sum_low) + (_add_2sum_low_0_)) + (_add_2sum_low_1_)) + (_add_2sum_low_2_)
    )
    return complex(
        (
            ((math.log(mx)) + ((half) * (math.log1p((one) if ((mn) == (mx)) else ((r) * (r))))))
            if ((mx) > ((math.sqrt(largest)) * (0.01)))
            else (
                ((half) * (math.log(((xp1) * (xp1)) + (square_dekker_high))))
                if (((abs(xp1)) + (ay)) < (0.2))
                else ((half) * (math.log1p(sum_2sum_high)))
            )
        ),
        math.atan2(y, xp1),
    )
