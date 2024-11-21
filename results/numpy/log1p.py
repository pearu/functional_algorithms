# This file is generated using functional_algorithms tool (0.11.1), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import numpy
import warnings


def make_complex(r, i):
    if r.dtype == numpy.float32 and i.dtype == numpy.float32:
        return numpy.array([r, i]).view(numpy.complex64)[0]
    elif i.dtype == numpy.float64 and i.dtype == numpy.float64:
        return numpy.array([r, i]).view(numpy.complex128)[0]
    raise NotImplementedError((r.dtype, i.dtype))


def log1p_0(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        x: numpy.float64 = (z).real
        ax: numpy.float64 = numpy.abs(x)
        y: numpy.float64 = (z).imag
        ay: numpy.float64 = numpy.abs(y)
        mx: numpy.float64 = max(ax, ay)
        half: numpy.float64 = numpy.float64(0.5)
        mn: numpy.float64 = min(ax, ay)
        one: numpy.float64 = numpy.float64(1.0)
        r: numpy.float64 = (mn) / (mx)
        xp1: numpy.float64 = (x) + (one)
        square_dekker_high: numpy.float64 = (y) * (y)
        x2h: numpy.float64 = (x) + (x)
        _add_2sum_high_2_: numpy.float64 = (x2h) + (square_dekker_high)
        _square_dekker_high_0_: numpy.float64 = (x) * (x)
        _add_2sum_high_1_: numpy.float64 = (_add_2sum_high_2_) + (_square_dekker_high_0_)
        veltkamp_splitter_constant: numpy.float64 = numpy.float64(134217729.0)
        multiply_veltkamp_splitter_constant_y: numpy.float64 = (veltkamp_splitter_constant) * (y)
        yh: numpy.float64 = (multiply_veltkamp_splitter_constant_y) + ((y) - (multiply_veltkamp_splitter_constant_y))
        yl: numpy.float64 = (y) - (yh)
        multiply_yh_yl: numpy.float64 = (yh) * (yl)
        square_dekker_low: numpy.float64 = (
            (((-(square_dekker_high)) + ((yh) * (yh))) + (multiply_yh_yl)) + (multiply_yh_yl)
        ) + ((yl) * (yl))
        _add_2sum_high_0_: numpy.float64 = (_add_2sum_high_1_) + (square_dekker_low)
        multiply_veltkamp_splitter_constant_x: numpy.float64 = (veltkamp_splitter_constant) * (x)
        xh: numpy.float64 = (multiply_veltkamp_splitter_constant_x) + ((x) - (multiply_veltkamp_splitter_constant_x))
        xl: numpy.float64 = (x) - (xh)
        multiply_xh_xl: numpy.float64 = (xh) * (xl)
        _square_dekker_low_0_: numpy.float64 = (
            (((-(_square_dekker_high_0_)) + ((xh) * (xh))) + (multiply_xh_xl)) + (multiply_xh_xl)
        ) + ((xl) * (xl))
        add_2sum_high: numpy.float64 = (_add_2sum_high_0_) + (_square_dekker_low_0_)
        subtract__add_2sum_high_2__x2h: numpy.float64 = (_add_2sum_high_2_) - (x2h)
        add_2sum_low: numpy.float64 = ((x2h) - ((_add_2sum_high_2_) - (subtract__add_2sum_high_2__x2h))) + (
            (square_dekker_high) - (subtract__add_2sum_high_2__x2h)
        )
        subtract__add_2sum_high_1___add_2sum_high_2_: numpy.float64 = (_add_2sum_high_1_) - (_add_2sum_high_2_)
        _add_2sum_low_0_: numpy.float64 = (
            (_add_2sum_high_2_) - ((_add_2sum_high_1_) - (subtract__add_2sum_high_1___add_2sum_high_2_))
        ) + ((_square_dekker_high_0_) - (subtract__add_2sum_high_1___add_2sum_high_2_))
        subtract__add_2sum_high_0___add_2sum_high_1_: numpy.float64 = (_add_2sum_high_0_) - (_add_2sum_high_1_)
        _add_2sum_low_1_: numpy.float64 = (
            (_add_2sum_high_1_) - ((_add_2sum_high_0_) - (subtract__add_2sum_high_0___add_2sum_high_1_))
        ) + ((square_dekker_low) - (subtract__add_2sum_high_0___add_2sum_high_1_))
        subtract_add_2sum_high__add_2sum_high_0_: numpy.float64 = (add_2sum_high) - (_add_2sum_high_0_)
        _add_2sum_low_2_: numpy.float64 = (
            (_add_2sum_high_0_) - ((add_2sum_high) - (subtract_add_2sum_high__add_2sum_high_0_))
        ) + ((_square_dekker_low_0_) - (subtract_add_2sum_high__add_2sum_high_0_))
        sum_2sum_high: numpy.float64 = (add_2sum_high) + (
            (((add_2sum_low) + (_add_2sum_low_0_)) + (_add_2sum_low_1_)) + (_add_2sum_low_2_)
        )
        result = make_complex(
            (
                (
                    (numpy.log(mx))
                    + ((half) * (numpy.log1p((one) if (numpy.equal(mn, mx, dtype=numpy.bool_)) else ((r) * (r)))))
                )
                if ((mx) > (numpy.float64(1.3407807929942595e152)))
                else (
                    ((half) * (numpy.log(((xp1) * (xp1)) + (square_dekker_high))))
                    if (((numpy.abs(xp1)) + (ay)) < (numpy.float64(0.2)))
                    else ((half) * (numpy.log1p(sum_2sum_high)))
                )
            ),
            numpy.arctan2(y, xp1),
        )
        return result


def log1p_1(z: numpy.complex64) -> numpy.complex64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        x: numpy.float32 = (z).real
        ax: numpy.float32 = numpy.abs(x)
        y: numpy.float32 = (z).imag
        ay: numpy.float32 = numpy.abs(y)
        mx: numpy.float32 = max(ax, ay)
        half: numpy.float32 = numpy.float32(0.5)
        mn: numpy.float32 = min(ax, ay)
        one: numpy.float32 = numpy.float32(1.0)
        r: numpy.float32 = (mn) / (mx)
        xp1: numpy.float32 = (x) + (one)
        square_dekker_high: numpy.float32 = (y) * (y)
        x2h: numpy.float32 = (x) + (x)
        _add_2sum_high_2_: numpy.float32 = (x2h) + (square_dekker_high)
        _square_dekker_high_0_: numpy.float32 = (x) * (x)
        _add_2sum_high_1_: numpy.float32 = (_add_2sum_high_2_) + (_square_dekker_high_0_)
        veltkamp_splitter_constant: numpy.float32 = numpy.float32(4097.0)
        multiply_veltkamp_splitter_constant_y: numpy.float32 = (veltkamp_splitter_constant) * (y)
        yh: numpy.float32 = (multiply_veltkamp_splitter_constant_y) + ((y) - (multiply_veltkamp_splitter_constant_y))
        yl: numpy.float32 = (y) - (yh)
        multiply_yh_yl: numpy.float32 = (yh) * (yl)
        square_dekker_low: numpy.float32 = (
            (((-(square_dekker_high)) + ((yh) * (yh))) + (multiply_yh_yl)) + (multiply_yh_yl)
        ) + ((yl) * (yl))
        _add_2sum_high_0_: numpy.float32 = (_add_2sum_high_1_) + (square_dekker_low)
        multiply_veltkamp_splitter_constant_x: numpy.float32 = (veltkamp_splitter_constant) * (x)
        xh: numpy.float32 = (multiply_veltkamp_splitter_constant_x) + ((x) - (multiply_veltkamp_splitter_constant_x))
        xl: numpy.float32 = (x) - (xh)
        multiply_xh_xl: numpy.float32 = (xh) * (xl)
        _square_dekker_low_0_: numpy.float32 = (
            (((-(_square_dekker_high_0_)) + ((xh) * (xh))) + (multiply_xh_xl)) + (multiply_xh_xl)
        ) + ((xl) * (xl))
        add_2sum_high: numpy.float32 = (_add_2sum_high_0_) + (_square_dekker_low_0_)
        subtract__add_2sum_high_2__x2h: numpy.float32 = (_add_2sum_high_2_) - (x2h)
        add_2sum_low: numpy.float32 = ((x2h) - ((_add_2sum_high_2_) - (subtract__add_2sum_high_2__x2h))) + (
            (square_dekker_high) - (subtract__add_2sum_high_2__x2h)
        )
        subtract__add_2sum_high_1___add_2sum_high_2_: numpy.float32 = (_add_2sum_high_1_) - (_add_2sum_high_2_)
        _add_2sum_low_0_: numpy.float32 = (
            (_add_2sum_high_2_) - ((_add_2sum_high_1_) - (subtract__add_2sum_high_1___add_2sum_high_2_))
        ) + ((_square_dekker_high_0_) - (subtract__add_2sum_high_1___add_2sum_high_2_))
        subtract__add_2sum_high_0___add_2sum_high_1_: numpy.float32 = (_add_2sum_high_0_) - (_add_2sum_high_1_)
        _add_2sum_low_1_: numpy.float32 = (
            (_add_2sum_high_1_) - ((_add_2sum_high_0_) - (subtract__add_2sum_high_0___add_2sum_high_1_))
        ) + ((square_dekker_low) - (subtract__add_2sum_high_0___add_2sum_high_1_))
        subtract_add_2sum_high__add_2sum_high_0_: numpy.float32 = (add_2sum_high) - (_add_2sum_high_0_)
        _add_2sum_low_2_: numpy.float32 = (
            (_add_2sum_high_0_) - ((add_2sum_high) - (subtract_add_2sum_high__add_2sum_high_0_))
        ) + ((_square_dekker_low_0_) - (subtract_add_2sum_high__add_2sum_high_0_))
        sum_2sum_high: numpy.float32 = (add_2sum_high) + (
            (((add_2sum_low) + (_add_2sum_low_0_)) + (_add_2sum_low_1_)) + (_add_2sum_low_2_)
        )
        result = make_complex(
            (
                (
                    (numpy.log(mx))
                    + ((half) * (numpy.log1p((one) if (numpy.equal(mn, mx, dtype=numpy.bool_)) else ((r) * (r)))))
                )
                if ((mx) > (numpy.float32(1.8446742e17)))
                else (
                    ((half) * (numpy.log(((xp1) * (xp1)) + (square_dekker_high))))
                    if (((numpy.abs(xp1)) + (ay)) < (numpy.float32(0.2)))
                    else ((half) * (numpy.log1p(sum_2sum_high)))
                )
            ),
            numpy.arctan2(y, xp1),
        )
        return result
