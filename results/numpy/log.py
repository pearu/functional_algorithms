# This file is generated using functional_algorithms tool (0.14.1.dev0+ge22be68.d20241231), see
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


def log_0(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        constant_fneg1: numpy.float64 = numpy.float64(-1.0)
        y: numpy.float64 = (z).imag
        square_dekker_high: numpy.float64 = (y) * (y)
        x: numpy.float64 = (z).real
        _square_dekker_high_0_: numpy.float64 = (x) * (x)
        gt_square_dekker_high__square_dekker_high_0_: numpy.bool_ = (square_dekker_high) > (_square_dekker_high_0_)
        mxh: numpy.float64 = (
            (square_dekker_high) if (gt_square_dekker_high__square_dekker_high_0_) else (_square_dekker_high_0_)
        )
        _add_fast2sum_high_2_: numpy.float64 = (constant_fneg1) + (mxh)
        mnh: numpy.float64 = (
            (_square_dekker_high_0_) if (gt_square_dekker_high__square_dekker_high_0_) else (square_dekker_high)
        )
        _add_fast2sum_high_1_: numpy.float64 = (_add_fast2sum_high_2_) + (mnh)
        veltkamp_splitter_constant: numpy.float64 = numpy.float64(134217729.0)
        multiply_veltkamp_splitter_constant_y: numpy.float64 = (veltkamp_splitter_constant) * (y)
        yh: numpy.float64 = (multiply_veltkamp_splitter_constant_y) + ((y) - (multiply_veltkamp_splitter_constant_y))
        yl: numpy.float64 = (y) - (yh)
        multiply_yh_yl: numpy.float64 = (yh) * (yl)
        square_dekker_low: numpy.float64 = (
            (((-(square_dekker_high)) + ((yh) * (yh))) + (multiply_yh_yl)) + (multiply_yh_yl)
        ) + ((yl) * (yl))
        _add_fast2sum_high_0_: numpy.float64 = (_add_fast2sum_high_1_) + (square_dekker_low)
        multiply_veltkamp_splitter_constant_x: numpy.float64 = (veltkamp_splitter_constant) * (x)
        xh: numpy.float64 = (multiply_veltkamp_splitter_constant_x) + ((x) - (multiply_veltkamp_splitter_constant_x))
        xl: numpy.float64 = (x) - (xh)
        multiply_xh_xl: numpy.float64 = (xh) * (xl)
        _square_dekker_low_0_: numpy.float64 = (
            (((-(_square_dekker_high_0_)) + ((xh) * (xh))) + (multiply_xh_xl)) + (multiply_xh_xl)
        ) + ((xl) * (xl))
        add_fast2sum_high: numpy.float64 = (_add_fast2sum_high_0_) + (_square_dekker_low_0_)
        add_fast2sum_low: numpy.float64 = (mxh) - ((_add_fast2sum_high_2_) - (constant_fneg1))
        _add_fast2sum_low_0_: numpy.float64 = (mnh) - ((_add_fast2sum_high_1_) - (_add_fast2sum_high_2_))
        _add_fast2sum_low_1_: numpy.float64 = (square_dekker_low) - ((_add_fast2sum_high_0_) - (_add_fast2sum_high_1_))
        _add_fast2sum_low_2_: numpy.float64 = (_square_dekker_low_0_) - ((add_fast2sum_high) - (_add_fast2sum_high_0_))
        sum_fast2sum_high: numpy.float64 = (add_fast2sum_high) + (
            (((add_fast2sum_low) + (_add_fast2sum_low_0_)) + (_add_fast2sum_low_1_)) + (_add_fast2sum_low_2_)
        )
        half: numpy.float64 = numpy.float64(0.5)
        abs_x: numpy.float64 = numpy.abs(x)
        abs_y: numpy.float64 = numpy.abs(y)
        mx: numpy.float64 = max(abs_x, abs_y)
        mn: numpy.float64 = min(abs_x, abs_y)
        mn_over_mx: numpy.float64 = (numpy.float64(1.0)) if (numpy.equal(mn, mx)) else ((mn) / (mx))
        result = make_complex(
            (
                ((half) * (numpy.log1p(sum_fast2sum_high)))
                if ((numpy.abs(sum_fast2sum_high)) < (half))
                else ((numpy.log(mx)) + ((half) * (numpy.log1p((mn_over_mx) * (mn_over_mx)))))
            ),
            numpy.arctan2(y, x),
        )
        return result


def log_1(z: numpy.complex64) -> numpy.complex64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        constant_fneg1: numpy.float32 = numpy.float32(-1.0)
        y: numpy.float32 = (z).imag
        square_dekker_high: numpy.float32 = (y) * (y)
        x: numpy.float32 = (z).real
        _square_dekker_high_0_: numpy.float32 = (x) * (x)
        gt_square_dekker_high__square_dekker_high_0_: numpy.bool_ = (square_dekker_high) > (_square_dekker_high_0_)
        mxh: numpy.float32 = (
            (square_dekker_high) if (gt_square_dekker_high__square_dekker_high_0_) else (_square_dekker_high_0_)
        )
        _add_fast2sum_high_2_: numpy.float32 = (constant_fneg1) + (mxh)
        mnh: numpy.float32 = (
            (_square_dekker_high_0_) if (gt_square_dekker_high__square_dekker_high_0_) else (square_dekker_high)
        )
        _add_fast2sum_high_1_: numpy.float32 = (_add_fast2sum_high_2_) + (mnh)
        veltkamp_splitter_constant: numpy.float32 = numpy.float32(4097.0)
        multiply_veltkamp_splitter_constant_y: numpy.float32 = (veltkamp_splitter_constant) * (y)
        yh: numpy.float32 = (multiply_veltkamp_splitter_constant_y) + ((y) - (multiply_veltkamp_splitter_constant_y))
        yl: numpy.float32 = (y) - (yh)
        multiply_yh_yl: numpy.float32 = (yh) * (yl)
        square_dekker_low: numpy.float32 = (
            (((-(square_dekker_high)) + ((yh) * (yh))) + (multiply_yh_yl)) + (multiply_yh_yl)
        ) + ((yl) * (yl))
        _add_fast2sum_high_0_: numpy.float32 = (_add_fast2sum_high_1_) + (square_dekker_low)
        multiply_veltkamp_splitter_constant_x: numpy.float32 = (veltkamp_splitter_constant) * (x)
        xh: numpy.float32 = (multiply_veltkamp_splitter_constant_x) + ((x) - (multiply_veltkamp_splitter_constant_x))
        xl: numpy.float32 = (x) - (xh)
        multiply_xh_xl: numpy.float32 = (xh) * (xl)
        _square_dekker_low_0_: numpy.float32 = (
            (((-(_square_dekker_high_0_)) + ((xh) * (xh))) + (multiply_xh_xl)) + (multiply_xh_xl)
        ) + ((xl) * (xl))
        add_fast2sum_high: numpy.float32 = (_add_fast2sum_high_0_) + (_square_dekker_low_0_)
        add_fast2sum_low: numpy.float32 = (mxh) - ((_add_fast2sum_high_2_) - (constant_fneg1))
        _add_fast2sum_low_0_: numpy.float32 = (mnh) - ((_add_fast2sum_high_1_) - (_add_fast2sum_high_2_))
        _add_fast2sum_low_1_: numpy.float32 = (square_dekker_low) - ((_add_fast2sum_high_0_) - (_add_fast2sum_high_1_))
        _add_fast2sum_low_2_: numpy.float32 = (_square_dekker_low_0_) - ((add_fast2sum_high) - (_add_fast2sum_high_0_))
        sum_fast2sum_high: numpy.float32 = (add_fast2sum_high) + (
            (((add_fast2sum_low) + (_add_fast2sum_low_0_)) + (_add_fast2sum_low_1_)) + (_add_fast2sum_low_2_)
        )
        half: numpy.float32 = numpy.float32(0.5)
        abs_x: numpy.float32 = numpy.abs(x)
        abs_y: numpy.float32 = numpy.abs(y)
        mx: numpy.float32 = max(abs_x, abs_y)
        mn: numpy.float32 = min(abs_x, abs_y)
        mn_over_mx: numpy.float32 = (numpy.float32(1.0)) if (numpy.equal(mn, mx)) else ((mn) / (mx))
        result = make_complex(
            (
                ((half) * (numpy.log1p(sum_fast2sum_high)))
                if ((numpy.abs(sum_fast2sum_high)) < (half))
                else ((numpy.log(mx)) + ((half) * (numpy.log1p((mn_over_mx) * (mn_over_mx)))))
            ),
            numpy.arctan2(y, x),
        )
        return result
