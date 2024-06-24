# This file is generated using functional_algorithms tool (0.4.0), see
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


def acosh_0(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        signed_y: numpy.float64 = (z).imag
        zero: numpy.float64 = numpy.float64(0)
        y: numpy.float64 = numpy.abs(signed_y)
        signed_x: numpy.float64 = (z).real
        x: numpy.float64 = numpy.abs(signed_x)
        safe_max: numpy.float64 = (numpy.sqrt(numpy.float64(numpy.finfo(numpy.float64).max))) / (numpy.float64(8))
        safe_max_opt: numpy.float64 = (
            ((safe_max) * (numpy.float64(1e-06)))
            if ((x) < ((safe_max) * (numpy.float64(1000000000000.0))))
            else ((safe_max) * (numpy.float64(100.0)))
        )
        y_gt_safe_max_opt: numpy.bool_ = (y) >= (safe_max_opt)
        mx: numpy.float64 = (y) if (y_gt_safe_max_opt) else (x)
        two: numpy.float64 = numpy.float64(2)
        half: numpy.float64 = numpy.float64(0.5)
        xoy: numpy.float64 = (
            ((x) / (y))
            if ((y_gt_safe_max_opt) and (not (numpy.equal(y, numpy.float64(numpy.float64(numpy.inf)), dtype=numpy.bool_))))
            else (zero)
        )
        one: numpy.float64 = numpy.float64(1)
        logical_and_lt_y_safe_min_lt_x_one: numpy.bool_ = (
            (y) < ((numpy.sqrt(numpy.float64(numpy.finfo(numpy.float64).tiny))) * (numpy.float64(4)))
        ) and ((x) < (one))
        xp1: numpy.float64 = (x) + (one)
        xm1: numpy.float64 = (x) - (one)
        abs_xp1: numpy.float64 = numpy.abs(xp1)
        _hypot_1_mx: numpy.float64 = max(abs_xp1, y)
        mn: numpy.float64 = min(abs_xp1, y)
        sqrt_two: numpy.float64 = numpy.sqrt(two)
        mn_over_mx: numpy.float64 = (mn) / (_hypot_1_mx)
        _hypot_1_r: numpy.float64 = (mn_over_mx) * (mn_over_mx)
        sqa: numpy.float64 = numpy.sqrt((one) + (_hypot_1_r))
        r: numpy.float64 = (
            ((sqrt_two) * (_hypot_1_mx))
            if (numpy.equal(_hypot_1_mx, mn, dtype=numpy.bool_))
            else (
                ((_hypot_1_mx) + (((_hypot_1_mx) * (_hypot_1_r)) / (two)))
                if ((numpy.equal(sqa, one, dtype=numpy.bool_)) and ((_hypot_1_r) > (zero)))
                else ((_hypot_1_mx) * (sqa))
            )
        )
        abs_xm1: numpy.float64 = numpy.abs(xm1)
        _hypot_2_mx: numpy.float64 = max(abs_xm1, y)
        _hypot_2_mn: numpy.float64 = min(abs_xm1, y)
        _hypot_2_mn_over_mx: numpy.float64 = (_hypot_2_mn) / (_hypot_2_mx)
        _hypot_2_r: numpy.float64 = (_hypot_2_mn_over_mx) * (_hypot_2_mn_over_mx)
        _hypot_2_sqa: numpy.float64 = numpy.sqrt((one) + (_hypot_2_r))
        s: numpy.float64 = (
            ((sqrt_two) * (_hypot_2_mx))
            if (numpy.equal(_hypot_2_mx, _hypot_2_mn, dtype=numpy.bool_))
            else (
                ((_hypot_2_mx) + (((_hypot_2_mx) * (_hypot_2_r)) / (two)))
                if ((numpy.equal(_hypot_2_sqa, one, dtype=numpy.bool_)) and ((_hypot_2_r) > (zero)))
                else ((_hypot_2_mx) * (_hypot_2_sqa))
            )
        )
        a: numpy.float64 = (half) * ((r) + (s))
        ap1: numpy.float64 = (a) + (one)
        yy: numpy.float64 = (y) * (y)
        half_yy: numpy.float64 = (half) * (yy)
        rpxp1: numpy.float64 = (r) + (xp1)
        divide_half_yy_rpxp1: numpy.float64 = (half_yy) / (rpxp1)
        spxm1: numpy.float64 = (s) + (xm1)
        smxm1: numpy.float64 = (s) - (xm1)
        x_ge_1_or_not: numpy.float64 = (
            ((divide_half_yy_rpxp1) + ((half) * (spxm1)))
            if ((x) >= (one))
            else (((divide_half_yy_rpxp1) + ((half_yy) / (smxm1))) if ((a) <= (numpy.float64(1.5))) else ((a) - (one)))
        )
        am1: numpy.float64 = (-(((xp1) * (xm1)) / (ap1))) if (logical_and_lt_y_safe_min_lt_x_one) else (x_ge_1_or_not)
        sq: numpy.float64 = numpy.sqrt((am1) * (ap1))
        imag: numpy.float64 = (
            (((numpy.log(two)) + (numpy.log(mx))) + ((half) * (numpy.log1p((xoy) * (xoy)))))
            if ((mx) >= ((safe_max_opt) if (y_gt_safe_max_opt) else (safe_max)))
            else (((y) / (sq)) if (logical_and_lt_y_safe_min_lt_x_one) else (numpy.log1p((am1) + (sq))))
        )
        half_apx: numpy.float64 = (half) * ((a) + (x))
        acos_real: numpy.float64 = numpy.arctan2(
            (
                (y)
                if ((max(x, y)) >= (safe_max))
                else (
                    (numpy.sqrt((half_apx) * (((yy) / (rpxp1)) + (smxm1))))
                    if ((x) <= (one))
                    else ((y) * (numpy.sqrt(((half_apx) / (rpxp1)) + ((half_apx) / (spxm1)))))
                )
            ),
            signed_x,
        )
        complex_negative_acos_signed_imag_acos_real: numpy.complex128 = make_complex(
            -((imag) if ((signed_y) < (zero)) else (-(imag))), acos_real
        )
        result = (
            (-(complex_negative_acos_signed_imag_acos_real))
            if ((signed_y) < (numpy.float64(0)))
            else (complex_negative_acos_signed_imag_acos_real)
        )
        return result


def acosh_1(z: numpy.complex64) -> numpy.complex64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        signed_y: numpy.float32 = (z).imag
        zero: numpy.float32 = numpy.float32(0)
        y: numpy.float32 = numpy.abs(signed_y)
        signed_x: numpy.float32 = (z).real
        x: numpy.float32 = numpy.abs(signed_x)
        safe_max: numpy.float32 = (numpy.sqrt(numpy.float32(numpy.finfo(numpy.float32).max))) / (numpy.float32(8))
        safe_max_opt: numpy.float32 = (
            ((safe_max) * (numpy.float32(1e-06)))
            if ((x) < ((safe_max) * (numpy.float32(1000000000000.0))))
            else ((safe_max) * (numpy.float32(100.0)))
        )
        y_gt_safe_max_opt: numpy.bool_ = (y) >= (safe_max_opt)
        mx: numpy.float32 = (y) if (y_gt_safe_max_opt) else (x)
        two: numpy.float32 = numpy.float32(2)
        half: numpy.float32 = numpy.float32(0.5)
        xoy: numpy.float32 = (
            ((x) / (y))
            if ((y_gt_safe_max_opt) and (not (numpy.equal(y, numpy.float32(numpy.float32(numpy.inf)), dtype=numpy.bool_))))
            else (zero)
        )
        one: numpy.float32 = numpy.float32(1)
        logical_and_lt_y_safe_min_lt_x_one: numpy.bool_ = (
            (y) < ((numpy.sqrt(numpy.float32(numpy.finfo(numpy.float32).tiny))) * (numpy.float32(4)))
        ) and ((x) < (one))
        xp1: numpy.float32 = (x) + (one)
        xm1: numpy.float32 = (x) - (one)
        abs_xp1: numpy.float32 = numpy.abs(xp1)
        _hypot_1_mx: numpy.float32 = max(abs_xp1, y)
        mn: numpy.float32 = min(abs_xp1, y)
        sqrt_two: numpy.float32 = numpy.sqrt(two)
        mn_over_mx: numpy.float32 = (mn) / (_hypot_1_mx)
        _hypot_1_r: numpy.float32 = (mn_over_mx) * (mn_over_mx)
        sqa: numpy.float32 = numpy.sqrt((one) + (_hypot_1_r))
        r: numpy.float32 = (
            ((sqrt_two) * (_hypot_1_mx))
            if (numpy.equal(_hypot_1_mx, mn, dtype=numpy.bool_))
            else (
                ((_hypot_1_mx) + (((_hypot_1_mx) * (_hypot_1_r)) / (two)))
                if ((numpy.equal(sqa, one, dtype=numpy.bool_)) and ((_hypot_1_r) > (zero)))
                else ((_hypot_1_mx) * (sqa))
            )
        )
        abs_xm1: numpy.float32 = numpy.abs(xm1)
        _hypot_2_mx: numpy.float32 = max(abs_xm1, y)
        _hypot_2_mn: numpy.float32 = min(abs_xm1, y)
        _hypot_2_mn_over_mx: numpy.float32 = (_hypot_2_mn) / (_hypot_2_mx)
        _hypot_2_r: numpy.float32 = (_hypot_2_mn_over_mx) * (_hypot_2_mn_over_mx)
        _hypot_2_sqa: numpy.float32 = numpy.sqrt((one) + (_hypot_2_r))
        s: numpy.float32 = (
            ((sqrt_two) * (_hypot_2_mx))
            if (numpy.equal(_hypot_2_mx, _hypot_2_mn, dtype=numpy.bool_))
            else (
                ((_hypot_2_mx) + (((_hypot_2_mx) * (_hypot_2_r)) / (two)))
                if ((numpy.equal(_hypot_2_sqa, one, dtype=numpy.bool_)) and ((_hypot_2_r) > (zero)))
                else ((_hypot_2_mx) * (_hypot_2_sqa))
            )
        )
        a: numpy.float32 = (half) * ((r) + (s))
        ap1: numpy.float32 = (a) + (one)
        yy: numpy.float32 = (y) * (y)
        half_yy: numpy.float32 = (half) * (yy)
        rpxp1: numpy.float32 = (r) + (xp1)
        divide_half_yy_rpxp1: numpy.float32 = (half_yy) / (rpxp1)
        spxm1: numpy.float32 = (s) + (xm1)
        smxm1: numpy.float32 = (s) - (xm1)
        x_ge_1_or_not: numpy.float32 = (
            ((divide_half_yy_rpxp1) + ((half) * (spxm1)))
            if ((x) >= (one))
            else (((divide_half_yy_rpxp1) + ((half_yy) / (smxm1))) if ((a) <= (numpy.float32(1.5))) else ((a) - (one)))
        )
        am1: numpy.float32 = (-(((xp1) * (xm1)) / (ap1))) if (logical_and_lt_y_safe_min_lt_x_one) else (x_ge_1_or_not)
        sq: numpy.float32 = numpy.sqrt((am1) * (ap1))
        imag: numpy.float32 = (
            (((numpy.log(two)) + (numpy.log(mx))) + ((half) * (numpy.log1p((xoy) * (xoy)))))
            if ((mx) >= ((safe_max_opt) if (y_gt_safe_max_opt) else (safe_max)))
            else (((y) / (sq)) if (logical_and_lt_y_safe_min_lt_x_one) else (numpy.log1p((am1) + (sq))))
        )
        half_apx: numpy.float32 = (half) * ((a) + (x))
        acos_real: numpy.float32 = numpy.arctan2(
            (
                (y)
                if ((max(x, y)) >= (safe_max))
                else (
                    (numpy.sqrt((half_apx) * (((yy) / (rpxp1)) + (smxm1))))
                    if ((x) <= (one))
                    else ((y) * (numpy.sqrt(((half_apx) / (rpxp1)) + ((half_apx) / (spxm1)))))
                )
            ),
            signed_x,
        )
        complex_negative_acos_signed_imag_acos_real: numpy.complex64 = make_complex(
            -((imag) if ((signed_y) < (zero)) else (-(imag))), acos_real
        )
        result = (
            (-(complex_negative_acos_signed_imag_acos_real))
            if ((signed_y) < (numpy.float32(0)))
            else (complex_negative_acos_signed_imag_acos_real)
        )
        return result


def acosh_2(z: numpy.float64) -> numpy.float64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.float64(z)
        constant_2: numpy.float64 = numpy.float64(2)
        constant_1: numpy.float64 = numpy.float64(1)
        sqrt_subtract_z_constant_1: numpy.float64 = numpy.sqrt((z) - (constant_1))
        result = (
            ((numpy.log(constant_2)) + (numpy.log(z)))
            if ((z) >= ((numpy.float64(numpy.finfo(numpy.float64).max)) / (constant_2)))
            else (
                numpy.log1p((sqrt_subtract_z_constant_1) * ((numpy.sqrt((z) + (constant_1))) + (sqrt_subtract_z_constant_1)))
            )
        )
        return result


def acosh_3(z: numpy.float32) -> numpy.float32:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.float32(z)
        constant_2: numpy.float32 = numpy.float32(2)
        constant_1: numpy.float32 = numpy.float32(1)
        sqrt_subtract_z_constant_1: numpy.float32 = numpy.sqrt((z) - (constant_1))
        result = (
            ((numpy.log(constant_2)) + (numpy.log(z)))
            if ((z) >= ((numpy.float32(numpy.finfo(numpy.float32).max)) / (constant_2)))
            else (
                numpy.log1p((sqrt_subtract_z_constant_1) * ((numpy.sqrt((z) + (constant_1))) + (sqrt_subtract_z_constant_1)))
            )
        )
        return result
