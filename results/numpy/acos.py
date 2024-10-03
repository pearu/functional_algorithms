# This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
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


def acos_0(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        signed_x: numpy.float64 = (z).real
        x: numpy.float64 = numpy.abs(signed_x)
        signed_y: numpy.float64 = (z).imag
        y: numpy.float64 = numpy.abs(signed_y)
        safe_max: numpy.float64 = (numpy.float64(1.3407807929942596e154)) / (numpy.float64(8.0))
        one: numpy.float64 = numpy.float64(1.0)
        half: numpy.float64 = numpy.float64(0.5)
        xp1: numpy.float64 = (x) + (one)
        r: numpy.float64 = numpy.hypot(xp1, y)
        xm1: numpy.float64 = (x) - (one)
        s: numpy.float64 = numpy.hypot(xm1, y)
        a: numpy.float64 = (half) * ((r) + (s))
        half_apx: numpy.float64 = (half) * ((a) + (x))
        yy: numpy.float64 = (y) * (y)
        rpxp1: numpy.float64 = (r) + (xp1)
        smxm1: numpy.float64 = (s) - (xm1)
        spxm1: numpy.float64 = (s) + (xm1)
        safe_max_opt: numpy.float64 = (
            ((safe_max) * (numpy.float64(1e-06)))
            if ((x) < ((safe_max) * (numpy.float64(1000000000000.0))))
            else ((safe_max) * (numpy.float64(100.0)))
        )
        y_gt_safe_max_opt: numpy.bool_ = (y) >= (safe_max_opt)
        mx: numpy.float64 = (y) if (y_gt_safe_max_opt) else (x)
        xoy: numpy.float64 = (
            ((x) / (y))
            if ((y_gt_safe_max_opt) and (not (numpy.equal(y, numpy.float64(numpy.inf), dtype=numpy.bool_))))
            else (numpy.float64(0.0))
        )
        logical_and_lt_y_safe_min_lt_x_one: numpy.bool_ = ((y) < (numpy.float64(5.966672584960166e-154))) and ((x) < (one))
        ap1: numpy.float64 = (a) + (one)
        half_yy: numpy.float64 = (half) * (yy)
        divide_half_yy_rpxp1: numpy.float64 = (half_yy) / (rpxp1)
        x_ge_1_or_not: numpy.float64 = (
            ((divide_half_yy_rpxp1) + ((half) * (spxm1)))
            if ((x) >= (one))
            else (((divide_half_yy_rpxp1) + ((half_yy) / (smxm1))) if ((a) <= (numpy.float64(1.5))) else ((a) - (one)))
        )
        am1: numpy.float64 = (-(((xp1) * (xm1)) / (ap1))) if (logical_and_lt_y_safe_min_lt_x_one) else (x_ge_1_or_not)
        sq: numpy.float64 = numpy.sqrt((am1) * (ap1))
        imag: numpy.float64 = (
            (((numpy.log(numpy.float64(2.0))) + (numpy.log(mx))) + ((half) * (numpy.log1p((xoy) * (xoy)))))
            if ((mx) >= ((safe_max_opt) if (y_gt_safe_max_opt) else (safe_max)))
            else (((y) / (sq)) if (logical_and_lt_y_safe_min_lt_x_one) else (numpy.log1p((am1) + (sq))))
        )
        result = make_complex(
            numpy.arctan2(
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
            ),
            (imag) if ((signed_y) < (numpy.float64(0.0))) else (-(imag)),
        )
        return result


def acos_1(z: numpy.complex64) -> numpy.complex64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        signed_x: numpy.float32 = (z).real
        x: numpy.float32 = numpy.abs(signed_x)
        signed_y: numpy.float32 = (z).imag
        y: numpy.float32 = numpy.abs(signed_y)
        safe_max: numpy.float32 = (numpy.float32(1.8446743e19)) / (numpy.float32(8.0))
        one: numpy.float32 = numpy.float32(1.0)
        half: numpy.float32 = numpy.float32(0.5)
        xp1: numpy.float32 = (x) + (one)
        r: numpy.float32 = numpy.hypot(xp1, y)
        xm1: numpy.float32 = (x) - (one)
        s: numpy.float32 = numpy.hypot(xm1, y)
        a: numpy.float32 = (half) * ((r) + (s))
        half_apx: numpy.float32 = (half) * ((a) + (x))
        yy: numpy.float32 = (y) * (y)
        rpxp1: numpy.float32 = (r) + (xp1)
        smxm1: numpy.float32 = (s) - (xm1)
        spxm1: numpy.float32 = (s) + (xm1)
        safe_max_opt: numpy.float32 = (
            ((safe_max) * (numpy.float32(1e-06)))
            if ((x) < ((safe_max) * (numpy.float32(1000000000000.0))))
            else ((safe_max) * (numpy.float32(100.0)))
        )
        y_gt_safe_max_opt: numpy.bool_ = (y) >= (safe_max_opt)
        mx: numpy.float32 = (y) if (y_gt_safe_max_opt) else (x)
        xoy: numpy.float32 = (
            ((x) / (y))
            if ((y_gt_safe_max_opt) and (not (numpy.equal(y, numpy.float32(numpy.inf), dtype=numpy.bool_))))
            else (numpy.float32(0.0))
        )
        logical_and_lt_y_safe_min_lt_x_one: numpy.bool_ = ((y) < (numpy.float32(4.3368087e-19))) and ((x) < (one))
        ap1: numpy.float32 = (a) + (one)
        half_yy: numpy.float32 = (half) * (yy)
        divide_half_yy_rpxp1: numpy.float32 = (half_yy) / (rpxp1)
        x_ge_1_or_not: numpy.float32 = (
            ((divide_half_yy_rpxp1) + ((half) * (spxm1)))
            if ((x) >= (one))
            else (((divide_half_yy_rpxp1) + ((half_yy) / (smxm1))) if ((a) <= (numpy.float32(1.5))) else ((a) - (one)))
        )
        am1: numpy.float32 = (-(((xp1) * (xm1)) / (ap1))) if (logical_and_lt_y_safe_min_lt_x_one) else (x_ge_1_or_not)
        sq: numpy.float32 = numpy.sqrt((am1) * (ap1))
        imag: numpy.float32 = (
            (((numpy.log(numpy.float32(2.0))) + (numpy.log(mx))) + ((half) * (numpy.log1p((xoy) * (xoy)))))
            if ((mx) >= ((safe_max_opt) if (y_gt_safe_max_opt) else (safe_max)))
            else (((y) / (sq)) if (logical_and_lt_y_safe_min_lt_x_one) else (numpy.log1p((am1) + (sq))))
        )
        result = make_complex(
            numpy.arctan2(
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
            ),
            (imag) if ((signed_y) < (numpy.float32(0.0))) else (-(imag)),
        )
        return result


def acos_2(z: numpy.float64) -> numpy.float64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.float64(z)
        constant_f1: numpy.float64 = numpy.float64(1.0)
        result = numpy.arctan2(numpy.sqrt(((constant_f1) - (z)) * ((constant_f1) + (z))), z)
        return result


def acos_3(z: numpy.float32) -> numpy.float32:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.float32(z)
        constant_f1: numpy.float32 = numpy.float32(1.0)
        result = numpy.arctan2(numpy.sqrt(((constant_f1) - (z)) * ((constant_f1) + (z))), z)
        return result
