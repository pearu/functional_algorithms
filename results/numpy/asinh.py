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


def asinh_0(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        w: numpy.complex128 = numpy.arcsin(make_complex(-((z).imag), (z).real))
        result = make_complex((w).imag, -((w).real))
        return result


def asinh_1(z: numpy.complex64) -> numpy.complex64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        w: numpy.complex64 = numpy.arcsin(make_complex(-((z).imag), (z).real))
        result = make_complex((w).imag, -((w).real))
        return result


def asinh_2(z: numpy.float64) -> numpy.float64:
    with warnings.catch_warnings(action="ignore"):
        signed_y = numpy.float64(z)
        xoy: numpy.float64 = numpy.float64(0)
        y: numpy.float64 = numpy.abs(signed_y)
        safe_max: numpy.float64 = (numpy.sqrt(numpy.float64(numpy.finfo(numpy.float64).max))) / (numpy.float64(8))
        safe_max_opt: numpy.float64 = (safe_max) * (numpy.float64(1e-06))
        y_gt_safe_max_opt: numpy.bool_ = (y) >= (safe_max_opt)
        mx: numpy.float64 = (y) if (y_gt_safe_max_opt) else (xoy)
        two: numpy.float64 = numpy.float64(2)
        lt_y_safe_min: numpy.bool_ = (y) < ((numpy.sqrt(numpy.float64(numpy.finfo(numpy.float64).tiny))) * (numpy.float64(4)))
        xm1: numpy.float64 = numpy.float64(-1)
        half: numpy.float64 = numpy.float64(0.5)
        xp1: numpy.float64 = numpy.float64(1)
        _hypot_2_mx: numpy.float64 = max(xp1, y)
        _hypot_2_mn_over_mx: numpy.float64 = (min(xp1, y)) / (_hypot_2_mx)
        _hypot_2_r: numpy.float64 = (_hypot_2_mn_over_mx) * (_hypot_2_mn_over_mx)
        _hypot_2_sqa: numpy.float64 = numpy.sqrt((xp1) + (_hypot_2_r))
        s: numpy.float64 = (
            ((numpy.sqrt(two)) * (_hypot_2_mx))
            if (numpy.equal(xp1, y, dtype=numpy.bool_))
            else (
                ((_hypot_2_mx) + (((_hypot_2_mx) * (_hypot_2_r)) / (two)))
                if ((numpy.equal(_hypot_2_sqa, xp1, dtype=numpy.bool_)) and ((_hypot_2_r) > (xoy)))
                else ((_hypot_2_mx) * (_hypot_2_sqa))
            )
        )
        apx: numpy.float64 = (half) * ((s) + (s))
        ap1: numpy.float64 = (apx) + (xp1)
        half_yy: numpy.float64 = (half) * ((y) * (y))
        x_ge_1_or_not: numpy.float64 = (
            (((half_yy) / ((s) + (xp1))) + ((half_yy) / ((s) - (xm1)))) if ((apx) <= (numpy.float64(1.5))) else ((apx) - (xp1))
        )
        am1: numpy.float64 = (-((xm1) / (ap1))) if (lt_y_safe_min) else (x_ge_1_or_not)
        sq: numpy.float64 = numpy.sqrt((am1) * (ap1))
        imag: numpy.float64 = (
            ((numpy.log(two)) + (numpy.log(mx)))
            if ((mx) >= ((safe_max_opt) if (y_gt_safe_max_opt) else (safe_max)))
            else (((y) / (sq)) if (lt_y_safe_min) else (numpy.log1p((am1) + (sq))))
        )
        result = (-(imag)) if ((signed_y) < (xoy)) else (imag)
        return result


def asinh_3(z: numpy.float32) -> numpy.float32:
    with warnings.catch_warnings(action="ignore"):
        signed_y = numpy.float32(z)
        xoy: numpy.float32 = numpy.float32(0)
        y: numpy.float32 = numpy.abs(signed_y)
        safe_max: numpy.float32 = (numpy.sqrt(numpy.float32(numpy.finfo(numpy.float32).max))) / (numpy.float32(8))
        safe_max_opt: numpy.float32 = (safe_max) * (numpy.float32(1e-06))
        y_gt_safe_max_opt: numpy.bool_ = (y) >= (safe_max_opt)
        mx: numpy.float32 = (y) if (y_gt_safe_max_opt) else (xoy)
        two: numpy.float32 = numpy.float32(2)
        lt_y_safe_min: numpy.bool_ = (y) < ((numpy.sqrt(numpy.float32(numpy.finfo(numpy.float32).tiny))) * (numpy.float32(4)))
        xm1: numpy.float32 = numpy.float32(-1)
        half: numpy.float32 = numpy.float32(0.5)
        xp1: numpy.float32 = numpy.float32(1)
        _hypot_2_mx: numpy.float32 = max(xp1, y)
        _hypot_2_mn_over_mx: numpy.float32 = (min(xp1, y)) / (_hypot_2_mx)
        _hypot_2_r: numpy.float32 = (_hypot_2_mn_over_mx) * (_hypot_2_mn_over_mx)
        _hypot_2_sqa: numpy.float32 = numpy.sqrt((xp1) + (_hypot_2_r))
        s: numpy.float32 = (
            ((numpy.sqrt(two)) * (_hypot_2_mx))
            if (numpy.equal(xp1, y, dtype=numpy.bool_))
            else (
                ((_hypot_2_mx) + (((_hypot_2_mx) * (_hypot_2_r)) / (two)))
                if ((numpy.equal(_hypot_2_sqa, xp1, dtype=numpy.bool_)) and ((_hypot_2_r) > (xoy)))
                else ((_hypot_2_mx) * (_hypot_2_sqa))
            )
        )
        apx: numpy.float32 = (half) * ((s) + (s))
        ap1: numpy.float32 = (apx) + (xp1)
        half_yy: numpy.float32 = (half) * ((y) * (y))
        x_ge_1_or_not: numpy.float32 = (
            (((half_yy) / ((s) + (xp1))) + ((half_yy) / ((s) - (xm1)))) if ((apx) <= (numpy.float32(1.5))) else ((apx) - (xp1))
        )
        am1: numpy.float32 = (-((xm1) / (ap1))) if (lt_y_safe_min) else (x_ge_1_or_not)
        sq: numpy.float32 = numpy.sqrt((am1) * (ap1))
        imag: numpy.float32 = (
            ((numpy.log(two)) + (numpy.log(mx)))
            if ((mx) >= ((safe_max_opt) if (y_gt_safe_max_opt) else (safe_max)))
            else (((y) / (sq)) if (lt_y_safe_min) else (numpy.log1p((am1) + (sq))))
        )
        result = (-(imag)) if ((signed_y) < (xoy)) else (imag)
        return result
