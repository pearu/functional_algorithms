# This file is generated using functional_algorithms tool (0.1.dev6+g8b58236), see
#   https://github.com/pearu/functional_algorithms
# for more information.


import numpy
import warnings

finfo_float32 = numpy.finfo(numpy.float32)
finfo_float64 = numpy.finfo(numpy.float64)


def make_complex(r, i):
    if r.dtype == numpy.float32 and i.dtype == numpy.float32:
        return numpy.array([r, i]).view(numpy.complex64)[0]
    elif i.dtype == numpy.float64 and i.dtype == numpy.float64:
        return numpy.array([r, i]).view(numpy.complex128)[0]
    raise NotImplementedError((r.dtype, i.dtype))


def asin_0(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        signed_x: numpy.float64 = (z).real
        x: numpy.float64 = numpy.abs(signed_x)
        signed_y: numpy.float64 = (z).imag
        y: numpy.float64 = numpy.abs(signed_y)
        largest: numpy.float64 = finfo_float64.max
        safe_max: numpy.float64 = (numpy.sqrt(largest)) / (numpy.float64(8))
        xp1: numpy.float64 = (x) + (numpy.float64(1))
        abs_xp1: numpy.float64 = numpy.abs(xp1)
        _hypot_1_mx: numpy.float64 = max(abs_xp1, y)
        mn: numpy.float64 = min(abs_xp1, y)
        sqrt_two: numpy.float64 = numpy.sqrt(numpy.float64(2))
        _square_1_z: numpy.float64 = (mn) / (_hypot_1_mx)
        r: numpy.float64 = (
            ((_hypot_1_mx) * (sqrt_two))
            if (numpy.equal(_hypot_1_mx, mn, dtype=numpy.bool_))
            else ((_hypot_1_mx) * (numpy.sqrt(((_square_1_z) * (_square_1_z)) + (numpy.float64(1)))))
        )
        xm1: numpy.float64 = (x) - (numpy.float64(1))
        abs_xm1: numpy.float64 = numpy.abs(xm1)
        _hypot_2_mx: numpy.float64 = max(abs_xm1, y)
        _hypot_2_mn: numpy.float64 = min(abs_xm1, y)
        _square_2_z: numpy.float64 = (_hypot_2_mn) / (_hypot_2_mx)
        s: numpy.float64 = (
            ((_hypot_2_mx) * (sqrt_two))
            if (numpy.equal(_hypot_2_mx, _hypot_2_mn, dtype=numpy.bool_))
            else ((_hypot_2_mx) * (numpy.sqrt(((_square_2_z) * (_square_2_z)) + (numpy.float64(1)))))
        )
        a: numpy.float64 = (numpy.float64(0.5)) * ((r) + (s))
        half_apx: numpy.float64 = (numpy.float64(0.5)) * ((a) + (x))
        yy: numpy.float64 = (y) * (y)
        rpxp1: numpy.float64 = (r) + (xp1)
        smxm1: numpy.float64 = (s) - (xm1)
        spxm1: numpy.float64 = (s) + (xm1)
        real: numpy.float64 = numpy.arctan2(
            signed_x,
            (
                (y)
                if ((max(x, y)) >= (safe_max))
                else (
                    (numpy.sqrt((half_apx) * (((yy) / (rpxp1)) + (smxm1))))
                    if ((x) <= (numpy.float64(1)))
                    else ((y) * (numpy.sqrt(((half_apx) / (rpxp1)) + ((half_apx) / (spxm1)))))
                )
            ),
        )
        safe_max_opt: numpy.float64 = (
            ((safe_max) * (numpy.float64(1e-06)))
            if ((x) < ((safe_max) * (numpy.float64(1000000000000.0))))
            else ((safe_max) * (numpy.float64(100.0)))
        )
        y_gt_safe_max_opt: numpy.bool_ = (y) >= (safe_max_opt)
        mx: numpy.float64 = (y) if (y_gt_safe_max_opt) else (x)
        posinf: numpy.float64 = numpy.float64(numpy.inf)
        xoy: numpy.float64 = (
            ((x) / (y)) if ((y_gt_safe_max_opt) and (not (numpy.equal(y, posinf, dtype=numpy.bool_)))) else (numpy.float64(0))
        )
        smallest: numpy.float64 = finfo_float64.tiny
        logical_and_lt_y_safe_min_lt_x_one: numpy.bool_ = ((y) < ((numpy.sqrt(smallest)) * (numpy.float64(4)))) and (
            (x) < (numpy.float64(1))
        )
        ap1: numpy.float64 = (a) + (numpy.float64(1))
        half_yy: numpy.float64 = (numpy.float64(0.5)) * (yy)
        divide_half_yy_rpxp1: numpy.float64 = (half_yy) / (rpxp1)
        x_ge_1_or_not: numpy.float64 = (
            ((divide_half_yy_rpxp1) + ((numpy.float64(0.5)) * (spxm1)))
            if ((x) >= (numpy.float64(1)))
            else (
                ((divide_half_yy_rpxp1) + ((half_yy) / (smxm1)))
                if ((a) <= (numpy.float64(1.5)))
                else ((a) - (numpy.float64(1)))
            )
        )
        am1: numpy.float64 = (-(((xp1) * (xm1)) / (ap1))) if (logical_and_lt_y_safe_min_lt_x_one) else (x_ge_1_or_not)
        sq: numpy.float64 = numpy.sqrt((am1) * (ap1))
        imag: numpy.float64 = (
            (((numpy.log(numpy.float64(2))) + (numpy.log(mx))) + ((numpy.float64(0.5)) * (numpy.log1p((xoy) * (xoy)))))
            if ((mx) >= ((safe_max_opt) if (y_gt_safe_max_opt) else (safe_max)))
            else (((y) / (sq)) if (logical_and_lt_y_safe_min_lt_x_one) else (numpy.log1p((am1) + (sq))))
        )
        result = make_complex(real, (-(imag)) if ((signed_y) < (numpy.float64(0))) else (imag))
        return result


def asin_1(z: numpy.complex64) -> numpy.complex64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        signed_x: numpy.float32 = (z).real
        x: numpy.float32 = numpy.abs(signed_x)
        signed_y: numpy.float32 = (z).imag
        y: numpy.float32 = numpy.abs(signed_y)
        largest: numpy.float32 = finfo_float32.max
        safe_max: numpy.float32 = (numpy.sqrt(largest)) / (numpy.float32(8))
        xp1: numpy.float32 = (x) + (numpy.float32(1))
        abs_xp1: numpy.float32 = numpy.abs(xp1)
        _hypot_1_mx: numpy.float32 = max(abs_xp1, y)
        mn: numpy.float32 = min(abs_xp1, y)
        sqrt_two: numpy.float32 = numpy.sqrt(numpy.float32(2))
        _square_1_z: numpy.float32 = (mn) / (_hypot_1_mx)
        r: numpy.float32 = (
            ((_hypot_1_mx) * (sqrt_two))
            if (numpy.equal(_hypot_1_mx, mn, dtype=numpy.bool_))
            else ((_hypot_1_mx) * (numpy.sqrt(((_square_1_z) * (_square_1_z)) + (numpy.float32(1)))))
        )
        xm1: numpy.float32 = (x) - (numpy.float32(1))
        abs_xm1: numpy.float32 = numpy.abs(xm1)
        _hypot_2_mx: numpy.float32 = max(abs_xm1, y)
        _hypot_2_mn: numpy.float32 = min(abs_xm1, y)
        _square_2_z: numpy.float32 = (_hypot_2_mn) / (_hypot_2_mx)
        s: numpy.float32 = (
            ((_hypot_2_mx) * (sqrt_two))
            if (numpy.equal(_hypot_2_mx, _hypot_2_mn, dtype=numpy.bool_))
            else ((_hypot_2_mx) * (numpy.sqrt(((_square_2_z) * (_square_2_z)) + (numpy.float32(1)))))
        )
        a: numpy.float32 = (numpy.float32(0.5)) * ((r) + (s))
        half_apx: numpy.float32 = (numpy.float32(0.5)) * ((a) + (x))
        yy: numpy.float32 = (y) * (y)
        rpxp1: numpy.float32 = (r) + (xp1)
        smxm1: numpy.float32 = (s) - (xm1)
        spxm1: numpy.float32 = (s) + (xm1)
        real: numpy.float32 = numpy.arctan2(
            signed_x,
            (
                (y)
                if ((max(x, y)) >= (safe_max))
                else (
                    (numpy.sqrt((half_apx) * (((yy) / (rpxp1)) + (smxm1))))
                    if ((x) <= (numpy.float32(1)))
                    else ((y) * (numpy.sqrt(((half_apx) / (rpxp1)) + ((half_apx) / (spxm1)))))
                )
            ),
        )
        safe_max_opt: numpy.float32 = (
            ((safe_max) * (numpy.float32(1e-06)))
            if ((x) < ((safe_max) * (numpy.float32(1000000000000.0))))
            else ((safe_max) * (numpy.float32(100.0)))
        )
        y_gt_safe_max_opt: numpy.bool_ = (y) >= (safe_max_opt)
        mx: numpy.float32 = (y) if (y_gt_safe_max_opt) else (x)
        posinf: numpy.float32 = numpy.float32(numpy.inf)
        xoy: numpy.float32 = (
            ((x) / (y)) if ((y_gt_safe_max_opt) and (not (numpy.equal(y, posinf, dtype=numpy.bool_)))) else (numpy.float32(0))
        )
        smallest: numpy.float32 = finfo_float32.tiny
        logical_and_lt_y_safe_min_lt_x_one: numpy.bool_ = ((y) < ((numpy.sqrt(smallest)) * (numpy.float32(4)))) and (
            (x) < (numpy.float32(1))
        )
        ap1: numpy.float32 = (a) + (numpy.float32(1))
        half_yy: numpy.float32 = (numpy.float32(0.5)) * (yy)
        divide_half_yy_rpxp1: numpy.float32 = (half_yy) / (rpxp1)
        x_ge_1_or_not: numpy.float32 = (
            ((divide_half_yy_rpxp1) + ((numpy.float32(0.5)) * (spxm1)))
            if ((x) >= (numpy.float32(1)))
            else (
                ((divide_half_yy_rpxp1) + ((half_yy) / (smxm1)))
                if ((a) <= (numpy.float32(1.5)))
                else ((a) - (numpy.float32(1)))
            )
        )
        am1: numpy.float32 = (-(((xp1) * (xm1)) / (ap1))) if (logical_and_lt_y_safe_min_lt_x_one) else (x_ge_1_or_not)
        sq: numpy.float32 = numpy.sqrt((am1) * (ap1))
        imag: numpy.float32 = (
            (((numpy.log(numpy.float32(2))) + (numpy.log(mx))) + ((numpy.float32(0.5)) * (numpy.log1p((xoy) * (xoy)))))
            if ((mx) >= ((safe_max_opt) if (y_gt_safe_max_opt) else (safe_max)))
            else (((y) / (sq)) if (logical_and_lt_y_safe_min_lt_x_one) else (numpy.log1p((am1) + (sq))))
        )
        result = make_complex(real, (-(imag)) if ((signed_y) < (numpy.float32(0))) else (imag))
        return result


def asin_2(z: numpy.float64) -> numpy.float64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.float64(z)
        result = (numpy.float64(2)) * (numpy.arctan2(z, (numpy.float64(1)) + (numpy.sqrt((numpy.float64(1)) - ((z) * (z))))))
        return result


def asin_3(z: numpy.float32) -> numpy.float32:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.float32(z)
        result = (numpy.float32(2)) * (numpy.arctan2(z, (numpy.float32(1)) + (numpy.sqrt((numpy.float32(1)) - ((z) * (z))))))
        return result
