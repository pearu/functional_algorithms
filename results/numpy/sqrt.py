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


def sqrt_0(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        x: numpy.float64 = (z).real
        constant_f0: numpy.float64 = numpy.float64(0.0)
        ax: numpy.float64 = numpy.abs(x)
        y: numpy.float64 = (z).imag
        ay: numpy.float64 = numpy.abs(y)
        eq_ax_ay: numpy.bool_ = numpy.equal(ax, ay, dtype=numpy.bool_)
        sq_ax: numpy.float64 = numpy.sqrt(ax)
        sq_2: numpy.float64 = numpy.float64(1.4142135623730951)
        two: numpy.float64 = numpy.float64(2.0)
        u_general: numpy.float64 = numpy.sqrt(((numpy.hypot(ax, ay)) / (two)) + ((ax) / (two)))
        logical_or_eq_u_general_constant_f0_eq_u_general_constant_posinf: numpy.bool_ = (
            numpy.equal(u_general, constant_f0, dtype=numpy.bool_)
        ) or (numpy.equal(u_general, numpy.float64(numpy.inf), dtype=numpy.bool_))
        gt_ax_ay: numpy.bool_ = (ax) > (ay)
        one: numpy.float64 = numpy.float64(1.0)
        lt_ax_ay: numpy.bool_ = (ax) < (ay)
        r: numpy.float64 = (one) if (eq_ax_ay) else (((ax) / (ay)) if (lt_ax_ay) else ((ay) / (ax)))
        h: numpy.float64 = numpy.hypot(one, r)
        sq_1h: numpy.float64 = numpy.sqrt((one) + (h))
        sq_ay: numpy.float64 = numpy.sqrt(ay)
        sq_rh: numpy.float64 = numpy.sqrt((r) + (h))
        u: numpy.float64 = (
            (((sq_ax) * (numpy.float64(1.5537739740300374))) / (sq_2))
            if (eq_ax_ay)
            else (
                (((sq_ax) * ((sq_1h) / (sq_2))) if (gt_ax_ay) else ((sq_ay) * ((sq_rh) / (sq_2))))
                if (logical_or_eq_u_general_constant_f0_eq_u_general_constant_posinf)
                else (u_general)
            )
        )
        ay_div_u: numpy.float64 = (
            ((sq_ay) / (numpy.float64(2.19736822693562)))
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
        lt_y_constant_f0: numpy.bool_ = (y) < (constant_f0)
        result = make_complex(
            (u) if ((x) >= (constant_f0)) else (ay_div_u),
            (
                ((-(u)) if (lt_y_constant_f0) else (u))
                if ((x) < (constant_f0))
                else ((-(ay_div_u)) if (lt_y_constant_f0) else (ay_div_u))
            ),
        )
        return result


def sqrt_1(z: numpy.complex64) -> numpy.complex64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        x: numpy.float32 = (z).real
        constant_f0: numpy.float32 = numpy.float32(0.0)
        ax: numpy.float32 = numpy.abs(x)
        y: numpy.float32 = (z).imag
        ay: numpy.float32 = numpy.abs(y)
        eq_ax_ay: numpy.bool_ = numpy.equal(ax, ay, dtype=numpy.bool_)
        sq_ax: numpy.float32 = numpy.sqrt(ax)
        sq_2: numpy.float32 = numpy.float32(1.4142135)
        two: numpy.float32 = numpy.float32(2.0)
        u_general: numpy.float32 = numpy.sqrt(((numpy.hypot(ax, ay)) / (two)) + ((ax) / (two)))
        logical_or_eq_u_general_constant_f0_eq_u_general_constant_posinf: numpy.bool_ = (
            numpy.equal(u_general, constant_f0, dtype=numpy.bool_)
        ) or (numpy.equal(u_general, numpy.float32(numpy.inf), dtype=numpy.bool_))
        gt_ax_ay: numpy.bool_ = (ax) > (ay)
        one: numpy.float32 = numpy.float32(1.0)
        lt_ax_ay: numpy.bool_ = (ax) < (ay)
        r: numpy.float32 = (one) if (eq_ax_ay) else (((ax) / (ay)) if (lt_ax_ay) else ((ay) / (ax)))
        h: numpy.float32 = numpy.hypot(one, r)
        sq_1h: numpy.float32 = numpy.sqrt((one) + (h))
        sq_ay: numpy.float32 = numpy.sqrt(ay)
        sq_rh: numpy.float32 = numpy.sqrt((r) + (h))
        u: numpy.float32 = (
            (((sq_ax) * (numpy.float32(1.553774))) / (sq_2))
            if (eq_ax_ay)
            else (
                (((sq_ax) * ((sq_1h) / (sq_2))) if (gt_ax_ay) else ((sq_ay) * ((sq_rh) / (sq_2))))
                if (logical_or_eq_u_general_constant_f0_eq_u_general_constant_posinf)
                else (u_general)
            )
        )
        ay_div_u: numpy.float32 = (
            ((sq_ay) / (numpy.float32(2.1973681)))
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
        lt_y_constant_f0: numpy.bool_ = (y) < (constant_f0)
        result = make_complex(
            (u) if ((x) >= (constant_f0)) else (ay_div_u),
            (
                ((-(u)) if (lt_y_constant_f0) else (u))
                if ((x) < (constant_f0))
                else ((-(ay_div_u)) if (lt_y_constant_f0) else (ay_div_u))
            ),
        )
        return result
