# This file is generated using functional_algorithms tool (0.15.1.dev20+g7dbbfd7db.d20261016), see
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


def exp_0(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        y: numpy.float64 = (z).imag
        y_is_not_finite: numpy.bool_ = numpy.logical_not(numpy.isfinite(y))
        x: numpy.float64 = (z).real
        constant_posinf: numpy.float64 = numpy.float64(numpy.inf)
        eq_x_constant_neginf: numpy.bool_ = numpy.equal(x, numpy.float64(-numpy.inf))
        e: numpy.float64 = numpy.exp(x)
        eq_e_constant_posinf: numpy.bool_ = numpy.equal(e, constant_posinf)
        e2: numpy.float64 = numpy.exp((x) * (numpy.float64(0.5)))
        cs: numpy.float64 = numpy.cos(y)
        zero: numpy.float64 = numpy.float64(0.0)
        sn: numpy.float64 = numpy.sin(y)
        result = make_complex(
            numpy.where(
                numpy.logical_and(y_is_not_finite, numpy.logical_or(numpy.equal(x, constant_posinf), eq_x_constant_neginf)),
                e,
                numpy.where(eq_e_constant_posinf, ((e2) * (cs)) * (e2), (e) * (cs)),
            ),
            numpy.where(
                numpy.logical_and(
                    numpy.logical_not(numpy.logical_and(eq_x_constant_neginf, y_is_not_finite)), numpy.not_equal(y, zero)
                ),
                numpy.where(eq_e_constant_posinf, ((e2) * (sn)) * (e2), (e) * (sn)),
                zero,
            ),
        )
        return result


def exp_1(z: numpy.complex64) -> numpy.complex64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        y: numpy.float32 = (z).imag
        y_is_not_finite: numpy.bool_ = numpy.logical_not(numpy.isfinite(y))
        x: numpy.float32 = (z).real
        constant_posinf: numpy.float32 = numpy.float32(numpy.inf)
        eq_x_constant_neginf: numpy.bool_ = numpy.equal(x, numpy.float32(-numpy.inf))
        e: numpy.float32 = numpy.exp(x)
        eq_e_constant_posinf: numpy.bool_ = numpy.equal(e, constant_posinf)
        e2: numpy.float32 = numpy.exp((x) * (numpy.float32(0.5)))
        cs: numpy.float32 = numpy.cos(y)
        zero: numpy.float32 = numpy.float32(0.0)
        sn: numpy.float32 = numpy.sin(y)
        result = make_complex(
            numpy.where(
                numpy.logical_and(y_is_not_finite, numpy.logical_or(numpy.equal(x, constant_posinf), eq_x_constant_neginf)),
                e,
                numpy.where(eq_e_constant_posinf, ((e2) * (cs)) * (e2), (e) * (cs)),
            ),
            numpy.where(
                numpy.logical_and(
                    numpy.logical_not(numpy.logical_and(eq_x_constant_neginf, y_is_not_finite)), numpy.not_equal(y, zero)
                ),
                numpy.where(eq_e_constant_posinf, ((e2) * (sn)) * (e2), (e) * (sn)),
                zero,
            ),
        )
        return result
