# This file is generated using functional_algorithms tool (0.15.1.dev3+ge93b47e.d20250113), see
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
        x: numpy.float64 = (z).real
        e: numpy.float64 = numpy.exp(x)
        eq_e_constant_posinf: numpy.bool_ = numpy.equal(e, numpy.float64(numpy.inf))
        e2: numpy.float64 = numpy.exp((x) * (numpy.float64(0.5)))
        y: numpy.float64 = (z).imag
        cs: numpy.float64 = numpy.cos(y)
        zero: numpy.float64 = numpy.float64(0.0)
        sn: numpy.float64 = numpy.sin(y)
        result = make_complex(
            (((e2) * (cs)) * (e2)) if (eq_e_constant_posinf) else ((e) * (cs)),
            (zero) if (numpy.equal(y, zero)) else ((((e2) * (sn)) * (e2)) if (eq_e_constant_posinf) else ((e) * (sn))),
        )
        return result


def exp_1(z: numpy.complex64) -> numpy.complex64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        x: numpy.float32 = (z).real
        e: numpy.float32 = numpy.exp(x)
        eq_e_constant_posinf: numpy.bool_ = numpy.equal(e, numpy.float32(numpy.inf))
        e2: numpy.float32 = numpy.exp((x) * (numpy.float32(0.5)))
        y: numpy.float32 = (z).imag
        cs: numpy.float32 = numpy.cos(y)
        zero: numpy.float32 = numpy.float32(0.0)
        sn: numpy.float32 = numpy.sin(y)
        result = make_complex(
            (((e2) * (cs)) * (e2)) if (eq_e_constant_posinf) else ((e) * (cs)),
            (zero) if (numpy.equal(y, zero)) else ((((e2) * (sn)) * (e2)) if (eq_e_constant_posinf) else ((e) * (sn))),
        )
        return result
