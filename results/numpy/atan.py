# This file is generated using functional_algorithms tool (0.11.0), see
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


def atan_0(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        w: numpy.complex128 = numpy.arctanh(make_complex(-((z).imag), (z).real))
        result = make_complex((w).imag, -((w).real))
        return result


def atan_1(z: numpy.complex64) -> numpy.complex64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        w: numpy.complex64 = numpy.arctanh(make_complex(-((z).imag), (z).real))
        result = make_complex((w).imag, -((w).real))
        return result
