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
        z = numpy.float64(z)
        ax: numpy.float64 = numpy.abs(z)
        ax2: numpy.float64 = (ax) * (ax)
        one: numpy.float64 = numpy.float64(1)
        result = (numpy.sign(z)) * (
            ((numpy.log(numpy.float64(2))) + (numpy.log(ax)))
            if ((ax) >= (numpy.sqrt(numpy.float64(numpy.finfo(numpy.float64).max))))
            else (numpy.log1p((ax) + ((ax2) / ((one) + (numpy.sqrt((one) + (ax2)))))))
        )
        return result


def asinh_3(z: numpy.float32) -> numpy.float32:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.float32(z)
        ax: numpy.float32 = numpy.abs(z)
        ax2: numpy.float32 = (ax) * (ax)
        one: numpy.float32 = numpy.float32(1)
        result = (numpy.sign(z)) * (
            ((numpy.log(numpy.float32(2))) + (numpy.log(ax)))
            if ((ax) >= (numpy.sqrt(numpy.float32(numpy.finfo(numpy.float32).max))))
            else (numpy.log1p((ax) + ((ax2) / ((one) + (numpy.sqrt((one) + (ax2)))))))
        )
        return result
