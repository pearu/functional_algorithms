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


def square_0(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        x: numpy.float64 = (z).real
        y: numpy.float64 = (z).imag
        result = make_complex(
            (
                (numpy.float64(0))
                if (numpy.equal(numpy.abs(x), numpy.abs(y), dtype=numpy.bool_))
                else (((x) - (y)) * ((x) + (y)))
            ),
            (numpy.float64(2)) * ((x) * (y)),
        )
        return result


def square_1(z: numpy.complex64) -> numpy.complex64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        x: numpy.float32 = (z).real
        y: numpy.float32 = (z).imag
        result = make_complex(
            (
                (numpy.float32(0))
                if (numpy.equal(numpy.abs(x), numpy.abs(y), dtype=numpy.bool_))
                else (((x) - (y)) * ((x) + (y)))
            ),
            (numpy.float32(2)) * ((x) * (y)),
        )
        return result


def square_2(z: numpy.float64) -> numpy.float64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.float64(z)
        result = (z) * (z)
        return result


def square_3(z: numpy.float32) -> numpy.float32:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.float32(z)
        result = (z) * (z)
        return result
