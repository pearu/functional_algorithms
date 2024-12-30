# This file is generated using functional_algorithms tool (0.13.3.dev1+g8d134ad.d20241230), see
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


def log10_0(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        lnz: numpy.complex128 = numpy.log(z)
        x: numpy.float64 = (lnz).real
        ln10: numpy.float64 = numpy.log(numpy.float64(10.0))
        result = make_complex((x) / (ln10), ((lnz).imag) / (ln10))
        return result


def log10_1(z: numpy.complex64) -> numpy.complex64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        lnz: numpy.complex64 = numpy.log(z)
        x: numpy.float32 = (lnz).real
        ln10: numpy.float32 = numpy.log(numpy.float32(10.0))
        result = make_complex((x) / (ln10), ((lnz).imag) / (ln10))
        return result
