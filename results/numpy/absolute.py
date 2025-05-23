# This file is generated using functional_algorithms tool (0.14.1.dev0+ge22be68.d20241231), see
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


def absolute_0(z: numpy.complex128) -> numpy.float64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        x: numpy.float64 = (z).real
        abs_x: numpy.float64 = numpy.abs(x)
        abs_y: numpy.float64 = numpy.abs((z).imag)
        mx: numpy.float64 = max(abs_x, abs_y)
        mn: numpy.float64 = min(abs_x, abs_y)
        constant_f1: numpy.float64 = numpy.float64(1.0)
        r: numpy.float64 = numpy.square((mn) / (mx))
        sqa: numpy.float64 = numpy.sqrt((constant_f1) + (r))
        result = (
            ((numpy.float64(1.4142135623730951)) * (mx))
            if (numpy.equal(mx, mn))
            else (
                ((mx) + (((mx) * (r)) / (numpy.float64(2.0))))
                if ((numpy.equal(sqa, constant_f1)) and ((r) > (numpy.float64(0.0))))
                else ((mx) * (sqa))
            )
        )
        return result


def absolute_1(z: numpy.complex64) -> numpy.float32:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        x: numpy.float32 = (z).real
        abs_x: numpy.float32 = numpy.abs(x)
        abs_y: numpy.float32 = numpy.abs((z).imag)
        mx: numpy.float32 = max(abs_x, abs_y)
        mn: numpy.float32 = min(abs_x, abs_y)
        constant_f1: numpy.float32 = numpy.float32(1.0)
        r: numpy.float32 = numpy.square((mn) / (mx))
        sqa: numpy.float32 = numpy.sqrt((constant_f1) + (r))
        result = (
            ((numpy.float32(1.4142135)) * (mx))
            if (numpy.equal(mx, mn))
            else (
                ((mx) + (((mx) * (r)) / (numpy.float32(2.0))))
                if ((numpy.equal(sqa, constant_f1)) and ((r) > (numpy.float32(0.0))))
                else ((mx) * (sqa))
            )
        )
        return result
