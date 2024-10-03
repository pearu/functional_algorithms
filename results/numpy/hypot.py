# This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
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


def hypot_0(x: numpy.float32, y: numpy.float32) -> numpy.float32:
    with warnings.catch_warnings(action="ignore"):
        x = numpy.float32(x)
        y = numpy.float32(y)
        abs_x: numpy.float32 = numpy.abs(x)
        abs_y: numpy.float32 = numpy.abs(y)
        mx: numpy.float32 = max(abs_x, abs_y)
        mn: numpy.float32 = min(abs_x, abs_y)
        constant_f1: numpy.float32 = numpy.float32(1.0)
        r: numpy.float32 = numpy.square((mn) / (mx))
        sqa: numpy.float32 = numpy.sqrt((constant_f1) + (r))
        result = (
            ((numpy.float32(1.4142135)) * (mx))
            if (numpy.equal(mx, mn, dtype=numpy.bool_))
            else (
                ((mx) + (((mx) * (r)) / (numpy.float32(2.0))))
                if ((numpy.equal(sqa, constant_f1, dtype=numpy.bool_)) and ((r) > (numpy.float32(0.0))))
                else ((mx) * (sqa))
            )
        )
        return result


def hypot_1(x: numpy.float64, y: numpy.float64) -> numpy.float64:
    with warnings.catch_warnings(action="ignore"):
        x = numpy.float64(x)
        y = numpy.float64(y)
        abs_x: numpy.float64 = numpy.abs(x)
        abs_y: numpy.float64 = numpy.abs(y)
        mx: numpy.float64 = max(abs_x, abs_y)
        mn: numpy.float64 = min(abs_x, abs_y)
        constant_f1: numpy.float64 = numpy.float64(1.0)
        r: numpy.float64 = numpy.square((mn) / (mx))
        sqa: numpy.float64 = numpy.sqrt((constant_f1) + (r))
        result = (
            ((numpy.float64(1.4142135623730951)) * (mx))
            if (numpy.equal(mx, mn, dtype=numpy.bool_))
            else (
                ((mx) + (((mx) * (r)) / (numpy.float64(2.0))))
                if ((numpy.equal(sqa, constant_f1, dtype=numpy.bool_)) and ((r) > (numpy.float64(0.0))))
                else ((mx) * (sqa))
            )
        )
        return result
