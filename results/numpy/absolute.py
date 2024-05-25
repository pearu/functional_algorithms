# This file is generated using functional_algorithms tool (0.1.2.dev2+g1428951.d20240525), see
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


def absolute_0(z: numpy.complex128) -> numpy.float64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        x: numpy.float64 = (z).real
        abs_x: numpy.float64 = numpy.abs(x)
        abs_y: numpy.float64 = numpy.abs((z).imag)
        mx: numpy.float64 = max(abs_x, abs_y)
        mn: numpy.float64 = min(abs_x, abs_y)
        mn_over_mx: numpy.float64 = (mn) / (mx)
        r: numpy.float64 = (mn_over_mx) * (mn_over_mx)
        sqa: numpy.float64 = numpy.sqrt((numpy.float64(1)) + (r))
        result = (
            ((numpy.sqrt(numpy.float64(2))) * (mx))
            if (numpy.equal(mx, mn, dtype=numpy.bool_))
            else (
                ((mx) + (((mx) * (r)) / (numpy.float64(2))))
                if ((numpy.equal(sqa, numpy.float64(1), dtype=numpy.bool_)) and ((r) > (numpy.float64(0))))
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
        mn_over_mx: numpy.float32 = (mn) / (mx)
        r: numpy.float32 = (mn_over_mx) * (mn_over_mx)
        sqa: numpy.float32 = numpy.sqrt((numpy.float32(1)) + (r))
        result = (
            ((numpy.sqrt(numpy.float32(2))) * (mx))
            if (numpy.equal(mx, mn, dtype=numpy.bool_))
            else (
                ((mx) + (((mx) * (r)) / (numpy.float32(2))))
                if ((numpy.equal(sqa, numpy.float32(1), dtype=numpy.bool_)) and ((r) > (numpy.float32(0))))
                else ((mx) * (sqa))
            )
        )
        return result