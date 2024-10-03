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


def log1p_0(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        x: numpy.float64 = (z).real
        one: numpy.float64 = numpy.float64(1.0)
        xp1: numpy.float64 = (x) + (one)
        y: numpy.float64 = (z).imag
        ay: numpy.float64 = numpy.abs(y)
        axp1: numpy.float64 = numpy.abs(xp1)
        mx: numpy.float64 = max(axp1, ay)
        mn: numpy.float64 = min(axp1, ay)
        r: numpy.float64 = (mn) / (mx)
        result = make_complex(
            (numpy.log1p((x) if ((xp1) >= (ay)) else ((mx) - (one))))
            + ((numpy.float64(0.5)) * (numpy.log1p((one) if (numpy.equal(mn, mx, dtype=numpy.bool_)) else ((r) * (r))))),
            numpy.arctan2(y, xp1),
        )
        return result


def log1p_1(z: numpy.complex64) -> numpy.complex64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        x: numpy.float32 = (z).real
        one: numpy.float32 = numpy.float32(1.0)
        xp1: numpy.float32 = (x) + (one)
        y: numpy.float32 = (z).imag
        ay: numpy.float32 = numpy.abs(y)
        axp1: numpy.float32 = numpy.abs(xp1)
        mx: numpy.float32 = max(axp1, ay)
        mn: numpy.float32 = min(axp1, ay)
        r: numpy.float32 = (mn) / (mx)
        result = make_complex(
            (numpy.log1p((x) if ((xp1) >= (ay)) else ((mx) - (one))))
            + ((numpy.float32(0.5)) * (numpy.log1p((one) if (numpy.equal(mn, mx, dtype=numpy.bool_)) else ((r) * (r))))),
            numpy.arctan2(y, xp1),
        )
        return result
