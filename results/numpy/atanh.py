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


def atanh_0(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        x: numpy.float64 = (z).real
        zero: numpy.float64 = numpy.float64(0.0)
        one: numpy.float64 = numpy.float64(1.0)
        constant_fneg1: numpy.float64 = numpy.float64(-1.0)
        ax: numpy.float64 = numpy.abs(x)
        safe_max: numpy.float64 = numpy.float64(8.112963841460672e31)
        y: numpy.float64 = (z).imag
        ay: numpy.float64 = numpy.abs(y)
        in_safe_region: numpy.bool_ = ((ax) < (safe_max)) and ((ay) < (safe_max))
        naxm1: numpy.float64 = (one) - (ax)
        y2: numpy.float64 = (y) * (y)
        constant_posinf: numpy.float64 = numpy.float64(numpy.inf)
        constant_neginf: numpy.float64 = numpy.float64(-numpy.inf)
        result = make_complex(
            (
                ((one) if ((x) >= (zero)) else (constant_fneg1))
                * (
                    numpy.log1p(
                        (numpy.float64(4.0))
                        * (
                            ((ax) / (((naxm1) * (naxm1)) + (y2)))
                            if (in_safe_region)
                            else (
                                ((one) / (ax))
                                if (((ay) * (numpy.float64(9007199254740994.0))) < (ax))
                                else (
                                    (zero)
                                    if (
                                        (
                                            (numpy.equal(x, constant_posinf, dtype=numpy.bool_))
                                            or (numpy.equal(x, constant_neginf, dtype=numpy.bool_))
                                        )
                                        or (
                                            (numpy.equal(y, constant_posinf, dtype=numpy.bool_))
                                            or (numpy.equal(y, constant_neginf, dtype=numpy.bool_))
                                        )
                                    )
                                    else (((one) / (((ax) / (y)) + ((y) / (ax)))) / (y))
                                )
                            )
                        )
                    )
                )
            )
            * (numpy.float64(0.25)),
            (
                (numpy.arctan2((y) + (y), ((naxm1) * ((one) + (ax))) - (y2)))
                if (in_safe_region)
                else (((one) if ((y) >= (numpy.float64(0.0))) else (constant_fneg1)) * (numpy.float64(3.141592653589793)))
            )
            * (numpy.float64(0.5)),
        )
        return result


def atanh_1(z: numpy.complex64) -> numpy.complex64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        x: numpy.float32 = (z).real
        zero: numpy.float32 = numpy.float32(0.0)
        one: numpy.float32 = numpy.float32(1.0)
        constant_fneg1: numpy.float32 = numpy.float32(-1.0)
        ax: numpy.float32 = numpy.abs(x)
        safe_max: numpy.float32 = numpy.float32(281475040000000.0)
        y: numpy.float32 = (z).imag
        ay: numpy.float32 = numpy.abs(y)
        in_safe_region: numpy.bool_ = ((ax) < (safe_max)) and ((ay) < (safe_max))
        naxm1: numpy.float32 = (one) - (ax)
        y2: numpy.float32 = (y) * (y)
        constant_posinf: numpy.float32 = numpy.float32(numpy.inf)
        constant_neginf: numpy.float32 = numpy.float32(-numpy.inf)
        result = make_complex(
            (
                ((one) if ((x) >= (zero)) else (constant_fneg1))
                * (
                    numpy.log1p(
                        (numpy.float32(4.0))
                        * (
                            ((ax) / (((naxm1) * (naxm1)) + (y2)))
                            if (in_safe_region)
                            else (
                                ((one) / (ax))
                                if (((ay) * (numpy.float32(16777218.0))) < (ax))
                                else (
                                    (zero)
                                    if (
                                        (
                                            (numpy.equal(x, constant_posinf, dtype=numpy.bool_))
                                            or (numpy.equal(x, constant_neginf, dtype=numpy.bool_))
                                        )
                                        or (
                                            (numpy.equal(y, constant_posinf, dtype=numpy.bool_))
                                            or (numpy.equal(y, constant_neginf, dtype=numpy.bool_))
                                        )
                                    )
                                    else (((one) / (((ax) / (y)) + ((y) / (ax)))) / (y))
                                )
                            )
                        )
                    )
                )
            )
            * (numpy.float32(0.25)),
            (
                (numpy.arctan2((y) + (y), ((naxm1) * ((one) + (ax))) - (y2)))
                if (in_safe_region)
                else (((one) if ((y) >= (numpy.float32(0.0))) else (constant_fneg1)) * (numpy.float32(3.1415927)))
            )
            * (numpy.float32(0.5)),
        )
        return result
