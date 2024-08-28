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


def atanh_0(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        x: numpy.float64 = (z).real
        zero: numpy.float64 = numpy.float64(0)
        one: numpy.float64 = numpy.float64(1)
        constant_neg1: numpy.float64 = numpy.float64(-1)
        ax: numpy.float64 = numpy.abs(x)
        largest: numpy.float64 = numpy.float64(numpy.finfo(numpy.float64).max)
        inv_negeps: numpy.float64 = (
            (numpy.float64(9007199254740994.0))
            if ((largest) > (numpy.float64(1e308)))
            else ((numpy.float64(16777218.0)) if ((largest) > (numpy.float64(1e38))) else (numpy.float64(2050.0)))
        )
        safe_max: numpy.float64 = (inv_negeps) * (inv_negeps)
        y: numpy.float64 = (z).imag
        ay: numpy.float64 = numpy.abs(y)
        in_safe_region: numpy.bool_ = ((ax) < (safe_max)) and ((ay) < (safe_max))
        naxm1: numpy.float64 = (one) - (ax)
        y2: numpy.float64 = (y) * (y)
        constant_posinf: numpy.float64 = numpy.float64(numpy.float64(numpy.inf))
        constant_neginf: numpy.float64 = numpy.float64(-numpy.float64(numpy.inf))
        result = make_complex(
            (
                ((one) if ((x) >= (zero)) else (constant_neg1))
                * (
                    numpy.log1p(
                        (numpy.float64(4))
                        * (
                            ((ax) / (((naxm1) * (naxm1)) + (y2)))
                            if (in_safe_region)
                            else (
                                ((one) / (ax))
                                if (((ay) * (inv_negeps)) < (ax))
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
                else (((one) if ((y) >= (numpy.float64(0))) else (constant_neg1)) * (numpy.float64(numpy.float64(numpy.pi))))
            )
            * (numpy.float64(0.5)),
        )
        return result


def atanh_1(z: numpy.complex64) -> numpy.complex64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex64(z)
        x: numpy.float32 = (z).real
        zero: numpy.float32 = numpy.float32(0)
        one: numpy.float32 = numpy.float32(1)
        constant_neg1: numpy.float32 = numpy.float32(-1)
        ax: numpy.float32 = numpy.abs(x)
        largest: numpy.float32 = numpy.float32(numpy.finfo(numpy.float32).max)
        inv_negeps: numpy.float32 = (
            (numpy.float32(9007199254740994.0))
            if ((largest) > (numpy.float32(1e308)))
            else ((numpy.float32(16777218.0)) if ((largest) > (numpy.float32(1e38))) else (numpy.float32(2050.0)))
        )
        safe_max: numpy.float32 = (inv_negeps) * (inv_negeps)
        y: numpy.float32 = (z).imag
        ay: numpy.float32 = numpy.abs(y)
        in_safe_region: numpy.bool_ = ((ax) < (safe_max)) and ((ay) < (safe_max))
        naxm1: numpy.float32 = (one) - (ax)
        y2: numpy.float32 = (y) * (y)
        constant_posinf: numpy.float32 = numpy.float32(numpy.float32(numpy.inf))
        constant_neginf: numpy.float32 = numpy.float32(-numpy.float32(numpy.inf))
        result = make_complex(
            (
                ((one) if ((x) >= (zero)) else (constant_neg1))
                * (
                    numpy.log1p(
                        (numpy.float32(4))
                        * (
                            ((ax) / (((naxm1) * (naxm1)) + (y2)))
                            if (in_safe_region)
                            else (
                                ((one) / (ax))
                                if (((ay) * (inv_negeps)) < (ax))
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
                else (((one) if ((y) >= (numpy.float32(0))) else (constant_neg1)) * (numpy.float32(numpy.float32(numpy.pi))))
            )
            * (numpy.float32(0.5)),
        )
        return result


def atanh_2(z: numpy.float64) -> numpy.float64:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.float64(z)
        result = numpy.arctanh(z)
        return result


def atanh_3(z: numpy.float32) -> numpy.float32:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.float32(z)
        result = numpy.arctanh(z)
        return result
