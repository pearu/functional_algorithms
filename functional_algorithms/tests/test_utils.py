import numpy
import pytest

from functional_algorithms import utils


@pytest.fixture(scope="function", params=[numpy.float32, numpy.float64])
def dtype(request):
    return request.param


def _check_real_samples(
    r, include_infinity=None, include_zero=None, include_subnormal=None, include_nan=None, nonnegative=None, include_huge=None
):
    fi = numpy.finfo(r.dtype)
    if nonnegative:
        if include_zero:
            assert r[0] == 0
            if include_subnormal:
                assert r[1] == fi.smallest_subnormal
            else:
                assert r[1] == fi.smallest_normal
        else:
            assert r[0] != 0
            if include_subnormal:
                assert r[0] == fi.smallest_subnormal
            else:
                assert r[0] == fi.smallest_normal
        if include_nan:
            assert numpy.isnan(r[-1])
            if include_infinity:
                if include_huge and r.size > 9:
                    assert numpy.nextafter(r[-4], numpy.inf, dtype=r.dtype) == fi.max
                assert r[-3] == fi.max
                assert numpy.isposinf(r[-2])
            else:
                assert r[-2] == fi.max
                if include_huge and r.size > 9:
                    assert numpy.nextafter(r[-3], numpy.inf, dtype=r.dtype) == fi.max
            for i in range(r.size - 2):
                assert r[i] < r[i + 1]
        else:
            if include_infinity:
                if include_huge and r.size > 8:
                    assert numpy.nextafter(r[-3], numpy.inf, dtype=r.dtype) == fi.max
                assert r[-2] == fi.max
                assert numpy.isposinf(r[-1])
            else:
                assert r[-1] == fi.max
                if include_huge and r.size > 8:
                    assert numpy.nextafter(r[-2], numpy.inf, dtype=r.dtype) == fi.max
            for i in range(r.size - 1):
                assert r[i] < r[i + 1]
    else:
        if include_infinity:
            assert numpy.isneginf(r[0])
            assert r[1] == -fi.max
        else:
            assert r[0] == -fi.max
        if include_nan:
            assert numpy.isnan(r[-1])
            if include_infinity:
                assert numpy.isposinf(r[-2])
            else:
                assert r[-2] == fi.max
            for i in range(r.size - 2):
                assert r[i] < r[i + 1]
        else:
            if include_infinity:
                assert numpy.isposinf(r[-1])
            else:
                assert r[-1] == fi.max
            for i in range(r.size - 1):
                assert r[i] < r[i + 1]
        if include_zero:
            size = r.size
            if include_nan:
                size -= 1
            if include_infinity:
                if include_huge:
                    loc = (size - 1) // 2
                else:
                    loc = (size - 1) // 2
            else:
                if include_huge:
                    loc = (size - 1) // 2
                else:
                    loc = (size - 1) // 2
            assert r[loc] == 0, (include_nan, include_infinity, include_huge)
            if include_subnormal:
                assert r[loc + 1] == fi.smallest_subnormal
                assert r[loc - 1] == -fi.smallest_subnormal
            else:
                assert r[loc + 1] == fi.smallest_normal
                assert r[loc - 1] == -fi.smallest_normal


def test_real_samples(dtype):
    for size in range(6, 20):
        for include_huge in [False, True]:
            for include_subnormal in [False, True]:
                for include_infinity in [False, True]:
                    for include_zero in [False, True]:
                        for include_nan in [False, True]:
                            for nonnegative in [False, True]:
                                r = utils.real_samples(
                                    size=size,
                                    dtype=dtype,
                                    include_infinity=include_infinity,
                                    include_zero=include_zero,
                                    include_subnormal=include_subnormal,
                                    include_nan=include_nan,
                                    nonnegative=nonnegative,
                                    include_huge=include_huge,
                                )
                                assert r.dtype == dtype
                                _check_real_samples(
                                    r,
                                    include_infinity=include_infinity,
                                    include_zero=include_zero,
                                    include_subnormal=include_subnormal,
                                    include_nan=include_nan,
                                    nonnegative=nonnegative,
                                    include_huge=include_huge,
                                )


def test_complex_samples(dtype):
    for size in [(6, 6), (6, 7), (7, 6), (13, 13), (13, 15), (15, 13)]:
        for include_huge in [False, True]:
            for include_subnormal in [False, True]:
                for include_infinity in [False, True]:
                    for include_zero in [False, True]:
                        for include_nan in [False, True]:
                            for nonnegative in [False, True]:
                                r = utils.complex_samples(
                                    size=size,
                                    dtype=dtype,
                                    include_infinity=include_infinity,
                                    include_zero=include_zero,
                                    include_subnormal=include_subnormal,
                                    include_nan=include_nan,
                                    nonnegative=nonnegative,
                                    include_huge=include_huge,
                                )
                                re = r.real
                                im = r.imag
                                assert re.dtype == dtype
                                assert im.dtype == dtype
                                for i in range(r.shape[0]):
                                    _check_real_samples(
                                        re[i],
                                        include_infinity=include_infinity,
                                        include_zero=include_zero,
                                        include_subnormal=include_subnormal,
                                        include_nan=include_nan,
                                        nonnegative=nonnegative,
                                        include_huge=include_huge,
                                    )
                                for j in range(r.shape[1]):
                                    _check_real_samples(
                                        im[:, j],
                                        include_infinity=include_infinity,
                                        include_zero=include_zero,
                                        include_subnormal=include_subnormal,
                                        include_nan=include_nan,
                                        nonnegative=nonnegative,
                                        include_huge=include_huge,
                                    )


def test_isclose(dtype):
    fi = numpy.finfo(dtype)
    atol = fi.eps
    rtol = fi.resolution * 1e-1

    safe_min = numpy.sqrt(fi.tiny) * 4
    safe_max = numpy.sqrt(fi.max) / 8

    if dtype == numpy.float32:
        x1, y1 = dtype(1.234567), dtype(1.2345671)
        x2, y2 = dtype(1.23457), dtype(1.234571)
        x3, y3 = dtype(1.23457), dtype(1.234572)
        x4, y4 = dtype(1.23457), dtype(1.234573)
        x5, y5 = dtype(1.23457), dtype(1.234575)
    elif dtype == numpy.float64:
        x1, y1 = dtype(1.123456789234567), dtype(1.1234567892345671)
        x2, y2 = dtype(1.12345678923457), dtype(1.123456789234571)
        x3, y3 = dtype(1.12345678923457), dtype(1.123456789234572)
        x4, y4 = dtype(1.12345678923457), dtype(1.123456789234573)
        x5, y5 = dtype(1.12345678923457), dtype(1.123456789234575)
    else:
        assert 0  # unreachable

    for s in [1.0, 1e10, 1e-10, safe_min, safe_max, fi.tiny, fi.max * 1e-1]:
        x, y = x1 * s, y1 * s
        assert utils.isclose(x, y, atol * s, rtol), (x, y, s)
        x, y = x2 * s, y2 * s
        assert not utils.isclose(x, y, atol * s, rtol), (x, y, s)
        assert utils.isclose(x, y, atol * s * 10, rtol), (x, y, s)

        x, y = x3 * s, y3 * s
        assert not utils.isclose(x, y, atol * s, rtol), (x, y, s)
        assert utils.isclose(x, y, atol * s * 20, rtol), (x, y, s)

        x, y = x4 * s, y4 * s
        assert not utils.isclose(x, y, atol * s, rtol), (x, y, s)
        assert not utils.isclose(x, y, atol * s * 10, rtol), (x, y, s)
        assert utils.isclose(x, y, atol * s * 30, rtol), (x, y, s)

        x, y = x5 * s, y5 * s
        assert not utils.isclose(x, y, atol * s, rtol), (x, y, s)
        assert not utils.isclose(x, y, atol * s * 20, rtol), (x, y, s)
        assert utils.isclose(x, y, atol * s * 50, rtol), (x, y, s)
