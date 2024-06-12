import numpy
import pytest
import itertools

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


def _iter_samples_parameters():
    for include_huge, include_subnormal, include_infinity, include_zero, include_nan, nonnegative in itertools.product(
        *(([False, True],) * 6)
    ):
        yield dict(
            include_huge=include_huge,
            include_subnormal=include_subnormal,
            include_infinity=include_infinity,
            include_zero=include_zero,
            include_nan=include_nan,
            nonnegative=nonnegative,
        )


def test_real_samples(dtype):
    for size in range(6, 20):
        for params in _iter_samples_parameters():
            r = utils.real_samples(size=size, dtype=dtype, **params)
            assert r.dtype == dtype
            _check_real_samples(r, **params)


def test_complex_samples(dtype):
    for size in [(6, 6), (6, 7), (7, 6), (13, 13), (13, 15), (15, 13)]:
        for params in _iter_samples_parameters():
            r = utils.complex_samples(size=size, dtype=dtype, **params)
            re, im = r.real, r.imag
            assert re.dtype == dtype
            assert im.dtype == dtype
            for i in range(r.shape[0]):
                _check_real_samples(re[i], **params)
            for j in range(r.shape[1]):
                _check_real_samples(im[:, j], **params)


def test_real_pair_samples(dtype):
    for size in [(6, 6), (6, 7), (7, 6), (13, 13), (13, 15), (15, 13)]:
        for params in _iter_samples_parameters():
            s1 = utils.real_samples(size=size[0], dtype=dtype, **params).size
            s2 = utils.real_samples(size=size[1], dtype=dtype, **params).size
            r1, r2 = utils.real_pair_samples(size=size, dtype=dtype, **params)
            assert r1.dtype == dtype
            assert r2.dtype == dtype
            assert r1.size == s1 * s2
            assert r2.size == s1 * s2
            for r in r1.reshape(s2, s1):
                _check_real_samples(r, **params)
            for r in r2.reshape(s2, s1).T:
                _check_real_samples(r, **params)


def test_complex_pair_samples(dtype):
    for size1 in [(6, 6), (6, 7)]:
        for size2 in [(6, 6), (7, 6)]:
            for params in _iter_samples_parameters():
                s1 = utils.complex_samples(size=size1, dtype=dtype, **params).shape
                s2 = utils.complex_samples(size=size2, dtype=dtype, **params).shape
                r1, r2 = utils.complex_pair_samples(size=(size1, size2), dtype=dtype, **params)
                re1, im1 = r1.real, r1.imag
                re2, im2 = r2.real, r2.imag
                assert re1.dtype == dtype
                assert re2.dtype == dtype
                assert r1.shape == (s1[0] * s2[0], s1[1] * s2[1])
                assert r2.shape == (s1[0] * s2[0], s1[1] * s2[1])

                for i in range(0, s1[0] * s2[0], s1[0]):
                    for j in range(0, s1[1] * s2[1], s1[1]):
                        r = re1[i : i + s1[0], j : j + s1[1]]
                        for i1 in range(r.shape[0]):
                            _check_real_samples(r[i1], **params)
                        r = im1[i : i + s1[0], j : j + s1[1]]
                        for j1 in range(r.shape[1]):
                            _check_real_samples(r[:, j1], **params)

                for i in range(s1[0]):
                    for j in range(s1[1]):
                        r = re2[i :: s1[0], j :: s1[1]]
                        for i1 in range(r.shape[0]):
                            _check_real_samples(r[i1], **params)
                        r = im2[i :: s1[0], j :: s1[1]]
                        for j1 in range(r.shape[1]):
                            _check_real_samples(r[:, j1], **params)
