import numpy
import pytest
import itertools
import warnings
import functional_algorithms as fa
from functional_algorithms import utils


@pytest.fixture(scope="function", params=[numpy.float16, numpy.float32, numpy.float64, numpy.float128])
def real_dtype(request):
    return request.param


@pytest.fixture(scope="function", params=[numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
def dtype(request):
    return request.param


def test_mpfnextafter(real_dtype):
    if real_dtype == numpy.longdouble:
        pytest.skip(f"{real_dtype.__name__} support not implemented")
    import mpmath

    fi = numpy.finfo(real_dtype)
    size = 1000
    samples = list(
        utils.real_samples(size=size, dtype=real_dtype, include_infinity=False, include_subnormal=False, include_zero=False)
    )
    mp_ctx = mpmath.mp
    with mp_ctx.workprec(utils.get_precision(real_dtype)):
        inf = real_dtype(numpy.inf)
        inf_mp = utils.float2mpf(mp_ctx, inf)
        for x in samples:
            x_mp = utils.float2mpf(mp_ctx, x)
            result_mp = utils.mpfnextafter(x_mp, inf_mp)
            with warnings.catch_warnings(action="ignore"):
                expected = numpy.nextafter(x, inf)
            if numpy.isinf(expected) or abs(expected) < fi.smallest_normal:
                continue
            expected_mp = utils.float2mpf(mp_ctx, expected)
            assert result_mp > x_mp, (x, result_mp, expected)
            assert result_mp - x_mp == expected_mp - x_mp
        for x in samples:
            x_mp = utils.float2mpf(mp_ctx, x)
            result_mp = utils.mpfnextafter(x_mp, -inf_mp)
            with warnings.catch_warnings(action="ignore"):
                expected = numpy.nextafter(x, -inf)
            if numpy.isinf(expected) or abs(expected) < fi.smallest_normal:
                continue
            expected_mp = utils.float2mpf(mp_ctx, expected)
            assert result_mp < x_mp, (x, result_mp, expected)
            assert result_mp - x_mp == expected_mp - x_mp


def test_matching_bits(real_dtype):
    if real_dtype == numpy.longdouble:
        pytest.skip(f"{real_dtype.__name__} support not implemented")
    import mpmath

    fi = numpy.finfo(real_dtype)
    prec = abs(fi.negep)
    size = 150
    samples = list(utils.real_samples(size=size, dtype=real_dtype, include_infinity=False))
    max_result = -fi.negep

    mp_ctx = mpmath.mp
    with mp_ctx.workprec(utils.get_precision(real_dtype)):
        samples_mp = [utils.float2mpf(mp_ctx, x) for x in samples]

        for x, x_mp in zip(samples, samples_mp):
            b = utils.matching_bits(x, x)
            assert b == prec
            b = utils.matching_bits(x_mp, x_mp)
            assert b == prec

            y = numpy.nextafter(x, real_dtype(0))
            if x != y:
                b = utils.matching_bits(x, y)
                if abs(x) <= fi.smallest_normal:
                    assert b < prec
                else:
                    assert b == prec - 1

        for x in samples[:: size // 100 or 1] + samples[-1:]:
            d = 1
            last = -1000
            last_mp = -1000
            x_mp = utils.float2mpf(mp_ctx, x)
            for y, y_mp in zip(samples, samples_mp):
                b = utils.matching_bits(x, y)
                b1 = utils.matching_bits(y, x)
                assert b == b1
                b_mp = utils.matching_bits(x_mp, y_mp)
                b_mp1 = utils.matching_bits(y_mp, x_mp)
                assert b_mp == b_mp1
                if b_mp > -1 and 0:
                    print(x, y, b, b_mp)
                if x == y:
                    d = -1
                    assert b == max_result
                else:
                    if d == 1:
                        assert round(b) >= round(last - 2)
                        assert round(b_mp) >= round(last_mp - 2), (x, y)
                    else:
                        assert round(b) <= round(last + 2)
                        assert round(b_mp) <= round(last_mp + 2), (x, y)
                last = b
                last_mp = b_mp
            assert d == -1


def test_diff_ulp(real_dtype):
    if real_dtype == numpy.longdouble:
        pytest.skip(f"{real_dtype.__name__} support not implemented")
    fi = numpy.finfo(real_dtype)

    assert utils.diff_ulp(real_dtype(0), fi.tiny, flush_subnormals=True) == 1
    assert utils.diff_ulp(real_dtype(0), numpy.nextafter(fi.tiny, fi.max), flush_subnormals=True) == 2

    assert utils.diff_ulp(real_dtype(0), -fi.tiny, flush_subnormals=True) == 1
    assert utils.diff_ulp(real_dtype(0), fi.smallest_subnormal, flush_subnormals=False) == 1
    assert utils.diff_ulp(real_dtype(0), -fi.smallest_subnormal, flush_subnormals=False) == 1

    assert utils.diff_ulp(fi.tiny, fi.tiny, flush_subnormals=True) == 0
    assert utils.diff_ulp(fi.tiny, fi.tiny, flush_subnormals=False) == 0
    assert utils.diff_ulp(fi.tiny, numpy.nextafter(fi.tiny, fi.max), flush_subnormals=True) == 1
    assert utils.diff_ulp(fi.tiny, numpy.nextafter(fi.tiny, fi.max), flush_subnormals=False) == 1

    assert utils.diff_ulp(-fi.tiny, fi.tiny, flush_subnormals=True) == 2


def test_diff_log2ulp(real_dtype):
    if real_dtype == numpy.longdouble:
        pytest.skip(f"{real_dtype.__name__} support not implemented")
    bw = {numpy.float16: 16, numpy.float32: 32, numpy.float64: 64}[real_dtype]
    fi = numpy.finfo(real_dtype)

    assert utils.diff_log2ulp(fi.tiny, fi.tiny, flush_subnormals=True) == 0
    assert utils.diff_log2ulp(real_dtype(0), real_dtype(0), flush_subnormals=True) == 0

    assert utils.diff_log2ulp(real_dtype(0.3), real_dtype(0.003), flush_subnormals=True) == -fi.negep + 2
    assert utils.diff_log2ulp(real_dtype(0.3), real_dtype(0.03), flush_subnormals=True) == -fi.negep + 1
    assert utils.diff_log2ulp(real_dtype(0.3), real_dtype(0.3), flush_subnormals=True) == 0
    assert utils.diff_log2ulp(real_dtype(0.3), real_dtype(30), flush_subnormals=True) == -fi.negep + 2
    assert utils.diff_log2ulp(real_dtype(0.3), real_dtype(300), flush_subnormals=True) == -fi.negep + 3
    assert utils.diff_log2ulp(real_dtype(0.3), real_dtype(30000), flush_subnormals=True) == -fi.negep + 4

    assert utils.diff_log2ulp(real_dtype(0), fi.tiny, flush_subnormals=True) == 1
    assert utils.diff_log2ulp(-fi.tiny, fi.tiny, flush_subnormals=True) == 2

    assert utils.diff_log2ulp(real_dtype(0), fi.smallest_subnormal, flush_subnormals=False) == 1
    assert utils.diff_log2ulp(real_dtype(0), fi.tiny, flush_subnormals=False) == -fi.negep
    assert utils.diff_log2ulp(-fi.tiny, fi.tiny, flush_subnormals=False) == -fi.negep + 1

    assert utils.diff_log2ulp(real_dtype(0), fi.max, flush_subnormals=True) == bw - 1
    assert utils.diff_log2ulp(fi.min, fi.max, flush_subnormals=True) == bw
    assert utils.diff_log2ulp(real_dtype(0), real_dtype(numpy.inf), flush_subnormals=True) == bw + 1
    assert utils.diff_log2ulp(real_dtype(-numpy.inf), real_dtype(numpy.inf), flush_subnormals=True) == bw + 1
    assert utils.diff_log2ulp(real_dtype(numpy.inf), real_dtype(numpy.inf), flush_subnormals=True) == 0


def _check_real_samples(
    r,
    include_infinity=None,
    include_zero=None,
    include_subnormal=None,
    include_nan=None,
    nonnegative=None,
    include_huge=None,
    min_value=None,
    max_value=None,
):
    fi = numpy.finfo(r.dtype)
    size = r.size
    if min_value is not None or max_value is not None:
        if min_value is not None:
            assert r[0] == min_value
        if max_value is not None:
            assert r[-1] == max_value
        for i in range(r.size - 1):
            assert r[i] < r[i + 1]
    elif nonnegative:
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
                if include_huge and size > 9:
                    assert numpy.nextafter(r[-4], numpy.inf, dtype=r.dtype) == fi.max
                assert r[-3] == fi.max
                assert numpy.isposinf(r[-2])
            else:
                assert r[-2] == fi.max
                if include_huge and size > 9:
                    assert numpy.nextafter(r[-3], numpy.inf, dtype=r.dtype) == fi.max
            for i in range(size - 2):
                assert r[i] < r[i + 1]
        else:
            if include_infinity:
                if include_huge and size > 8:
                    assert numpy.nextafter(r[-3], numpy.inf, dtype=r.dtype) == fi.max
                assert r[-2] == fi.max
                assert numpy.isposinf(r[-1])
            else:
                assert r[-1] == fi.max
                if include_huge and size > 8:
                    assert numpy.nextafter(r[-2], numpy.inf, dtype=r.dtype) == fi.max
            for i in range(size - 1):
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
            loc = numpy.where(r == 0)[0][0]
            if include_subnormal:
                assert r[loc + 1] == fi.smallest_subnormal
                assert r[loc - 1] == -fi.smallest_subnormal
            else:
                assert r[loc + 1] == fi.smallest_normal
                assert r[loc - 1] == -fi.smallest_normal


def _iter_samples_parameters(dims=1, iscomplex=False):
    for (
        include_huge,
        include_subnormal,
        include_infinity,
        include_zero,
        include_nan,
        nonnegative,
    ) in itertools.product(*(([False, True],) * 6)):
        yield dict(
            include_huge=include_huge,
            include_subnormal=include_subnormal,
            include_infinity=include_infinity,
            include_zero=include_zero,
            include_nan=include_nan,
            nonnegative=nonnegative,
        )
    if iscomplex:
        for min_value, max_value in [(1, 10), (1, None), (-2, None), (-2, 2), (None, 2), (None, -2)]:
            yield dict(
                min_real_value=min_value,
                max_real_value=max_value,
                min_imag_value=min_value,
                max_imag_value=max_value,
            )
    else:
        for min_value, max_value in [(1, 10), (1, None), (-2, None), (-2, 2), (None, 2), (None, -2)]:
            yield dict(min_value=min_value, max_value=max_value)
        if dims == 2:
            for min_value, max_value in [((1, 2), (10, 8))]:
                yield dict(min_value=min_value, max_value=max_value)
        elif dims == 3:
            for min_value, max_value in [((1, 2, 3), (10, 8, 6))]:
                yield dict(min_value=min_value, max_value=max_value)


def test_real_samples(real_dtype):
    if real_dtype not in {numpy.float16, numpy.float32, numpy.float64}:
        pytest.skip(f"{real_dtype.__name__} not supported")
    for size in range(6, 20):
        for params in _iter_samples_parameters():
            r = utils.real_samples(size=size, dtype=real_dtype, **params)
            assert r.dtype == real_dtype
            _check_real_samples(r, **params)


def test_complex_samples(real_dtype):
    if real_dtype not in {numpy.float32, numpy.float64}:
        pytest.skip(f"{real_dtype.__name__} not supported")
    for size in [(6, 6), (6, 7), (7, 6), (13, 13), (13, 15), (15, 13)]:
        for params in _iter_samples_parameters(dims=1, iscomplex=True):
            min_real_value = params.get("min_real_value")
            max_real_value = params.get("max_real_value")
            min_imag_value = params.get("min_imag_value")
            max_imag_value = params.get("max_imag_value")
            params_re = params.copy()
            params_im = params.copy()
            params_re.pop("min_real_value", None)
            params_re.pop("max_real_value", None)
            params_re.pop("min_imag_value", None)
            params_re.pop("max_imag_value", None)
            params_im.pop("min_real_value", None)
            params_im.pop("max_real_value", None)
            params_im.pop("min_imag_value", None)
            params_im.pop("max_imag_value", None)
            params_re.update(min_value=min_real_value, max_value=max_real_value)
            params_im.update(min_value=min_imag_value, max_value=max_imag_value)
            r = utils.complex_samples(size=size, dtype=real_dtype, **params)
            re, im = r.real, r.imag
            assert re.dtype == real_dtype
            assert im.dtype == real_dtype
            for i in range(r.shape[0]):
                _check_real_samples(re[i], **params_re)
            for j in range(r.shape[1]):
                _check_real_samples(im[:, j], **params_im)


def test_real_pair_samples(real_dtype):
    if real_dtype not in {numpy.float16, numpy.float32, numpy.float64}:
        pytest.skip(f"{real_dtype.__name__} not supported")
    for size in [(6, 6), (6, 7), (7, 6), (13, 13), (13, 15), (15, 13)]:
        for params in _iter_samples_parameters(dims=2):
            min_value = params.get("min_value")
            max_value = params.get("max_value")
            params1 = params.copy()
            params2 = params.copy()
            if isinstance(min_value, tuple):
                params1.update(min_value=min_value[0])
                params2.update(min_value=min_value[1])
            if isinstance(max_value, tuple):
                params1.update(max_value=max_value[0])
                params2.update(max_value=max_value[1])
            s1 = utils.real_samples(size=size[0], dtype=real_dtype, **params1).size
            s2 = utils.real_samples(size=size[1], dtype=real_dtype, **params2).size
            r1, r2 = utils.real_pair_samples(size=size, dtype=real_dtype, **params)
            assert r1.dtype == real_dtype
            assert r2.dtype == real_dtype
            assert r1.size == s1 * s2
            assert r2.size == s1 * s2
            for r in r1.reshape(s2, s1):
                _check_real_samples(r, **params1)
            for r in r2.reshape(s2, s1).T:
                _check_real_samples(r, **params2)


def test_complex_pair_samples(real_dtype):
    if real_dtype not in {numpy.float32, numpy.float64}:
        pytest.skip(f"{real_dtype.__name__} not supported")
    for size1 in [(6, 6), (6, 7)]:
        for size2 in [(6, 6), (7, 6)]:
            for params in _iter_samples_parameters(dims=2, iscomplex=True):
                min_real_value = params.get("min_real_value")
                max_real_value = params.get("max_real_value")
                min_imag_value = params.get("min_imag_value")
                max_imag_value = params.get("max_imag_value")
                params1 = params.copy()
                params2 = params.copy()
                params_re = params.copy()
                params_im = params.copy()
                params_re.pop("min_real_value", None)
                params_re.pop("max_real_value", None)
                params_re.pop("min_imag_value", None)
                params_re.pop("max_imag_value", None)
                params_im.pop("min_real_value", None)
                params_im.pop("max_real_value", None)
                params_im.pop("min_imag_value", None)
                params_im.pop("max_imag_value", None)
                params1_re = params_re.copy()
                params2_re = params_re.copy()
                params1_im = params_im.copy()
                params2_im = params_im.copy()
                min_real_values = utils._fix_limit_value(min_real_value, dims=2)
                max_real_values = utils._fix_limit_value(max_real_value, dims=2)
                min_imag_values = utils._fix_limit_value(min_imag_value, dims=2)
                max_imag_values = utils._fix_limit_value(max_imag_value, dims=2)

                params1.update(
                    min_real_value=min_real_values[0],
                    max_real_value=max_real_values[0],
                    min_imag_value=min_imag_values[0],
                    max_imag_value=max_imag_values[0],
                )
                params2.update(
                    min_real_value=min_real_values[1],
                    max_real_value=max_real_values[1],
                    min_imag_value=min_imag_values[1],
                    max_imag_value=max_imag_values[1],
                )
                params1_re.update(
                    min_value=min_real_values[0],
                    max_value=max_real_values[0],
                )
                params2_re.update(
                    min_value=min_real_values[1],
                    max_value=max_real_values[1],
                )
                params1_im.update(
                    min_value=min_imag_values[0],
                    max_value=max_imag_values[0],
                )
                params2_im.update(
                    min_value=min_imag_values[1],
                    max_value=max_imag_values[1],
                )

                s1 = utils.complex_samples(size=size1, dtype=real_dtype, **params1).shape
                s2 = utils.complex_samples(size=size2, dtype=real_dtype, **params2).shape
                r1, r2 = utils.complex_pair_samples(size=(size1, size2), dtype=real_dtype, **params)
                re1, im1 = r1.real, r1.imag
                re2, im2 = r2.real, r2.imag
                assert re1.dtype == real_dtype
                assert re2.dtype == real_dtype
                assert r1.shape == (s1[0] * s2[0], s1[1] * s2[1])
                assert r2.shape == (s1[0] * s2[0], s1[1] * s2[1])

                for i in range(0, s1[0] * s2[0], s1[0]):
                    for j in range(0, s1[1] * s2[1], s1[1]):
                        r = re1[i : i + s1[0], j : j + s1[1]]
                        for i1 in range(r.shape[0]):
                            _check_real_samples(r[i1], **params1_re)
                        r = im1[i : i + s1[0], j : j + s1[1]]
                        for j1 in range(r.shape[1]):
                            _check_real_samples(r[:, j1], **params1_im)

                for i in range(s1[0]):
                    for j in range(s1[1]):
                        r = re2[i :: s1[0], j :: s1[1]]
                        for i1 in range(r.shape[0]):
                            _check_real_samples(r[i1], **params2_re)
                        r = im2[i :: s1[0], j :: s1[1]]
                        for j1 in range(r.shape[1]):
                            _check_real_samples(r[:, j1], **params2_im)


def test_periodic_samples(real_dtype):
    if real_dtype not in {numpy.float16, numpy.float32, numpy.float64}:
        pytest.skip(f"{real_dtype.__name__} not supported")
    for period in [1, 3.14, numpy.pi / 2]:
        for size in [10, 11, 51]:
            for periods in [4, 5, 13]:
                samples = utils.periodic_samples(period=period, size=size, dtype=real_dtype, periods=periods)
                assert samples.dtype == real_dtype
                assert samples.size == size * periods
                assert numpy.diff(samples).min() >= 0
                assert numpy.diff(samples).max() > 0
                for i in range(periods):
                    samples_per_period = samples[i * size : (i + 1) * size]
                    d = utils.diff_ulp(samples_per_period[:-1], samples_per_period[1:])
                    if periods % 2 and i == periods // 2:
                        # exclude zero:
                        d1 = d[: size // 2 - 1]
                        d2 = d[size // 2 + 1 :]
                        assert numpy.diff(d1).min() >= -1
                        assert numpy.diff(d1).max() <= 1

                        assert numpy.diff(d2).min() >= -1
                        assert numpy.diff(d2).max() <= 1
                    else:
                        assert numpy.diff(d).min() >= -1
                        assert numpy.diff(d).max() <= 1

                    p = samples_per_period[-1] - samples_per_period[0]
                    assert p <= period * size / (size - 1) and p >= period * (size - 1) / size


def test_real_triple_samples(real_dtype):
    if real_dtype not in {numpy.float16, numpy.float32, numpy.float64}:
        pytest.skip(f"{real_dtype.__name__} not supported")
    for size in [(6, 6, 6), (6, 7, 8), (13, 7, 6)]:
        for params in _iter_samples_parameters():
            min_value = params.get("min_value")
            max_value = params.get("max_value")
            params1 = params.copy()
            params2 = params.copy()
            params3 = params.copy()
            if isinstance(min_value, tuple):
                params1.update(min_value=min_value[0])
                params2.update(min_value=min_value[1])
                params3.update(min_value=min_value[2])
            if isinstance(max_value, tuple):
                params1.update(max_value=max_value[0])
                params2.update(max_value=max_value[1])
                params3.update(max_value=max_value[2])
            s1 = utils.real_samples(size=size[0], dtype=real_dtype, **params1).size
            s2 = utils.real_samples(size=size[1], dtype=real_dtype, **params2).size
            s3 = utils.real_samples(size=size[2], dtype=real_dtype, **params3).size
            r1, r2, r3 = utils.real_triple_samples(size=size, dtype=real_dtype, **params)
            assert r1.dtype == real_dtype
            assert r2.dtype == real_dtype
            assert r3.dtype == real_dtype
            assert r1.size == s1 * s2 * s3
            assert r2.size == s1 * s2 * s3
            assert r3.size == s1 * s2 * s3
            for r in r3.reshape(s1, s2, s3):
                for r_ in r:
                    _check_real_samples(r_, **params3)
            for r in r2.reshape(s1, s2, s3):
                for r_ in r.T:
                    _check_real_samples(r_, **params2)
            for r in r1.reshape(s1, s2, s3).T:
                for r_ in r:
                    _check_real_samples(r_, **params1)


@pytest.fixture(scope="function", params=["mpmath", "jax"])
def backend(request):
    return request.param


@pytest.fixture(scope="function", params=["cpu", "cuda"])
def device(request):
    return request.param


def _square(x):
    return x * x


def test_vectorize_with_backend(backend, dtype, device):
    pytest.importorskip(backend)

    assert_equal = numpy.testing.assert_equal

    vectorize_with_backend = getattr(utils, f"vectorize_with_{backend}")

    if not vectorize_with_backend.backend_is_available(device):
        pytest.skip(f"{device} support is unavailable")

    func = vectorize_with_backend(_square, device=device)

    arr = numpy.array([1, 2, 3], dtype=dtype)
    expected = _square(arr)

    def test(result):
        assert isinstance(result, numpy.ndarray)
        assert result.dtype == expected.dtype
        assert_equal(result, expected)

    test(func(arr))
    # Using call for JAX backend is slow and unreasonable, so
    # multiprocessing is disabled for JAX:
    test(func.call(arr, workers=1 if backend == "jax" else None))


def test_numpy_with_backend(backend, dtype, device):
    pytest.importorskip(backend)

    assert_equal = numpy.testing.assert_equal

    numpy_with_backend = getattr(utils, f"numpy_with_{backend}")(device=device)

    func = numpy_with_backend.square

    if not func.backend_is_available(device):
        pytest.skip(f"{device} support is unavailable")

    arr = numpy.array([1, 2, 3], dtype=dtype)

    expected = numpy.square(arr)

    def test(result):
        assert isinstance(result, numpy.ndarray)
        assert result.dtype == expected.dtype
        assert_equal(result, expected)

    test(func(arr))


@pytest.mark.parametrize("prec_multiplier", [1, 2])
def test_float2mpf(real_dtype, prec_multiplier):
    import mpmath

    ctx = mpmath.mp
    fi = numpy.finfo(real_dtype)
    dtype_name = real_dtype.__name__
    dtype_name = dict(longdouble="float128").get(dtype_name, dtype_name)

    with ctx.workprec(-fi.negep * prec_multiplier):
        for f in [
            -numpy.inf,
            -utils.vectorize_with_mpmath.float_max[dtype_name],
            -150,
            -0.3,
            -fi.eps,
            -fi.smallest_normal,
            -fi.smallest_subnormal * 15,
            -fi.smallest_subnormal,
            0,
            fi.smallest_subnormal,
            fi.smallest_subnormal * 10,
            fi.smallest_normal,
            fi.eps,
            0.3,
            150,
            utils.vectorize_with_mpmath.float_max[dtype_name],
            numpy.inf,
            numpy.nan,
        ]:
            v = real_dtype(f)
            m = utils.float2mpf(ctx, v)
            r = utils.mpf2float(real_dtype, m)
            assert type(r) is type(v)
            if not numpy.isnan(f):
                assert r == v

            if real_dtype != numpy.longdouble:
                assert utils.mpf2bin(m) == utils.float2bin(v)

        if real_dtype in {numpy.float32, numpy.float64}:
            for f in utils.real_samples(1000, dtype=real_dtype, min_value=-1, max_value=1):
                v = real_dtype(f)
                m = utils.float2mpf(ctx, v)
                r = utils.mpf2float(real_dtype, m)
                assert type(r) is type(v)
                if not numpy.isnan(f):
                    assert r == v
                assert utils.mpf2bin(m) == utils.float2bin(v)


def test_split_veltkamp(real_dtype):

    fi = numpy.finfo(real_dtype)
    dtype_name = real_dtype.__name__
    dtype_name = dict(longdouble="float128").get(dtype_name, dtype_name)
    if dtype_name == "float128":
        pytest.skip(f"{dtype_name} support not implemented")
    p = {numpy.float16: 11, numpy.float32: 24, numpy.float64: 53, numpy.longdouble: 64}[real_dtype]
    for f in [
        -utils.vectorize_with_mpmath.float_max[dtype_name],
        -150,
        -0.3,
        -fi.eps,
        -fi.smallest_normal,
        -fi.smallest_subnormal * 15,
        -fi.smallest_subnormal,
        0,
        fi.smallest_subnormal,
        fi.smallest_subnormal * 10,
        fi.smallest_normal,
        fi.eps,
        0.3,
        150,
        utils.vectorize_with_mpmath.float_max[dtype_name],
        numpy.pi,
    ]:
        x = real_dtype(f)
        s1 = (p + 1) // 2
        with numpy.errstate(over="ignore", invalid="ignore"):
            max_s = utils.split_veltkamp_max(x)
        for s in range(2, max_s):
            xh, xl = utils.split_veltkamp(x, s)
            assert x == xh + xl

            bh = utils.tobinary(xh).split("p")[0].lstrip("-")
            bl = utils.tobinary(xl).split("p")[0].lstrip("-")
            bh = bh[1 + bh.startswith("1.") :].lstrip("0")
            bl = bl[1 + bl.startswith("1.") :].lstrip("0")
            if bh.endswith("1"):
                bh = bh[:-1].rstrip("0")
            if bl.endswith("1"):
                bl = bl[:-1].rstrip("0")

            if isinstance(x, numpy.longdouble) and s != s1:
                continue

            assert len(bh) <= p - s
            assert len(bl) <= s

    for x in utils.real_samples(1000, dtype=real_dtype, include_infinity=False):
        if abs(x) > numpy.sqrt(fi.max):
            # veltkamp splitting does not work reliably with large numbers
            continue
        s = (p + 1) // 2
        xh, xl = utils.split_veltkamp(x, s)
        assert x == xh + xl

        bh = utils.tobinary(xh).split("p")[0].lstrip("-")
        bl = utils.tobinary(xl).split("p")[0].lstrip("-")
        bh = bh[1 + bh.startswith("1.") :].lstrip("0")
        bl = bl[1 + bl.startswith("1.") :].lstrip("0")
        if bh.endswith("1"):
            bh = bh[:-1].rstrip("0")
        if bl.endswith("1"):
            bl = bl[:-1].rstrip("0")

        assert len(bh) <= p - s
        assert len(bl) <= s


def test_multiply_dekker(real_dtype):
    import mpmath

    fi = numpy.finfo(real_dtype)
    dtype_name = real_dtype.__name__
    dtype_name = dict(longdouble="float128").get(dtype_name, dtype_name)
    if dtype_name == "float128":
        pytest.skip("longdouble not supported")

    ctx = mpmath.mp
    ctx.prec = utils.vectorize_with_mpmath.float_prec[real_dtype.__name__] * 2

    size = 100
    for x in utils.real_samples(size, dtype=real_dtype, include_infinity=False):
        for y in utils.real_samples(size, dtype=real_dtype, include_infinity=False):
            with numpy.errstate(over="ignore", invalid="ignore"):
                r1, r2 = utils.multiply_dekker(x, y)

            if not (numpy.isfinite(r1) and numpy.isfinite(r2)):
                continue

            x_mp = utils.float2mpf(ctx, x)
            y_mp = utils.float2mpf(ctx, y)

            r1_mp = utils.float2mpf(ctx, r1)
            r2_mp = utils.float2mpf(ctx, r2)

            if abs(r2) >= fi.smallest_normal:
                assert x_mp * y_mp == r1_mp + r2_mp, (x_mp * y_mp, r1_mp + r2_mp, r1, r2)
            else:
                # Dekker product is inaccurate when r2 is subnormal
                pass


def test_square_dekker(real_dtype):
    import mpmath

    fi = numpy.finfo(real_dtype)
    dtype_name = real_dtype.__name__
    dtype_name = dict(longdouble="float128").get(dtype_name, dtype_name)
    if dtype_name == "float128":
        pytest.skip("longdouble not supported")

    ctx = mpmath.mp
    ctx.prec = utils.vectorize_with_mpmath.float_prec[real_dtype.__name__] * 2

    size = 100
    for x in utils.real_samples(size, dtype=real_dtype, include_infinity=False):
        with numpy.errstate(over="ignore", invalid="ignore"):
            r1, r2 = utils.square_dekker(x)

        x_mp = utils.float2mpf(ctx, x)

        r1_mp = utils.float2mpf(ctx, r1)
        r2_mp = utils.float2mpf(ctx, r2)

        if abs(r2) >= fi.smallest_normal:
            assert x_mp * x_mp == r1_mp + r2_mp
        else:
            # Dekker product is inaccurate when r2 is subnormal
            pass


def test_mpf2multiword(real_dtype):
    import mpmath

    dtype = real_dtype

    prec = utils.vectorize_with_mpmath.float_prec[dtype.__name__]
    max_m = {numpy.float16: 3, numpy.float32: 7, numpy.float64: 21, numpy.longdouble: 256}[dtype]
    max_m = min(max_m, 25)  # longdouble will be partially tested
    for m in range(1, max_m + 1):
        working_prec = prec * m
        with mpmath.mp.workprec(working_prec):
            ctx = mpmath.mp
            for x in [ctx.pi + 0, ctx.log(2)]:
                lst = utils.mpf2multiword(dtype, x)
                for w in lst:
                    assert w > 0
                result = sum([utils.float2mpf(ctx, w) for w in lst], utils.float2mpf(ctx, dtype(0)))
                assert result == x or abs(utils.mpf2float(dtype, result - x)) == 0
                if lst[-1] == 0:
                    break


def test_mpf2multiword_log2(real_dtype):
    """Split log(2) into hi and lo so that

      k * log(2) = k * hi + k * lo

    is exact for all integral values 1 <= k <= kmax where

      kmax = 2 * (bitwidth / 8) ** 3

    kmax is defined as the maximal k value such that

      exp(k * log(2)) is finite

    within the given floating-point system.

    The p argument of mpf2multiword defines the precision of
    multiwords that maximal value is the precision of the
    floating-point system. However, depending on the expected
    cumulative precision of the multiword, there exists a subset of p
    values for which the expected cumulative precision will be achived
    for any k <= kmax. The following table is provided to help
    choosing an optimal p values for various dtypes:

              | expected    |                              | the number
              | cumulative  |                              | of
    dtype     | precision   | p values                     | words
    ----------+-------------+------------------------------+------------
    float64   | 1075        | 29, 30, 34:37, 39, 40, 42:45 | 37...24
              | 1060        | 24, 28, 32, 33, 36:38, 40:45 | 44...24
              | 795         | 31, 32, 35, 37:43, 45        | 26...18
              | 530         | 28:36:2, 37-43, 45           | 19...12
              | 265         | 27, 30, 31, 34:36, 38:43, 45 | 10...6
              | 212         | 27, 31:32, 36:40, 43:45      | 8...5
              | 159         | 27:28, 32:37, 40:45          | 6...4
              | 106         | 26:28, 34:45                 | 4...3
              | 80          | 26:32, 39:45                 | 3...2
              | 53          | 27:45                        | 2
    ----------+-------------+------------------------------+------------
    float32   | 148         | 14:15, 17:19                 | 11...8
              | 144         | 16, 18, 19                   | 9...8
              | 120         | 15, 18                       | 8...7
              | 72          | 12, 15, 16, 18, 19           | 6...4
              | 48          | 12:14, 16:19                 | 4...3
              | 36          | 12:14, 18, 19                | 3...2
              | 24          | 11:19                        | 2
    ----------+-------------+------------------------------+------------
    float16   | 27          | 7:9                          | 3
              | 22          | 7:9                          | 3
              | 16          | 8, 9                         | 2
              | 11          | 5:8                          | 2

    For example::

      with mpmath.workprec(53 * 2):
          print(utils.mpf2multiword(numpy.float64, mpmath.log(2), p=32, max_length=2))

    outputs

      [0.6931471803691238, 1.9082149292705877e-10]

    that contains the values of Ln2Hi and Ln2Lo in libgo/go/math/expm1.go

    """
    if real_dtype == numpy.longdouble:
        pytest.skip(f"test not implemented for {real_dtype.__name__}")
    import mpmath

    dtype = real_dtype

    bitwidth = {numpy.float16: 16, numpy.float32: 32, numpy.float64: 64}[dtype]
    kmax = 2 * (bitwidth // 8) ** 3 // 4 + 1

    dtype_p = utils.vectorize_with_mpmath.float_prec[dtype.__name__]
    max_p = {numpy.float16: 9, numpy.float32: 19, numpy.float64: 45}[dtype]
    min_p = {numpy.float16: 5, numpy.float32: 11, numpy.float64: 26}[dtype]
    working_prec = int(dtype_p * 2)

    print(f"\n{kmax=} {min_p=} {max_p=} {working_prec=}")
    with mpmath.workprec(working_prec):
        ctx = mpmath.mp
        ln2 = ctx.log(2)
        stats = dict(non_redundant=0, redundant=0)
        for p in range(min_p, max_p + 1):
            lst = utils.mpf2multiword(dtype, ln2, p=p)

            # Accept only multiwords that provide exact representations of
            #   k * ln2
            # for all 1 <= k <= kmax.
            ok = True
            for k in range(1, kmax + 1):
                exact = k * ln2
                k_np = dtype(k)
                exact_native_split = sum(
                    [utils.float2mpf(ctx, k_np * v) for v in reversed(lst)], utils.float2mpf(ctx, dtype(0))
                )
                if exact_native_split != exact:
                    ok = False
                    break
            if not ok:
                continue

            # Find minimal exact multiwords of ln2
            for n in range(1, len(lst)):
                lst2 = utils.mpf2multiword(dtype, ln2, p=p, max_length=n + 1)
                assert len(lst2) <= n + 1

                ok = True
                for k in range(1, kmax + 1):
                    exact = k * ln2
                    k_np = dtype(k)
                    exact_native_split2 = sum(
                        [utils.float2mpf(ctx, k_np * v) for v in reversed(lst2)], utils.float2mpf(ctx, dtype(0))
                    )
                    if exact_native_split2 != exact:
                        ok = False
                        break
                if ok:
                    if len(lst) == len(lst2):
                        print(f"{p=} {lst=}", flush=True)
                        stats["non_redundant"] += 1
                    else:
                        print(f"{p=} {lst2=}", flush=True)
                        stats["redundant"] += 1
                    break
        assert stats["non_redundant"] > 0


def test_ulp(real_dtype):
    if real_dtype == numpy.longdouble:
        pytest.skip(f"test not implemented for {real_dtype.__name__}")
    import math

    size = 10000
    for x in utils.real_samples(size, dtype=real_dtype, include_huge=True):
        if real_dtype == numpy.float64:
            assert math.ulp(float(x)) == utils.ulp(x), (x, numpy.frexp(x))
        if numpy.isfinite(x):
            with warnings.catch_warnings(action="ignore"):
                if x > 0:
                    y = numpy.nextafter(x, real_dtype("inf"))
                    y1 = x + utils.ulp(x)
                    assert y == y1, (x, y, y1, utils.ulp(x))
                else:
                    y = numpy.nextafter(x, real_dtype("-inf"))
                    y1 = x - utils.ulp(x)
                    assert y == y1, (x, y, y1, utils.ulp(x))
        elif numpy.isposinf(x):
            assert utils.ulp(x) == real_dtype("inf")
        elif numpy.isneginf(x):
            assert utils.ulp(x) == real_dtype("inf")
        else:
            assert 0  # unreachable

    nan = real_dtype("nan")
    assert numpy.isnan(utils.ulp(nan))


def test_overlapping(real_dtype):
    if real_dtype == numpy.longdouble:
        pytest.skip(f"test not implemented for {real_dtype.__name__}")
    size = 10000
    for x in utils.real_samples(size, dtype=real_dtype, include_infinity=False):
        with warnings.catch_warnings(action="ignore"):
            x1 = x / real_dtype(7)
            y1 = x + x1
            y2 = x1 - (y1 - x)
        if not numpy.isfinite(y1) or y1 == 0:
            continue
        assert utils.overlapping(x, x1)
        assert not utils.overlapping(y1, y2)
        assert utils.overlapping(y1, x)
        assert utils.overlapping(y1, x1)
        assert y1 + y2 == y1
        assert x + x1 == y1


def test_float2fraction(real_dtype):
    import mpmath

    if real_dtype == numpy.longdouble:
        pytest.skip(f"support not implemented for {real_dtype.__name__}")

    prec = {numpy.float16: 11, numpy.float32: 24, numpy.float64: 53}[real_dtype]

    mp_ctx = mpmath.mp
    for x in utils.real_samples(10_000, dtype=real_dtype, include_subnormal=True):
        q = utils.float2fraction(x)
        r = utils.fraction2float(real_dtype, q)
        assert x == r
        with mp_ctx.workprec(prec):
            m = utils.float2mpf(mp_ctx, x)
            f = utils.float2fraction(m)
        assert f == q


def test_bin2float(real_dtype):
    if real_dtype == numpy.longdouble:
        pytest.skip(f"test not implemented for {real_dtype.__name__}")
    for x in utils.real_samples(10_000, dtype=real_dtype, include_subnormal=True, include_infinity=not False):
        b = utils.float2bin(x)
        y = utils.bin2float(real_dtype, b)
        assert x == y


def test_fma_samples_fraction(real_dtype):
    if real_dtype == numpy.longdouble:
        pytest.skip(f"test not implemented for {real_dtype.__name__}")

    import mpmath

    mp_ctx = mpmath.mp
    fi = numpy.finfo(real_dtype)

    size = 30
    x, y, z = fa.utils.real_triple_samples((size, size, size), dtype=real_dtype, target_func="fma", include_infinity=False)

    x_f = list(map(fa.utils.float2fraction, x))
    y_f = list(map(fa.utils.float2fraction, y))
    z_f = list(map(fa.utils.float2fraction, z))

    result_f = numpy.array(
        [
            fa.utils.fraction2float(real_dtype, x_ * y_ + z_, prec=fa.utils.get_precision(real_dtype))
            for x_, y_, z_ in zip(x_f, y_f, z_f)
        ],
        dtype=real_dtype,
    )

    x_mp = fa.utils.float2mpf(mp_ctx, x)
    y_mp = fa.utils.float2mpf(mp_ctx, y)
    z_mp = fa.utils.float2mpf(mp_ctx, z)

    result_mp = numpy.array(
        [
            fa.utils.mpf2float(real_dtype, mp_ctx.fadd(mp_ctx.fmul(x_, y_, exact=True), z_, exact=True))
            for x_, y_, z_ in zip(x_mp, y_mp, z_mp)
        ],
        dtype=real_dtype,
    )

    numpy.testing.assert_equal(result_f, result_mp)

    # that is, doing fma computation with infinite precision floats
    # and fractions are equivalent.


def test_fma_samples_pyfma(real_dtype):
    """Test the equivalence of pyfma and mpmath exact calculations.

    pyfma is much faster than mpmath and therefore we want to use it
    as a reference for fma whenever possible.
    """
    if real_dtype in {numpy.float16, numpy.longdouble}:
        pytest.skip(f"test not implemented for {real_dtype.__name__}")

    try:
        import pyfma
    except ImportError as msg:
        pytest.skip(f"test requires pyfma: {msg}")

    import mpmath

    mp_ctx = mpmath.mp

    fi = numpy.finfo(real_dtype)

    size = 30
    x, y, z = fa.utils.real_triple_samples((size, size, size), dtype=real_dtype, target_func="fma", include_infinity=False)

    result_f = pyfma.fma(x, y, z)
    assert result_f.dtype == real_dtype

    # Filter out subnormals as mpmath rounding of subnormals may be
    # incorrect
    normal_indices = numpy.where(abs(result_f) >= fi.smallest_normal)
    subnormal_indices = numpy.where(abs(result_f) < fi.smallest_normal)

    x_mp = fa.utils.float2mpf(mp_ctx, x)
    y_mp = fa.utils.float2mpf(mp_ctx, y)
    z_mp = fa.utils.float2mpf(mp_ctx, z)
    result_mp = numpy.array(
        [
            fa.utils.mpf2float(real_dtype, mp_ctx.fadd(mp_ctx.fmul(x_, y_, exact=True), z_, exact=True))
            for x_, y_, z_ in zip(x_mp, y_mp, z_mp)
        ],
        dtype=real_dtype,
    )

    numpy.testing.assert_equal(result_f[normal_indices], result_mp[normal_indices])
    assert result_mp[subnormal_indices].max() <= fi.smallest_normal
