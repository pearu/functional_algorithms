import numpy
import pytest
import itertools

from functional_algorithms import utils


@pytest.fixture(scope="function", params=[numpy.float16, numpy.float32, numpy.float64, numpy.float128])
def real_dtype(request):
    return request.param


@pytest.fixture(scope="function", params=[numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
def dtype(request):
    return request.param


def test_diff_ulp(real_dtype):
    if real_dtype == numpy.longdouble:
        pytest.skip(f"support not implemented")
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


def _check_real_samples(
    r,
    include_infinity=None,
    include_zero=None,
    include_subnormal=None,
    include_nan=None,
    nonnegative=None,
    include_huge=None,
):
    fi = numpy.finfo(r.dtype)
    size = r.size
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


def _iter_samples_parameters():
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


def test_real_samples(real_dtype):
    if real_dtype not in {numpy.float32, numpy.float64}:
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
        for params in _iter_samples_parameters():
            r = utils.complex_samples(size=size, dtype=real_dtype, **params)
            re, im = r.real, r.imag
            assert re.dtype == real_dtype
            assert im.dtype == real_dtype
            for i in range(r.shape[0]):
                _check_real_samples(re[i], **params)
            for j in range(r.shape[1]):
                _check_real_samples(im[:, j], **params)


def test_real_pair_samples(real_dtype):
    if real_dtype not in {numpy.float32, numpy.float64}:
        pytest.skip(f"{real_dtype.__name__} not supported")
    for size in [(6, 6), (6, 7), (7, 6), (13, 13), (13, 15), (15, 13)]:
        for params in _iter_samples_parameters():
            s1 = utils.real_samples(size=size[0], dtype=real_dtype, **params).size
            s2 = utils.real_samples(size=size[1], dtype=real_dtype, **params).size
            r1, r2 = utils.real_pair_samples(size=size, dtype=real_dtype, **params)
            assert r1.dtype == real_dtype
            assert r2.dtype == real_dtype
            assert r1.size == s1 * s2
            assert r2.size == s1 * s2
            for r in r1.reshape(s2, s1):
                _check_real_samples(r, **params)
            for r in r2.reshape(s2, s1).T:
                _check_real_samples(r, **params)


def test_complex_pair_samples(real_dtype):
    if real_dtype not in {numpy.float32, numpy.float64}:
        pytest.skip(f"{real_dtype.__name__} not supported")
    for size1 in [(6, 6), (6, 7)]:
        for size2 in [(6, 6), (7, 6)]:
            for params in _iter_samples_parameters():
                s1 = utils.complex_samples(size=size1, dtype=real_dtype, **params).shape
                s2 = utils.complex_samples(size=size2, dtype=real_dtype, **params).shape
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


def test_periodic_samples(real_dtype):
    if real_dtype not in {numpy.float32, numpy.float64}:
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


def test_float2mpf(real_dtype):
    import mpmath

    ctx = mpmath.mp
    ctx.prec = utils.vectorize_with_mpmath.float_prec[real_dtype.__name__]
    fi = numpy.finfo(real_dtype)
    dtype_name = real_dtype.__name__
    dtype_name = dict(longdouble="float128").get(dtype_name, dtype_name)
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
        for f in utils.real_samples(1000, dtype=real_dtype):
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
        pytest.skip("NOT IMPL")
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

            x_mp = utils.float2mpf(ctx, x)
            y_mp = utils.float2mpf(ctx, y)

            r1_mp = utils.float2mpf(ctx, r1)
            r2_mp = utils.float2mpf(ctx, r2)

            if abs(r2) >= fi.smallest_normal:
                assert x_mp * y_mp == r1_mp + r2_mp
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
