import numpy
import pytest
import warnings
from collections import defaultdict
import functional_algorithms as fa


@pytest.fixture(scope="function", params=[numpy.float16, numpy.float32, numpy.float64, numpy.float128])
def dtype(request):
    return request.param


@pytest.mark.parametrize(
    "overlapping,length",
    [
        ("non-overlapping", 2),
        ("non-overlapping", 3),
        ("non-overlapping", 4),
        ("overlapping", 2),
        ("overlapping", 3),
        ("overlapping", 4),
    ],
)
def test_expansion_samples(dtype, overlapping, length):
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    fi = numpy.finfo(dtype)
    size = 4000
    for x in fa.utils.expansion_samples(size, length, overlapping=overlapping, dtype=dtype):
        for i in range(len(x) - 1):
            if not overlapping:
                assert not fa.utils.overlapping(x[i], x[i + 1]) or (x[i] == 0 and x[i + 1] == 0)
            else:
                assert fa.utils.overlapping(x[i], x[i + 1]) or x[i + 1] == 0

        for i in range(len(x)):
            assert x[i] == 0 or abs(x[i]) >= fi.smallest_normal, (x, i)


@pytest.mark.parametrize(
    "functional,fast",
    [
        ("non-functional", "fast_two_sum"),
        ("non-functional", "two_sum"),
        ("functional", "fast_two_sum"),
        ("functional", "two_sum"),
    ],
)
def test_renormalize(dtype, functional, fast):
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    import mpmath

    functional = {"non-functional": False, "functional": True}[functional]
    fast = {"fast_two_sum": True, "two_sum": False}[fast]
    max_prec = {numpy.float16: 4 * 11, numpy.float32: 4 * 24, numpy.float64: 4 * 53}[dtype]
    size = 100
    ctx = fa.utils.NumpyContext(dtype)
    fi = numpy.finfo(dtype)
    c = numpy.ldexp(dtype(1), fi.negep + 6)
    mp_ctx = mpmath.mp
    samples = numpy.array(fa.utils.real_samples(size, dtype=dtype, include_infinity=False, include_huge=False))
    samples3_lst = []
    results3_lst = []
    with mp_ctx.workprec(max_prec):
        for x1 in samples:
            x2 = x1 / dtype(3) * c
            x3 = x2 / dtype(13) * c
            x4 = x3 / dtype(19) * c
            assert fa.utils.overlapping(x1, x2) or x2 == 0
            assert fa.utils.overlapping(x2, x3) or x3 == 0
            assert fa.utils.overlapping(x3, x4) or x4 == 0

            for e1 in [
                [x1],
                [x1, x2],
                [x1, x1],
                [x1, x3, x4],
                [x1, x2, x3, x4],
                [x1, -x2, x3, -x4],
                *(([x3, x1], [x4, x3, x2, x1], [x1, x4, x2, x3]) if not fast else ()),
            ]:

                with warnings.catch_warnings(action="ignore"):
                    if not numpy.isfinite(sum(e1, dtype(0))):
                        continue

                e2 = fa.apmath.renormalize(ctx, e1, functional=functional, fast=fast)
                if functional and len(e1) == 3:
                    samples3_lst.append(e1)
                    results3_lst.append(e2)

                if [abs(e_) for e_ in e1 if e_ != dtype(0)] != [
                    e_ for e_ in sorted(map(abs, e1), reverse=True) if e_ != dtype(0)
                ]:
                    assert not fast
                    # ensure non-overlapping property
                    e2 = fa.apmath.renormalize(ctx, e2, functional=functional, fast=fast)

                if functional:
                    assert len(e2) == len(e1)
                else:
                    assert len(e2) <= len(e1)

                for i in range(len(e2) - 1):
                    if functional:
                        assert (
                            not fa.utils.overlapping(e2[i], e2[i + 1])
                            or (e2[i] == dtype(0) and e2[i + 1] == dtype(0))
                            or (e2[i + 1] == dtype(0) and i == len(e2) - 2)
                        )
                    else:
                        assert not fa.utils.overlapping(e2[i], e2[i + 1])

                e1_mp = sum([fa.utils.float2mpf(mp_ctx, e) for e in e1], fa.utils.float2mpf(mp_ctx, dtype(0)))
                e2_mp = sum([fa.utils.float2mpf(mp_ctx, e) for e in e2], fa.utils.float2mpf(mp_ctx, dtype(0)))

                assert e1_mp == e2_mp

    if functional:
        x1, x2, x3 = map(numpy.array, zip(*samples3_lst))
        e1, e2, e3 = map(numpy.array, zip(*results3_lst))
        r1, r2, r3 = fa.apmath.renormalize(ctx, [x1, x2, x3], functional=functional, fast=fast)
        assert numpy.array_equal(r1, e1)
        assert numpy.array_equal(r2, e2)
        assert numpy.array_equal(r3, e3)


@pytest.mark.parametrize(
    "functional,overlapping",
    [
        ("non-functional", "non-overlapping"),
        ("non-functional", "overlapping"),
        ("functional", "non-overlapping"),
        ("functional", "overlapping"),
    ],
)
def test_multiply(dtype, functional, overlapping):
    """Consider a multiplication of a two 64-bit FP expansions, for instance, of

      e1 = [1e130, 1e-110]
      e2 = [1e120, 1e-150]

    The expected result of multiply(e1, e2) is [1e250, 1e-10, 1e-20, 1e-160]
    that represents a value of

      1e250 + 1e-10 + 1e-20 + 1e-160

    with precision approximately 1362 bits. That is, to verify a
    product of two FP expansions using mpmath mpf object, its
    precision must be at least 2097 bits (in the extreme case, the
    square of sqrt(largest) + sqrt(smallest_subnormal) is
    approximately 2 ** 1024 + 2 **-49 + 2 ** -1073).

    If the mpmath precision is less than the precision of an exact
    value, false-positive mismatches will be reported.

    An alternative to using mpmath mpf is using Fraction that
    arithmetics is always exact. However, not all resulting fractions
    can be converted to the used dtype (e.g. when numerator or
    denomerator is too large to be converted to Python
    float). Therefore, we still have to use mpmath in conversion from
    Fraction to float.

    Performance-wise, using Fraction is twice faster than using mpmath
    mpf.
    """
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    import mpmath

    use_mpf = False  # disabled by default as mpmath mpf is twice
    # slower than using Fraction
    use_Fraction = True

    length = 2
    # Renormalize inputs as expansion_samples may not be normalized
    # even when overlapping is False
    renormalize = False
    # Enabling overlapping allows more samples:
    overlapping = {"non-overlapping": False, "overlapping": True}[overlapping]
    # Functional means that the size of outputs from apmath functions is constant.
    functional = {"non-functional": False, "functional": True}[functional]

    fi = numpy.finfo(dtype)
    if use_mpf:
        max_prec = fi.maxexp - numpy.frexp(fi.smallest_subnormal)[1] + 20
    else:
        max_prec = 53

    size = 20
    ctx = fa.utils.NumpyContext(dtype)

    mp_ctx = mpmath.mp
    ulps_fraction = defaultdict(int)
    ulps_mp = defaultdict(int)
    samples = list(fa.utils.expansion_samples(size=size, length=length, overlapping=overlapping, dtype=dtype))
    samples1_lst = []
    samples2_lst = []
    results_lst = []
    with mp_ctx.workprec(max_prec):
        for e1 in samples:
            if renormalize:
                e1 = fa.apmath.renormalize(ctx, e1, functional=functional) or [dtype(0)]
            if use_mpf:
                e1_mp = fa.utils.expansion2mpf(mp_ctx, e1)
            for e2 in fa.utils.expansion_samples(size=size, length=length, overlapping=overlapping, dtype=dtype):
                if renormalize:
                    e2 = fa.apmath.renormalize(ctx, e2, functional=functional) or [dtype(0)]

                if functional:
                    e = fa.apmath.multiply(ctx, e1, e2, functional=functional, size=length + 1)
                    samples1_lst.append(e1)
                    samples2_lst.append(e2)
                    results_lst.append(e)
                else:
                    e = fa.apmath.multiply(ctx, e1, e2, functional=functional) or [dtype(0)]

                # skip samples that result overflow to infinity
                s = sum(e[:-1], e[-1])
                if not numpy.isfinite(s):
                    continue

                if use_mpf:
                    e2_mp = fa.utils.expansion2mpf(mp_ctx, e2)
                    result_mp = fa.utils.expansion2mpf(mp_ctx, e)
                    expected_mp = e1_mp * e2_mp
                    err_mp = result_mp - expected_mp
                    u_mp = fa.utils.diff_ulp(fa.utils.mpf2float(dtype, err_mp), dtype(0))
                    ulps_mp[u_mp] += 1

                if use_Fraction:
                    result_fraction = fa.utils.float2fraction(e)
                    expected_fraction = fa.utils.float2fraction(e1) * fa.utils.float2fraction(e2)
                    err_fraction = result_fraction - expected_fraction
                    err_fraction_mp = mp_ctx.mpf(err_fraction.numerator) / err_fraction.denominator
                    u_fraction = fa.utils.diff_ulp(fa.utils.mpf2float(dtype, err_fraction_mp), dtype(0))
                    ulps_fraction[u_fraction] += 1

                    if u_fraction > 2 and 0:
                        print(f"{e1=} {e2=} {e=}")
    fa.utils.show_ulp(ulps_fraction, title="using Fraction")
    if ulps_mp:
        fa.utils.show_ulp(ulps_mp, title="using mpmath mpf")

    if functional:
        x1, x2 = map(numpy.array, zip(*samples1_lst))
        y1, y2 = map(numpy.array, zip(*samples2_lst))
        xy1, xy2, xy3 = map(numpy.array, zip(*results_lst))

        r1, r2, r3 = fa.apmath.multiply(ctx, [x1, x2], [y1, y2], functional=functional, size=length + 1)
        assert numpy.array_equal(r1, xy1, equal_nan=True)
        assert numpy.array_equal(r2, xy2, equal_nan=True)
        assert numpy.array_equal(r3, xy3, equal_nan=True)


@pytest.mark.parametrize(
    "functional,overlapping",
    [
        ("non-functional", "non-overlapping"),
        ("non-functional", "overlapping"),
        ("functional", "non-overlapping"),
        ("functional", "overlapping"),
    ],
)
def test_square(dtype, functional, overlapping):
    overlapping = {"non-overlapping": False, "overlapping": True}[overlapping]
    functional = {"non-functional": False, "functional": True}[functional]
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    import mpmath

    fi = numpy.finfo(dtype)
    max_prec = -fi.negep * 10
    size = 1000
    length = 2
    renormalize = False
    ctx = fa.utils.NumpyContext(dtype)
    mp_ctx = mpmath.mp
    precs = defaultdict(int)
    samples = list(fa.utils.expansion_samples(size=size, length=length, overlapping=overlapping, dtype=dtype))
    results = []
    with mp_ctx.workprec(max_prec):
        for e1 in samples:
            e1_mp = fa.utils.expansion2mpf(mp_ctx, e1)
            if renormalize:
                e1 = fa.apmath.renormalize(ctx, e1, functional=functional) or [dtype(0)]

            if functional:
                e = fa.apmath.square(ctx, e1, functional=functional, size=length + 1)
                results.append(e)
            else:
                e = fa.apmath.square(ctx, e1, functional=functional) or [dtype(0)]

            # skip samples that result overflow to infinity
            s = sum(e[:-1], e[-1])
            if not numpy.isfinite(s):
                continue

            result_mp = fa.utils.expansion2mpf(mp_ctx, e)
            expected_mp = e1_mp * e1_mp
            err_mp = result_mp - expected_mp
            prec = fa.utils.diff_prec(result_mp, expected_mp)
            precs[prec] += 1
            if prec <= -fi.negep and 0:
                print(
                    f"{prec=} {e1=} {e=} {result_mp=} {expected_mp=} {fa.apmath.multiply(ctx, e1, e1, functional=functional)=}"
                )

            if not overlapping:
                if s < fi.smallest_normal:
                    min_prec = min(0, fa.utils.diff_prec(s, fi.smallest_normal)) - 2 - fi.negep
                else:
                    min_prec = -fi.negep
                assert prec >= min_prec

    fa.utils.show_prec(precs)

    if functional:
        x1, x2 = map(numpy.array, zip(*samples))
        xx1, xx2, xx3 = map(numpy.array, zip(*results))

        r1, r2, r3 = fa.apmath.square(ctx, [x1, x2], functional=functional, size=length + 1)
        assert numpy.array_equal(r1, xx1, equal_nan=True)
        assert numpy.array_equal(r2, xx2, equal_nan=True)
        assert numpy.array_equal(r3, xx3, equal_nan=True)


@pytest.mark.parametrize(
    "functional,overlapping",
    [
        ("non-functional", "non-overlapping"),
        ("non-functional", "overlapping"),
        ("functional", "non-overlapping"),
        ("functional", "overlapping"),
    ],
)
def test_reciprocal(dtype, functional, overlapping):
    overlapping = {"non-overlapping": False, "overlapping": True}[overlapping]
    functional = {"non-functional": False, "functional": True}[functional]
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")

    import mpmath

    fi = numpy.finfo(dtype)
    max_prec = -fi.negep * 10
    size = 1000
    length = 2
    renormalize = False
    ctx = fa.utils.NumpyContext(dtype)
    mp_ctx = mpmath.mp
    precs = defaultdict(int)
    with mp_ctx.workprec(max_prec):
        for e1 in fa.utils.expansion_samples(size=size, length=length, overlapping=overlapping, dtype=dtype):
            e1_mp = fa.utils.expansion2mpf(mp_ctx, e1)
            if renormalize:
                e1 = fa.apmath.renormalize(ctx, e1, functional=functional) or [dtype(0)]

            e = fa.apmath.reciprocal(ctx, e1, functional=functional) or [dtype(0)]

            # skip samples that result overflow to infinity
            s = sum(e[:-1], e[-1])
            if not numpy.isfinite(s):
                continue

            result_mp = fa.utils.expansion2mpf(mp_ctx, e)
            expected_mp = 1 / e1_mp
            err_mp = result_mp - expected_mp
            prec = fa.utils.diff_prec(result_mp, expected_mp)
            precs[prec] += 1
            if prec <= -fi.negep and 0:
                print(f"{prec=} {e1=} {e=} {result_mp=} {expected_mp=}")

            if not overlapping:
                assert prec > -fi.negep - 3

    fa.utils.show_prec(precs)


@pytest.mark.parametrize(
    "functional,overlapping",
    [
        ("non-functional", "non-overlapping"),
        ("non-functional", "overlapping"),
        ("functional", "non-overlapping"),
        ("functional", "overlapping"),
    ],
)
def test_sqrt(dtype, functional, overlapping):
    overlapping = {"non-overlapping": False, "overlapping": True}[overlapping]
    functional = {"non-functional": False, "functional": True}[functional]
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")

    import mpmath

    fi = numpy.finfo(dtype)
    max_prec = fi.maxexp - numpy.frexp(fi.smallest_subnormal)[1] + 20
    size = 500
    length = 2
    renormalize = False
    ctx = fa.utils.NumpyContext(dtype)
    mp_ctx = mpmath.mp
    precs = defaultdict(int)
    with mp_ctx.workprec(max_prec):
        for e1 in fa.utils.expansion_samples(size=size, length=length, min_value=1, overlapping=overlapping, dtype=dtype):
            e1_mp = fa.utils.expansion2mpf(mp_ctx, e1)
            if renormalize:
                e1 = fa.apmath.renormalize(ctx, e1, functional=functional) or [dtype(0)]

            e = fa.apmath.sqrt(ctx, e1, functional=functional) or [dtype(0)]

            # skip samples that result overflow to infinity
            s = sum(e[:-1], e[-1])
            if not numpy.isfinite(s):
                continue

            result_mp = fa.utils.expansion2mpf(mp_ctx, e)
            expected_mp = mp_ctx.sqrt(e1_mp)
            err_mp = result_mp - expected_mp
            prec = fa.utils.diff_prec(result_mp, expected_mp)
            precs[prec] += 1
            if prec <= -fi.negep and 1:
                print(f"{prec=} {e1=} {e=} {result_mp=} {expected_mp=}")

            if not overlapping:
                assert prec > -fi.negep - 2, (e1, e, expected_mp)

    fa.utils.show_prec(precs)


@pytest.mark.parametrize(
    "functional,overlapping",
    [
        ("non-functional", "non-overlapping"),
        ("non-functional", "overlapping"),
        ("functional", "non-overlapping"),
        ("functional", "overlapping"),
    ],
)
def test_add(dtype, functional, overlapping):
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    import mpmath

    length = 2
    # Renormalize inputs as expansion_samples may not be normalized
    # even when overlapping is False
    renormalize = False
    # Enabling overlapping allows more samples:
    overlapping = {"non-overlapping": False, "overlapping": True}[overlapping]
    # Functional means that the size of outputs from apmath functions is constant.
    functional = {"non-functional": False, "functional": True}[functional]

    fi = numpy.finfo(dtype)
    max_prec = fi.maxexp - numpy.frexp(fi.smallest_subnormal)[1] + 20

    size = 50
    ctx = fa.utils.NumpyContext(dtype)

    mp_ctx = mpmath.mp
    precs = defaultdict(int)
    samples = fa.utils.expansion_samples(size=size, length=length, overlapping=overlapping, dtype=dtype)
    with mp_ctx.workprec(max_prec):
        for e1 in samples:
            if renormalize:
                e1 = fa.apmath.renormalize(ctx, e1, functional=functional) or [dtype(0)]
            e1_mp = fa.utils.expansion2mpf(mp_ctx, e1)
            for e2 in fa.utils.expansion_samples(size=size, length=length, overlapping=overlapping, dtype=dtype):
                if renormalize:
                    e2 = fa.apmath.renormalize(ctx, e2, functional=functional) or [dtype(0)]

                e = fa.apmath.add(ctx, e1, e2, functional=functional, size=length + 1) or [dtype(0)]

                # skip samples that result overflow to infinity
                s = sum(e[:-1], e[-1])
                if not numpy.isfinite(s):
                    continue

                result_mp = fa.utils.expansion2mpf(mp_ctx, e)
                e2_mp = fa.utils.expansion2mpf(mp_ctx, e2)
                expected_mp = e1_mp + e2_mp
                prec = fa.utils.diff_prec(result_mp, expected_mp)
                precs[prec] += 1

                assert prec > (-fi.negep) * length

    fa.utils.show_prec(precs)


@pytest.mark.parametrize(
    "functional,overlapping",
    [
        # ("non-functional", "non-overlapping"),
        # ("non-functional", "overlapping"),
        ("functional", "non-overlapping"),
        # ("functional", "overlapping"),
    ],
)
def test_exponential(dtype, functional, overlapping):
    overlapping = {"non-overlapping": False, "overlapping": True}[overlapping]
    functional = {"non-functional": False, "functional": True}[functional]
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")

    import mpmath

    fi = numpy.finfo(dtype)
    max_prec = fi.maxexp - numpy.frexp(fi.smallest_subnormal)[1] + 20
    size = 500000
    length = 2
    renormalize = False
    ctx = fa.utils.NumpyContext(dtype)
    mp_ctx = mpmath.mp
    precs = defaultdict(int)
    with mp_ctx.workprec(max_prec):
        for e1 in fa.utils.expansion_samples(size=size, length=length, min_value=1, overlapping=overlapping, dtype=dtype):
            e1_mp = fa.utils.expansion2mpf(mp_ctx, e1)
            if renormalize:
                e1 = fa.apmath.renormalize(ctx, e1, functional=functional) or [dtype(0)]

            e = fa.apmath.exponential(ctx, e1, functional=functional)

            # skip samples that result overflow to infinity
            s = sum(e[:-1], e[-1])
            if not numpy.isfinite(s):
                continue

            result_mp = fa.utils.expansion2mpf(mp_ctx, e)
            expected_mp = mp_ctx.exp(e1_mp)

            prec = fa.utils.matching_bits(result_mp, expected_mp)
            precs[prec] += 1

    fa.utils.show_prec(precs)


@pytest.mark.parametrize(
    "function,functional",
    [
        ("exp", "non-functional"),
        ("exp", "functional"),
        ("cos", "non-functional"),
        ("j0", "non-functional"),
        ("j1", "non-functional"),
    ],
)
def test_hypergeometric(dtype, function, functional):
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    import mpmath
    import fractions

    size = 100
    functional = {"non-functional": False, "functional": True}[functional]
    ctx = fa.utils.NumpyContext(dtype)
    mp_ctx = mpmath.mp
    fi = numpy.finfo(dtype)
    min_prec = -fi.negep
    max_prec = fi.maxexp - numpy.frexp(fi.smallest_subnormal)[1] + 20

    min_value = 1
    max_value = 20
    niter = 20
    length = 2
    if function == "exp":
        a, b = [], []  # exp(z)
        sample_func = lambda z: z
        length = 2
        min_value = 0
        max_value = 9
        niter = {
            numpy.float16: 22,  # max z == 9
            numpy.float32: 30,  # max z == 9
            numpy.float64: 45,  # max z == 9
        }.get(dtype, niter)
    elif function == "cos":
        a, b = [], [fractions.Fraction(1, 2)]  # cos(-4 * sqrt(z))
        sample_func = lambda z: -z
        min_value = 0
        max_value = 10
        length = 2
        niter = {
            numpy.float16: 11,  # max z == 5.516
            numpy.float32: 28,  # max z == 49.94
            numpy.float64: 66,  # max z == 326.0
        }.get(dtype, niter)
        niter = {
            numpy.float16: 14,  # max z == 15
            numpy.float32: 19,  # max z == 22
            numpy.float64: 25,  # max z == 21
        }.get(dtype, niter)
        niter = {
            numpy.float16: 14,  # max z == 10
            numpy.float32: 15,  # max z == 10
            numpy.float64: 21,  # max z == 11
        }.get(dtype, niter)
    elif function == "j0":
        a, b = [], [1]  # J0(-4 * sqrt(z)) * 2 / z
        sample_func = lambda z: -z
        min_value = 0
        max_value = 10
        length = 2
        niter = {
            numpy.float16: 13,  # max z == 10
            numpy.float32: 15,  # max z == 10
            numpy.float64: 21,  # max z == 11
        }.get(dtype, niter)
    elif function == "j1":
        a, b = [], [2]  # J1(-4 * sqrt(z)) * (2 / z) ** 2 * 2
        sample_func = lambda z: -z
        min_value = 0
        max_value = 10
        length = 3
        niter = {
            # numpy.float16: 13,  # max z == 3.68 [length==2]
            numpy.float16: 11,  # max z == 10 [length==3]
            numpy.float32: 14,  # max z == 10 [length==2 or 3]
            numpy.float64: 20,  # max z == 11
        }.get(dtype, niter)
    else:
        assert 0, function  # not implemented

    with mp_ctx.workprec(max_prec):
        for z in map(sample_func, fa.utils.real_samples(size=size, dtype=dtype, min_value=min_value, max_value=max_value)):
            e = fa.apmath.hypergeometric(ctx, a, b, [z], niter=niter, functional=functional, size=length)

            # skip samples that result overflow to infinity
            s = sum(e[:-1], e[-1])
            if not numpy.isfinite(s):
                print(f"{z=} {e=}")
                break

            result_mp = fa.utils.expansion2mpf(mp_ctx, e)

            z_mp = fa.utils.float2mpf(mp_ctx, z)

            expected_mp = mp_ctx.hyper(a, b, z_mp)

            prec = fa.utils.diff_prec(result_mp, expected_mp)

            # print(f'{prec=} {z=} {e=}')

            if size <= 1000:
                assert prec > min_prec

            if prec <= min_prec:
                break


def test_two_over_pi(dtype):
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    import mpmath

    fi = numpy.finfo(dtype)
    ctx = fa.utils.NumpyContext(dtype)
    mp_ctx = mpmath.mp

    # max size is
    # float16: 2
    # float32: 6
    # float64: 20
    for size in range(1, 22):
        e = fa.apmath.two_over_pi(ctx, dtype, size=size)
        if e[-1] == 0:
            break
        max_prec = -fi.negep * size
        with mp_ctx.workprec(max_prec):
            result = fa.utils.expansion2mpf(mp_ctx, e)
            expected = 2 / mp_ctx.pi

            with mp_ctx.workprec(max_prec - min(size, 3)):
                # adding 0 normalizes
                assert result + 0 == expected + 0

    assert size == 1 + {numpy.float16: 2, numpy.float32: 6, numpy.float64: 20}[dtype], (dtype, size)


def test_mul_two_over_pi(dtype):
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    import mpmath

    fi = numpy.finfo(dtype)
    ctx = fa.utils.NumpyContext(dtype)
    mp_ctx = mpmath.mp

    size = 2
    single_prec = -fi.negep
    max_prec = single_prec * size
    min_prec = max_prec * 2

    if dtype == numpy.float16:
        # min prec: 13
        max_prec += 8
        base = 1  # 256, min prec: 0
        base = 2  # min prec: 7.5
        base = 4  # min prec: 12
        base = 8  # 16, min prec: 13.1
        base = 64  # 128, min prec: 13.3
        base = 32  # min prec: 13.3
        scale = base < 128
        e = fa.apmath.two_over_pi(ctx, dtype, size=size + 1, base=base)
    elif dtype == numpy.float32:
        max_prec += 200
        base = 1  # min prec: 27.1
        base = 2  # min prec: 42.8
        base = 4  # min prec: 50
        base = 8192  # min prec: 50
        base = 32  # min prec: 48.4
        scale = base < 8192
        e = fa.apmath.two_over_pi(ctx, dtype, size=size + 5, base=base)
    elif dtype == numpy.float64:
        max_prec += 1200
        base = 1  # min prec: 105.8
        base = 2  # min prec: 105.8
        base = 268435456  # min prec: 105.8
        base = 32  # min prec: 105.8, 99.9(size=10 000 000)
        scale = base < 268435456
        e = fa.apmath.two_over_pi(ctx, dtype, size=size + 18, base=base)

    st = set()
    mx_r = -3
    mn_r = 3
    with mp_ctx.workprec(max_prec):
        for x in fa.utils.real_samples(size=1000, dtype=dtype, min_value=fi.smallest_normal**0.5, include_infinity=False):
            expected = fa.utils.number2mpf(mp_ctx, x) * 2 / mp_ctx.pi
            expected_mod4 = expected - mp_ctx.floor(expected / 4) * 4
            assert expected_mod4 >= 0
            if expected_mod4 > 2:
                expected_mod4 -= 4
            expected_mod4_p4 = expected_mod4 + 4

            result_mod4 = fa.apmath.multiply_mod4(ctx, [x], e, size=size, functional=True, base=base, scale=scale) or [
                dtype(0)
            ]
            result_mod4_mp = fa.utils.expansion2mpf(mp_ctx, result_mod4)

            # k = ctx.trunc(result_mod4[0])  # -1, 0, 1, 2, abs(r) < 1
            k = ctx.round(result_mod4[0])  # -2, -1, 0, 1, 2, abs(r) <= 0.5005
            r = fa.apmath.add(ctx, result_mod4, [-k])
            r = sum(r[:-1], r[-1])
            if r > mx_r:
                mx_r = r
            if r < mn_r:
                mn_r = r
            if k not in st:
                st.add(k)
            assert abs(r) <= 0.5005

            if k == -2:
                assert r > 0

            if k == 2:
                assert r < 0.0005

            prec_mod4 = fa.utils.matching_bits(result_mod4_mp, expected_mod4)
            if prec_mod4 == -1:
                prec_mod4 = fa.utils.matching_bits(result_mod4_mp, expected_mod4_p4)

            if prec_mod4 < min_prec:
                min_prec = prec_mod4

            assert prec_mod4 >= single_prec

    assert st == {-2, -1, 0, 1, 2}
    assert mn_r >= -0.5005
    assert mx_r <= 0.5005
    print(f"{min_prec=}")


def test_argument_reduction(dtype):
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    import mpmath

    fi = numpy.finfo(dtype)
    ctx = fa.utils.NumpyContext(dtype)
    mp_ctx = mpmath.mp

    size = 2
    single_prec = -fi.negep

    base = 32
    if dtype == numpy.float16:
        max_prec = 50
    elif dtype == numpy.float32:
        max_prec = 250
    elif dtype == numpy.float64:
        max_prec = 1350
    else:
        assert 0  # unreachable

    with mp_ctx.workprec(max_prec):
        for x in fa.utils.real_samples(size=100, dtype=dtype, include_infinity=False):
            expected_sn = mp_ctx.sin(fa.utils.number2mpf(mp_ctx, x))
            expected_cs = mp_ctx.cos(fa.utils.number2mpf(mp_ctx, x))

            k, rseq = fa.apmath.argument_reduction(ctx, [x], size=size, functional=True, base=base, scale=True)

            if abs(x) <= 0.7 and abs(x) >= fi.smallest_normal * 8:
                assert k == 0
                assert x == sum(rseq[:-1], rseq[-1])

            if k == 0:
                result_sn = mp_ctx.sin(fa.utils.expansion2mpf(mp_ctx, rseq))
            elif k == 1:
                result_sn = mp_ctx.cos(fa.utils.expansion2mpf(mp_ctx, rseq))
            elif k == 2:
                result_sn = -mp_ctx.sin(fa.utils.expansion2mpf(mp_ctx, rseq))
            elif k == -1:
                result_sn = -mp_ctx.cos(fa.utils.expansion2mpf(mp_ctx, rseq))
            else:
                assert 0, k  # unreachable
            prec_sn = fa.utils.matching_bits(result_sn, expected_sn)

            u = fa.utils.diff_ulp(fa.utils.mpf2float(dtype, result_sn), fa.utils.mpf2float(dtype, expected_sn))

            assert prec_sn >= single_prec or u <= 1


def test_sine_cosine(dtype):
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    import mpmath

    fi = numpy.finfo(dtype)
    ctx = fa.utils.NumpyContext(dtype)
    mp_ctx = mpmath.mp

    size = 2
    single_prec = -fi.negep

    base = 32
    if dtype == numpy.float16:
        max_prec = 50
    elif dtype == numpy.float32:
        max_prec = 250
    elif dtype == numpy.float64:
        max_prec = 1350
    else:
        assert 0  # unreachable

    samples = numpy.array(fa.utils.real_samples(size=61441 and 1000, dtype=dtype, include_infinity=False))
    ulps2 = defaultdict(int)
    ulps2cs = defaultdict(int)
    precs2 = defaultdict(int)
    precs2cs = defaultdict(int)
    sn_results = []
    cs_results = []
    with mp_ctx.workprec(max_prec):
        for x in samples:
            expected_sn = mp_ctx.sin(fa.utils.number2mpf(mp_ctx, x))
            expected_cs = mp_ctx.cos(fa.utils.number2mpf(mp_ctx, x))

            sn, cs = fa.apmath.sine_cosine(ctx, [x], size=size, functional=True)
            sn_results.append(sn)
            cs_results.append(cs)

            sn = fa.utils.expansion2mpf(mp_ctx, sn)

            prec_sn2 = fa.utils.matching_bits(sn, expected_sn)
            precs2[prec_sn2] += 1

            u2 = fa.utils.diff_ulp(fa.utils.mpf2float(dtype, sn), fa.utils.mpf2float(dtype, expected_sn))
            ulps2[u2] += 1

            assert prec_sn2 >= single_prec or u2 <= 1

            cs = fa.utils.expansion2mpf(mp_ctx, cs)

            prec_cs2 = fa.utils.matching_bits(cs, expected_cs)
            precs2cs[prec_cs2] += 1

            u2 = fa.utils.diff_ulp(fa.utils.mpf2float(dtype, cs), fa.utils.mpf2float(dtype, expected_cs))
            ulps2cs[u2] += 1

            assert prec_cs2 >= single_prec or u2 <= 1 or (u2 <= 2 and abs(cs) < fi.smallest_normal * 2), (x, cs)

    fa.utils.show_ulp(ulps2, title="apmath.sine")
    fa.utils.show_prec(precs2, title="apmath.sine")

    fa.utils.show_ulp(ulps2cs, title="apmath.cosine")
    fa.utils.show_prec(precs2cs, title="apmath.cosine")

    sn, cs = fa.apmath.sine_cosine(ctx, [samples], size=size, functional=True)
    for r, e in zip(sn, map(numpy.array, zip(*sn_results))):
        assert numpy.array_equal(r, e)
    for r, e in zip(cs, map(numpy.array, zip(*cs_results))):
        assert numpy.array_equal(r, e)


def test_log_of_two(dtype):
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    import mpmath

    fi = numpy.finfo(dtype)
    ctx = fa.utils.NumpyContext(dtype)
    mp_ctx = mpmath.mp

    # max size is
    # float16: 2
    # float32: 6
    # float64: 20
    for size in range(1, 22):
        e = fa.apmath.log_of_two(ctx, dtype, size=size)
        if e[-1] == 0:
            break
        max_prec = -fi.negep * size
        with mp_ctx.workprec(max_prec):
            result = fa.utils.expansion2mpf(mp_ctx, e)
            expected = mp_ctx.log(2)

            with mp_ctx.workprec(max_prec):
                # adding 0 normalizes
                assert result + 0 == expected + 0

    assert size == 1 + {numpy.float16: 2, numpy.float32: 6, numpy.float64: 20}[dtype], (dtype, size)


def test_reciprocal_log_of_two(dtype):
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    import mpmath

    fi = numpy.finfo(dtype)
    ctx = fa.utils.NumpyContext(dtype)
    mp_ctx = mpmath.mp

    # max size is
    # float16: 3
    # float32: 6
    # float64: 20
    for size in range(1, 22):
        e = fa.apmath.reciprocal_log_of_two(ctx, dtype, size=size)
        if e[-1] == 0:
            break
        max_prec = -fi.negep * size
        with mp_ctx.workprec(max_prec):
            result = fa.utils.expansion2mpf(mp_ctx, e)
            expected = 1 / mp_ctx.log(2)

            with mp_ctx.workprec(max_prec - 8):
                # adding 0 normalizes
                assert result + 0 == expected + 0

    assert size == 1 + {numpy.float16: 3, numpy.float32: 6, numpy.float64: 20}[dtype], (dtype, size)


def test_argument_reduction_exponential(dtype):
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    import mpmath

    fi = numpy.finfo(dtype)
    ctx = fa.utils.NumpyContext(dtype)
    mp_ctx = mpmath.mp

    size = 2
    single_prec = -fi.negep

    base = None
    max_prec = 2**fi.nexp

    precs = defaultdict(int)
    samples = numpy.array(
        fa.utils.real_samples(size=1000, dtype=dtype, max_value=2 ** (-fi.negep) * 0.6, include_infinity=False)
    )
    k_lst = []
    rseq_lst = []

    with mp_ctx.workprec(max_prec):
        largest_log2 = fa.utils.mpf2float(dtype, mp_ctx.log(2) * fa.utils.float2mpf(mp_ctx, fi.max))
        smallest_integer_log2 = fa.utils.mpf2float(dtype, mp_ctx.log(2) * 2 ** (single_prec - 1))
        for x in samples:
            expected_mp = fa.utils.float2mpf(mp_ctx, x)
            k, rseq = fa.apmath.argument_reduction_exponential(
                ctx, [x], size=size, functional=True, base=base, scale=not False
            )
            k_lst.append(k)
            rseq_lst.append(rseq)
            result_mp = k * mp_ctx.log(2) + fa.utils.expansion2mpf(mp_ctx, rseq)
            prec = fa.utils.matching_bits(result_mp, expected_mp)

            precs[prec] += 1
            assert k == int(k), k
            assert abs(sum(rseq[:-1], rseq[-1])) < numpy.log(2) * 0.75
            if abs(x) <= smallest_integer_log2:
                assert prec >= single_prec * size, (x, k, rseq)

    fa.utils.show_prec(precs)

    k, rseq = fa.apmath.argument_reduction_exponential(ctx, [samples], size=size, functional=True, base=base, scale=False)
    for r, e in zip(rseq, map(numpy.array, zip(*rseq_lst))):
        assert numpy.array_equal(r, e)
    assert numpy.array_equal(k, k_lst)


def test_exponential(dtype):
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    import mpmath

    fi = numpy.finfo(dtype)
    mp_ctx = mpmath.mp

    size = 2
    max_prec = 2**fi.nexp * 2
    precs = defaultdict(int)
    ulps = defaultdict(int)
    samples = numpy.array(fa.utils.real_samples(size=100, dtype=dtype, include_infinity=False))
    results = []
    with mp_ctx.workprec(max_prec):
        ctx = fa.utils.NumpyContext(dtype, mp_ctx)
        result = fa.apmath.exponential(ctx, [samples], size=size, functional=True)
        result = numpy.array(result).T

        for i, x in enumerate(samples):
            expected = fa.apmath.exponential.mp(ctx, [x], size=size, functional=True)

            result_mp = fa.utils.expansion2mpf(mp_ctx, result[i])
            expected_mp = fa.utils.expansion2mpf(mp_ctx, expected)

            prec = fa.utils.matching_bits(result_mp, expected_mp)
            u_mp = fa.utils.diff_ulp(sum(result[i][:-1], result[i][-1]), fa.utils.mpf2float(dtype, expected_mp))
            ulps[u_mp] += 1
            precs[prec] += 1

            if u_mp and 0:
                print(f"{x=} {result[i]=} {expected=} {u_mp=} {prec=} {i=}")

            assert u_mp <= 1

    fa.utils.show_prec(precs)
    fa.utils.show_ulp(ulps)
    return
    result = fa.apmath.exponential(ctx, [samples], size=size, functional=True)
    for r, e in zip(result, map(numpy.array, zip(*results))):
        assert numpy.array_equal(r, e)


def test_logarithm(dtype):
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    import mpmath

    fi = numpy.finfo(dtype)
    ctx = fa.utils.NumpyContext(dtype)
    mp_ctx = mpmath.mp

    size = 2
    max_prec = 2**fi.nexp * 2
    precs = defaultdict(int)
    ulps = defaultdict(int)
    samples = numpy.array(fa.utils.real_samples(size=100, dtype=dtype, min_value=fi.smallest_normal, include_infinity=False))
    results = []
    with mp_ctx.workprec(max_prec):
        result = fa.apmath.logarithm(ctx, [samples], size=size, functional=True)
        result = numpy.array(result).T

        for i, x in enumerate(samples):
            result_mp = fa.utils.expansion2mpf(mp_ctx, result[i])

            expected_mp = fa.apmath.logarithm(ctx, [x], size=size, functional=True, mp_ctx=mp_ctx, asmp=True)

            prec = fa.utils.matching_bits(result_mp, expected_mp)
            u_mp = fa.utils.diff_ulp(sum(result[i][:-1], result[i][-1]), fa.utils.mpf2float(dtype, expected_mp))
            ulps[u_mp] += 1
            precs[prec] += 1

            if u_mp:
                print(f"{x=} {result[i]=} {expected_mp=} {u_mp=} {prec=}")

            assert u_mp <= 1

    fa.utils.show_prec(precs)
    fa.utils.show_ulp(ulps)

    result = fa.apmath.logarithm(ctx, [samples], size=size, functional=True)
    for r, e in zip(result, map(numpy.array, zip(*results))):
        assert numpy.array_equal(r, e)


def test_hypergeometric0f1_taylor(dtype):
    if dtype == numpy.longdouble:
        pytest.skip(f"test not implemented")
    import mpmath

    fi = numpy.finfo(dtype)
    ctx = fa.utils.NumpyContext(dtype)
    mp_ctx = mpmath.mp

    b = 1
    size = 2
    max_prec = 2**fi.nexp * 2
    precs = defaultdict(int)
    ulps = defaultdict(int)
    max_value = {numpy.float16: 25, numpy.float32: 74, numpy.float64: 320}[dtype]
    samples = -numpy.array(
        fa.utils.real_samples(size=10000, dtype=dtype, min_value=0, max_value=max_value, include_infinity=False)
    )
    results = []
    with mp_ctx.workprec(max_prec):
        result = fa.apmath.hypergeometric0f1_taylor(ctx, b, [samples], size=size, functional=True)
        result = numpy.array(result).T

        for i, x in enumerate(samples):
            expected_mp = fa.apmath.hypergeometric0f1_taylor(ctx, b, [x], size=size, functional=True, mp_ctx=mp_ctx, asmp=True)
            result_mp = fa.utils.expansion2mpf(mp_ctx, result[i])

            prec = fa.utils.matching_bits(result_mp, expected_mp)
            u_mp = fa.utils.diff_ulp(sum(result[i][:-1], result[i][-1]), fa.utils.mpf2float(dtype, expected_mp))
            ulps[u_mp] += 1
            precs[prec] += 1

            if u_mp > 1:
                print(f"{x=} {result[i]=} {expected_mp=} {u_mp=} {prec=}")

            if dtype in {numpy.float32, numpy.float64}:
                assert u_mp <= 1
            else:
                assert u_mp <= 15

    fa.utils.show_prec(precs)
    fa.utils.show_ulp(ulps)


def test_nztopk():
    import itertools

    dtype = numpy.float16

    ctx = fa.utils.NumpyContext(dtype)

    def make_samples(n):
        for s in itertools.product(numpy.arange(2, dtype=dtype), repeat=n):
            r = numpy.random.randn(n).astype(dtype)
            yield [r[i] * s[i] for i in range(n)]

    for size in [1, 2, 4, 6]:
        expected_lst = defaultdict(list)
        samples_lst = []
        for seq in make_samples(size):
            assert len(seq) == size
            samples_lst.append(seq)
            for k in range(1, size + 1):
                expected = [s for s in seq if s != 0][:k]
                expected += [dtype(0)] * (k - len(expected))
                expected_lst[k].append(expected)

                result = fa.apmath.nztopk(ctx, seq, k)
                assert result == expected

        samples = list(numpy.array(list(zip(*samples_lst)), dtype=dtype))
        for k in range(1, size + 1):
            result = fa.apmath.nztopk(ctx, samples, k)
            assert (numpy.array(result) == numpy.array(expected_lst[k]).T).all()


def test_renormalize_max_size(dtype):

    fi = numpy.finfo(dtype)
    max_size = (fi.maxexp - fi.minexp - fi.machep) // (-fi.negep - 1)

    ctx = fa.utils.NumpyContext(dtype)
    x = [fi.max, fi.max * dtype(fi.eps * 1e-1)]
    while x[-1] != 0 and x[-1] != x[-2]:
        x.append(x[-1] * dtype(fi.eps))

    y = fa.apmath.renormalize(ctx, x)
    assert len(y) <= max_size
    print(f"{len(y), max_size=}")
