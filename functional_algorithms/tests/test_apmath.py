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
    ctx = fa.utils.NumpyContext()
    fi = numpy.finfo(dtype)
    c = numpy.ldexp(dtype(1), fi.negep + 6)
    mp_ctx = mpmath.mp
    with mp_ctx.workprec(max_prec):
        for x1 in fa.utils.real_samples(size, dtype=dtype, include_infinity=False, include_huge=False):
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
    ctx = fa.utils.NumpyContext()

    mp_ctx = mpmath.mp
    ulps_fraction = defaultdict(int)
    ulps_mp = defaultdict(int)
    with mp_ctx.workprec(max_prec):
        for e1 in fa.utils.expansion_samples(size=size, length=length, overlapping=overlapping, dtype=dtype):
            if renormalize:
                e1 = fa.apmath.renormalize(ctx, e1, functional=functional) or [dtype(0)]
            if use_mpf:
                e1_mp = fa.utils.expansion2mpf(mp_ctx, e1)
            for e2 in fa.utils.expansion_samples(size=size, length=length, overlapping=overlapping, dtype=dtype):
                if renormalize:
                    e2 = fa.apmath.renormalize(ctx, e2, functional=functional) or [dtype(0)]

                e = fa.apmath.multiply(ctx, e1, e2, functional=functional) or [dtype(0)]

                # e = fa.apmath.mergesort(ctx, e, mth="a>a")[:length+1]

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

                    if u_fraction > 2:
                        print(f"{e1=} {e2=} {e=}")
    fa.utils.show_ulp(ulps_fraction, title="using Fraction")
    fa.utils.show_ulp(ulps_mp, title="using mpmath mpf")


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
    ctx = fa.utils.NumpyContext()
    mp_ctx = mpmath.mp
    precs = defaultdict(int)
    with mp_ctx.workprec(max_prec):
        for e1 in fa.utils.expansion_samples(size=size, length=length, overlapping=overlapping, dtype=dtype):
            e1_mp = fa.utils.expansion2mpf(mp_ctx, e1)
            if renormalize:
                e1 = fa.apmath.renormalize(ctx, e1, functional=functional) or [dtype(0)]

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
    ctx = fa.utils.NumpyContext()
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
    ctx = fa.utils.NumpyContext()
    mp_ctx = mpmath.mp
    precs = defaultdict(int)
    with mp_ctx.workprec(max_prec):
        for e1 in fa.utils.expansion_samples(size=size, length=length, overlapping=overlapping, dtype=dtype):
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
            if prec <= -fi.negep and 0:
                print(f"{prec=} {e1=} {e=} {result_mp=} {expected_mp=}")

            if not overlapping:
                assert prec > -fi.negep - 2

    fa.utils.show_prec(precs)
