import numpy
import pytest
import warnings
import functional_algorithms as fa


@pytest.fixture(scope="function", params=[numpy.float16, numpy.float32, numpy.float64, numpy.float128])
def dtype(request):
    return request.param


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
