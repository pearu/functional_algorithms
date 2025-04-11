import numpy
import pytest
import warnings
from collections import defaultdict
import functional_algorithms as fa
import functional_algorithms.floating_point_algorithms as fpa
import functional_algorithms.generalized_hypergeometric_functions as ghf
import fractions


@pytest.fixture(scope="function", params=[numpy.float16, numpy.float32, numpy.float64])
def dtype(request):
    return request.param


@pytest.mark.parametrize(
    "transform",
    ["drummond", "levin"],
)
def test_reference(transform):
    dtype = numpy.float64

    ctx = fa.utils.NumpyContext()

    samples = list(fa.utils.real_samples(size=4, dtype=dtype, min_value=0.1, max_value=100))

    for alpha, beta in [
        ([], []),
        ([], [fractions.Fraction(1, 2)]),
        ([1, 2], [3]),
        ([1, 2], [3, 4, 5]),
    ]:
        ref = ghf.Reference(alpha, beta, transform)

        for n in range(4):
            for k in range(6):
                for z in samples:
                    assert numpy.isclose(ref.N_ref(z, n, k), ref.Nsplit(z, n, k))
                    assert numpy.isclose(ref.Nm1(z, n, k), ref.Nm1_ref(z, n, k))
                    assert numpy.isclose(ref.D(z, n, k), ref.D_ref(z, n, k))
                    assert numpy.isclose(ref.T(z, n, k), ref.T_ref(z, n, k))
                    result = fpa.fast_polynomial(ctx, numpy.reciprocal(z), ref.Nm1_poly(n, k), reverse=False) / z
                    expected = dtype(ref.Nm1_ref(z, n, k))
                    assert numpy.isclose(result, expected)
                    result = fpa.fast_polynomial(ctx, numpy.reciprocal(z), ref.D_poly(n, k), reverse=False) / z ** (n + 1)
                    expected = ref.D_ref(z, n, k)
                    assert numpy.isclose(result, expected)

                    if n == 0:
                        assert numpy.isclose(ref.s(z, n - 1), 0)
                        assert numpy.isclose(ref.s(z, n), 1)


@pytest.mark.parametrize(
    "transform",
    ["drummond", "levin", "taylor"],
)
def test_pFq_coeffs(dtype, transform):

    if transform == "drummond":
        pFq_coeffs = ghf.pFq_drummond_coeffs
        params = dict()
    elif transform == "levin":
        pFq_coeffs = ghf.pFq_levin_coeffs
        params = dict(gamma=3)
    elif transform == "taylor":
        pFq_coeffs = ghf.pFq_taylor_coeffs
        params = dict()
    else:
        assert 0  # not impl

    for alpha, beta in [
        ([], []),
        ([], [fractions.Fraction(1, 2)]),
        ([fractions.Fraction(1, 2)], []),
        ([1, 2], [3]),
        ([1, 2], [3, 4, 5]),
    ]:
        ref = ghf.Reference(alpha, beta, transform, **params)

        if transform == "taylor":
            for k in range(6):
                T = pFq_coeffs(alpha, beta, k)
                for i, t in enumerate(T):
                    assert t == ref.A(i)
            continue

        for n in range(4):
            for k in range(6):
                Nm1_ref = ref.Nm1_poly(n, k)
                D_ref = ref.D_poly(n, k)

                Nm1, D = pFq_coeffs(alpha, beta, k, n=n, **params)
                assert D[-1] == 1
                assert [c * D_ref[-1] for c in Nm1] == Nm1_ref
                assert [c * D_ref[-1] for c in D] == D_ref

                for i in [0, (k + 1) // 2]:
                    Nm1, D = pFq_coeffs(alpha, beta, k, n=n, normalization_index=i, **params)
                    assert [c * D_ref[i] for c in Nm1] == Nm1_ref
                    assert [c * D_ref[i] for c in D] == D_ref


@pytest.mark.parametrize(
    "transform",
    ["drummond", "levin"],
)
def test_pFq_eval(transform):
    dtype = numpy.float64
    if transform == "drummond":
        params = dict(transform=transform)
    elif transform == "levin":
        params = dict(transform=transform, gamma=3)
    else:
        assert 0  # not impl
    pFq_eval = ghf.pFq_impl
    pFqm1_eval = ghf.pFq_minus_one_impl
    rtol = {numpy.float16: 1e-3, numpy.float32: 1e-4, numpy.float64: 1e-5}[dtype]
    atol = {numpy.float16: 1e-1, numpy.float32: 1e-5, numpy.float64: 1e-8}[dtype]

    ctx = fa.utils.NumpyContext()
    size = 10
    samples = list(fa.utils.real_samples(size=size, dtype=dtype, min_value=0.1, max_value=100))

    for alpha, beta in [
        ([], []),
        ([], [fractions.Fraction(1, 2)]),
        ([fractions.Fraction(1, 2)], []),
        ([], [3]),
        ([1, 2], [3, 4, 5]),
    ]:
        ref = ghf.Reference(alpha, beta, **params)
        for k in range(3, 4):
            for z in samples:
                z = dtype(1)
                expected = ref.T(z, 1, k)
                expected_m1 = ref.Tm1(z, 1, k)
                assert numpy.isclose(expected_m1, expected - 1)
                for m in range(k + 1):
                    for normalization_index in range(k + 1):
                        result_m1 = pFqm1_eval(
                            ctx, dtype, alpha, beta, z, k=k, m=m, normalization_index=normalization_index, **params
                        )
                        assert numpy.isclose(result_m1, expected_m1, rtol=rtol, atol=atol), (k, z, alpha, beta)

                        result = pFq_eval(
                            ctx, dtype, alpha, beta, z, k=k, m=m, normalization_index=normalization_index, **params
                        )
                        assert numpy.isclose(result, expected, rtol=rtol, atol=atol), (k, z, alpha, beta)


@pytest.mark.parametrize(
    "transform",
    ["drummond", "levin", "taylor"],
)
def test_pFq_small(dtype, transform):
    import mpmath

    fi = numpy.finfo(dtype)

    min_value = fi.smallest_normal
    max_value = 0.1
    if transform == "drummond":
        params = dict(transform=transform)
    elif transform == "levin":
        params = dict(transform=transform, gamma=2)
    elif transform == "taylor":
        params = dict(transform=transform)
    else:
        assert 0  # not impl
    pFqm1 = ghf.pFq_minus_one_impl
    ctx = fa.utils.NumpyContext()
    size = 100
    samples = list(fa.utils.real_samples(size=size, dtype=dtype, min_value=min_value, max_value=max_value))
    mp_ctx = mpmath.mp
    max_prec = {numpy.float16: 4 * 11, numpy.float32: 8 * 24, numpy.float64: 22 * 53}[dtype]

    for alpha, beta in [
        ([], [1]),  # J0
        ([], [2]),  # J1
    ]:
        # J0, most accurate combinations:
        # Using horner scheme, float64, levin:
        # Range: 0.1...10
        # alpha=[], beta=[1], k=16 m=0 i=11 w=0.78
        # alpha=[], beta=[1], k=14 m=0 i=4 w=0.685
        # alpha=[], beta=[1], k=12 m=0 i=5 w=0.704
        # alpha=[], beta=[1], k=10 m=0 i=5 w=0.901
        # float32:
        # alpha=[], beta=[1], k=14 m=0 i=0 w=0.741
        # alpha=[], beta=[1], k=12 m=0 i=11 w=0.701
        # alpha=[], beta=[1], k=10 m=0 i=3 w=0.718
        # alpha=[], beta=[1], k=8 m=0 i=0 w=0.663
        # alpha=[], beta=[1], k=6 m=0 i=2 w=0.642
        # alpha=[], beta=[1], k=5 m=0 i=0 w=1.101
        # float16:
        # alpha=[], beta=[1], k=8 m=0 i=8 w=0.854
        # alpha=[], beta=[1], k=7 m=0 i=7 w=0.699
        # alpha=[], beta=[1], k=6 m=0 i=3 w=0.582
        # alpha=[], beta=[1], k=5 m=0 i=4 w=0.532
        # alpha=[], beta=[1], k=4 m=0 i=1 w=0.587
        # Range: 10...100
        # float64
        # alpha=[], beta=[1], k=24 m=0 i=14 w=5306.157
        # alpha=[], beta=[1], k=22 m=0 i=3 w=3976.263
        # alpha=[], beta=[1], k=20 m=0 i=2 w=3159.068
        # alpha=[], beta=[1], k=18 m=0 i=8 w=1828.227
        # alpha=[], beta=[1], k=17 m=0 i=5 w=24308.984
        # alpha=[], beta=[1], k=16 m=0 i=10 w=498900.125
        # Using horner scheme, float64, taylor:
        # Range: 0.1...10
        # alpha=[], beta=[1], k=16 m=0 i=0 w=77.096
        # alpha=[], beta=[1], k=18 m=0 i=0 w=1.064
        # alpha=[], beta=[1], k=20,22 m=0 i=0 w=0.995
        # float32:
        # alpha=[], beta=[1], k=10 m=0 i=0 w=16.701
        # alpha=[], beta=[1], k=12 m=0 i=0 w=0.932
        # alpha=[], beta=[1], k=14, 16 m=0 i=0 w=0.98
        # float16:
        # alpha=[], beta=[1], k=6 m=0 i=0 w=17.657
        # alpha=[], beta=[1], k=7,8,10 m=0 i=0 w=13.562

        k = 10
        with mp_ctx.workprec(max_prec):
            min_count = None
            for m in range(1):
                for i in list(range(k + 1 if transform != "taylor" else 1))[-1:]:
                    ulp_counts = defaultdict(int)
                    for z in samples:
                        z = -z
                        result = pFqm1(ctx, dtype, alpha, beta, z, k=k, m=m, normalization_index=i, **params)
                        expected = fa.utils.mpf2float(dtype, mp_ctx.hyper(alpha, beta, fa.utils.float2mpf(mp_ctx, z)) - 1)
                        u = fa.utils.diff_ulp(result, expected)
                        assert u <= 2, (z, result, expected)
                        ulp_counts[u] += 1

                    w = sum(k_ * v_ for k_, v_ in ulp_counts.items()) / len(samples)
                    # print(f'{m=} {i=} {w=}')
                    if min_count is None or w < min_count:
                        min_count = w
                        fa.utils.show_ulp(ulp_counts, title=f"{alpha=}, {beta=}, {k=} {m=} {i=} {w=}")
