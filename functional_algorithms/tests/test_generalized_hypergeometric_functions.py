import numpy
import pytest
import math
import warnings
from collections import defaultdict
import functional_algorithms as fa
import functional_algorithms.floating_point_algorithms as fpa
import functional_algorithms.generalized_hypergeometric_functions as ghf
import functional_algorithms.polynomial as fpp
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

    ctx = fa.utils.NumpyContext(dtype)

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
                    assert numpy.isclose(ref.N_ref(z, n, k), ref.N(z, n, k))
                    assert numpy.isclose(ref.Nm1(z, n, k), ref.Nm1_ref(z, n, k))
                    assert numpy.isclose(ref.D(z, n, k), ref.D_ref(z, n, k))
                    assert numpy.isclose(ref.T(z, n, k), ref.T_ref(z, n, k))

                    result = fpa.fast_polynomial(ctx, numpy.reciprocal(z), ref.Nm1_poly(n, k), reverse=False) / z
                    expected = dtype(ref.Nm1_ref(z, n, k))
                    assert numpy.isclose(result, expected)

                    nresult = fpa.fast_polynomial(ctx, numpy.reciprocal(z), ref.N_poly(n, k), reverse=False)
                    expected = dtype(ref.N_ref(z, n, k))
                    assert numpy.isclose(nresult / z, expected)

                    dresult = fpa.fast_polynomial(ctx, numpy.reciprocal(z), ref.D_poly(n, k), reverse=False)
                    expected = ref.D_ref(z, n, k)
                    assert numpy.isclose(dresult / z ** (n + 1), expected)

                    tresult = nresult / dresult * z**n
                    expected = ref.T_ref(z, n, k)
                    assert numpy.isclose(tresult, expected)

                    if n == 0:
                        assert numpy.isclose(ref.s(z, n - 1), 0)
                        assert numpy.isclose(ref.s(z, n), 1)


@pytest.mark.parametrize(
    "transform",
    ["drummond", "drummond-m1", "levin", "levin-m1", "taylor"],
)
def test_pFq_coeffs(dtype, transform):

    if transform == "drummond-m1":
        pFq_coeffs = ghf.pFqm1_drummond_coeffs
        params = dict()
    elif transform == "drummond":
        pFq_coeffs = ghf.pFq_drummond_coeffs
        params = dict()
    elif transform == "levin-m1":
        pFq_coeffs = ghf.pFqm1_levin_coeffs
        params = dict(gamma=3)
    elif transform == "levin":
        pFq_coeffs = ghf.pFq_levin_coeffs
        params = dict(gamma=3)
    elif transform == "taylor":
        pFq_coeffs = ghf.pFq_taylor_coeffs
        params = dict()
    else:
        assert 0, transform  # not impl

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
            for k in range(5):
                if transform.endswith("-m1"):
                    N_ref = ref.Nm1_poly(n, k)
                else:
                    N_ref = ref.N_poly(n, k)
                D_ref = ref.D_poly(n, k)

                N, D = pFq_coeffs(alpha, beta, k, n=n, **params)

                assert N == N_ref
                assert D == D_ref

                for i in [0, (k + 1) // 2][:0]:
                    Nm1, D = pFq_coeffs(alpha, beta, k, n=n, **params)
                    Nm1, D = ghf.normalize_rational_sequences(Nm1, D, normalization_index=i)
                    assert [c * D_ref[i] for c in N] == N_ref
                    assert [c * D_ref[i] for c in D] == D_ref


@pytest.mark.parametrize(
    "transform",
    ["drummond", "levin"],
)
def test_pFq_eval(transform):
    dtype = numpy.float64
    params = dict()

    if transform.startswith("levin"):
        params.update(gamma=3)

    pFq_eval = ghf.pFq_impl

    ctx = fa.utils.NumpyContext(dtype)
    size = 4
    samples = list(fa.utils.real_samples(size=size, dtype=dtype, min_value=0.1, max_value=100))

    for alpha, beta in [
        ([], []),
        ([], [fractions.Fraction(1, 2)]),
        ([fractions.Fraction(1, 2)], []),
        ([], [3]),
        ([1, 2], [3, 4, 5]),
    ]:
        ref = ghf.Reference(alpha, beta, transform=transform, **params)
        refm1 = ghf.Reference(alpha, beta, transform=transform + "-m1", **params)
        for k in range(4):
            for n in range(3):
                for z in samples:
                    expected = ref.T(z, n, k)
                    expected_m1 = refm1.Tm1(z, n, k)
                    assert numpy.isclose(expected_m1, ref.T(z, n, k) - 1)
                    for m in range(-1, 2):
                        for normalization_index in list(range(k + 1)) + [None]:
                            result_m1 = pFq_eval(
                                ctx,
                                dtype,
                                alpha,
                                beta,
                                z,
                                k=k,
                                m=m,
                                n=n,
                                normalization_index=normalization_index,
                                transform=transform + "-m1",
                                **params,
                            )
                            assert numpy.isclose(result_m1, expected_m1), expected

                            result = pFq_eval(
                                ctx,
                                dtype,
                                alpha,
                                beta,
                                z,
                                k=k,
                                m=m,
                                n=n,
                                normalization_index=normalization_index,
                                transform=transform,
                                **params,
                            )
                            assert numpy.isclose(result, expected)


def test_taylorat(dtype):
    from fractions import Fraction

    D = [Fraction(1, 270), Fraction(-2, 27), Fraction(1, 1), Fraction(-112, 15)]
    T = ghf.taylorat(D, -1)

    fctx = fa.utils.FractionContext()
    assert fpa.laurent(fctx, Fraction(-1), D, m=0) == fpa.laurent(fctx, Fraction(0), T, m=0)
    assert fpa.laurent(fctx, Fraction(-4), D, m=0) == fpa.laurent(fctx, Fraction(-3), T, m=0)

    ctx = fa.utils.NumpyContext(dtype)
    atol = 1e-1 if dtype == numpy.float16 else 1e-7
    assert numpy.isclose(fpa.laurent(ctx, dtype(-1), D, m=0), fpa.laurent(ctx, dtype(0), T, m=0), atol=atol)
    assert numpy.isclose(fpa.laurent(ctx, dtype(-4), D, m=0), fpa.laurent(ctx, dtype(-3), T, m=0), atol=atol * 10)


@pytest.mark.parametrize(
    "transform",
    ["drummond-m1", "levin-m1", "drummond", "levin", "taylor", "taylor-m1", "scipy"],
)
def test_pFq_validate(dtype, transform):
    import mpmath

    fi = numpy.finfo(dtype)

    min_value = fi.smallest_normal
    max_value = 0.1

    min_value = numpy.sqrt(fi.eps)
    # min_value = 1e-5
    # max_value = 5e-5
    min_value = 1
    max_value = 10

    pFq = ghf.pFq_impl
    if transform == "scipy":
        try:
            import scipy
        except ImportError as msg:
            pytest.skip(f"failed to import scipy: {msg}")
        params = dict()
    else:
        params = dict(transform=transform)
        if transform.startswith("levin"):
            params.update(gamma=2)

    ctx = fa.utils.NumpyContext(dtype)
    fctx = fa.utils.FractionContext()
    size = 100
    samples = list(fa.utils.real_samples(size=size, dtype=dtype, min_value=min_value, max_value=max_value))
    mp_ctx = mpmath.mp
    max_prec = {numpy.float16: 4 * 11, numpy.float32: 8 * 24, numpy.float64: 22 * 53}[dtype]

    for alpha, beta in [
        ([], [1]),  # J0
        ([], [2]),  # J1
    ][:1]:
        if transform == "scipy":
            assert len(alpha) == 0
            assert len(beta) == 1
            pFq = lambda x: dtype(scipy.special.hyp0f1(beta[0], x))
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

        print()

        with mp_ctx.workprec(max_prec):
            ulp_counts = defaultdict(int)
            min_k = 1
            for z in samples:
                z = -z
                fz = fa.utils.float2fraction(z)
                if transform.endswith("-m1"):
                    expected = fa.utils.mpf2float(dtype, mp_ctx.hyper(alpha, beta, fa.utils.float2mpf(mp_ctx, z)) - 1)
                else:
                    expected = fa.utils.mpf2float(dtype, mp_ctx.hyper(alpha, beta, fa.utils.float2mpf(mp_ctx, z)))

                if transform == "scipy":
                    result = pFq(z)
                    u = fa.utils.diff_ulp(result, expected)
                else:
                    # m - defines z ** m scaling
                    # normalization_index - defines coefficients scaling, None means no scaling
                    # n - defines rational scaling
                    # n + 2 * k - is the number of coefficients in series
                    # take n = 0 for balanced rational approximation
                    # take k = 0 for taylor approximation
                    if 0:
                        # Find minimal k that ensures ULP=0 approximation
                        # for the given dtype using exact arithmetics
                        for k in range(min_k, 1000):
                            fresult = pFq(fctx, dtype, alpha, beta, fz, k=k, m=0, n=1, normalization_index=None, **params)
                            u = fa.utils.diff_ulp(dtype(fresult), expected)
                            if u == 0:
                                min_k = k
                                break
                        else:
                            print(f"no k found for {z=}, current ULP count is {u}")
                        print(f"{z=} {min_k=} {abs(z) / (min_k ** 4)=} {int(1 + 28 * numpy.sqrt(numpy.sqrt(abs(z))))=}")
                        # the estimate to k dependent on abs(z) is implemented in ghf.pFq_impl for alpha=[], beta=[1]
                    z0 = -100.5
                    z0 = -2.5
                    z0 = -2.0
                    z0 = -5.1
                    z0 = -(2.40483**2) / 4
                    result = pFq(
                        ctx,
                        dtype,
                        alpha,
                        beta,
                        z,
                        k=None,
                        m=0,
                        n=1,
                        normalization_index=["with-smallest", "with-largest", "with-maximal-range", -1][1],
                        expansion_length=1,
                        z0=z0,
                        **params,
                    )
                    u = fa.utils.diff_ulp(result, expected)
                    if u > 10 and 0:
                        print(f"{z=} {result=} {expected=} {type(result)} {u=}")

                ulp_counts[u] += 1
                mark = (str(u) if u else ".") if u < 8 else ("^" if numpy.isfinite(result) else "N")
                print(mark, end="", flush=True)
            fa.utils.show_ulp(ulp_counts, title=f"{alpha=}, {beta=}")


@pytest.mark.parametrize(
    "transform",
    ["drummond", "levin", "taylor", "scipy"],
)
def test_hyp0f1(dtype, transform):
    import mpmath

    fi = numpy.finfo(dtype)

    min_value = fi.smallest_normal
    min_value = 0
    min_value = numpy.sqrt(fi.eps)
    # min_value = 7
    max_value = 70 * 2
    # min_value = 1e-5
    # max_value = 5e-5
    # min_value = 1
    # min_value = 34
    # max_value = 1e23

    pFq = ghf.hyp0f1
    if transform == "scipy":
        try:
            import scipy
        except ImportError as msg:
            pytest.skip(f"failed to import scipy: {msg}")
        params = dict()
    else:
        params = dict(transform=transform)
        if transform.startswith("levin"):
            params.update(gamma=2)

    ctx = fa.utils.NumpyContext(dtype)
    fctx = fa.utils.FractionContext()
    size = 100
    samples = list(fa.utils.real_samples(size=size, dtype=dtype, min_value=min_value, max_value=max_value))
    mp_ctx = mpmath.mp
    max_prec = {numpy.float16: 4 * 11, numpy.float32: 8 * 24, numpy.float64: 22 * 53}[dtype]
    max_prec = 2200
    alpha, beta = [], [1]

    if transform == "scipy":
        assert len(alpha) == 0
        assert len(beta) == 1
        pFq = lambda x: dtype(scipy.special.hyp0f1(beta[0], x))

    k = 34
    n = 0
    with mp_ctx.workprec(1200):
        zeros = fa.utils.number2fraction(ghf.hyp0f1_zeros(1, end=10, niter=20))
        N, D = ghf.pFq_levin_coeffs(alpha, beta, k, n=n)
        N, D = ghf.normalize_rational_sequences(dtype, N, D, normalization_index="with-largest")
        C = ghf.pFq_taylor_coeffs(alpha, beta, k)
        rC = fpp.asrpolynomial(C)
        Clst = [C]
        Nlst = [N]
        start, step = 0, 0
        # print(f'{fa.utils.number2float(dtype, fpp.asrpolynomial(Clst[-1]))=}')
        for i in range(start, start + step):
            if 0:
                Z1 = [1, -1 / zeros[i]]
                Clst[-1] = fpp.multiply(Clst[-1], zeros[i])
            else:
                Z1 = [-zeros[i], 1]
            Clst[-1], R1 = fpp.divmod(Clst[-1], Z1, reverse=False)
            Nlst[-1], R1 = fpp.divmod(Nlst[-1], Z1[::-1], reverse=True)
            # print(f'{fa.utils.number2float(dtype, R1)=}')
            if 0:
                Clst[0] = fpp.multiply(Clst[0], Z1)
            else:
                Clst.insert(0, Z1)
                Nlst.insert(0, Z1[::-1])
            # print(f'{fa.utils.number2float(dtype, fpp.asrpolynomial(Clst[-1]))=}')
        # print(f'{Clst=}')

        z0 = zeros[start]
        z1 = zeros[start + 1]
        C0 = fpp.asrpolynomial(fpp.taylorat(C, z0, reverse=False)[1:], reverse=False)
        print()
        # print(f'{fa.utils.number2float(dtype, C0)=}')
        # return
        rD = fpp.asrpolynomial(D, reverse=True)
        rN = fpp.asrpolynomial(Nlst[-1], reverse=True)
        # rNlst = Nlst[:-1] + [fpp.asrpolynomial(Nlst[-1])]

        Clst = fa.utils.number2float(dtype, Clst)
        Nlst = fa.utils.number2float(dtype, Nlst)
        if transform == "taylor":
            params.update(taylor_rseries=fa.utils.number2float(dtype, rC))
            params.update(taylor_series=fa.utils.number2float(dtype, C))
            params.update(
                taylor_series_with_zero=(
                    fa.utils.number2float(dtype, fpp.taylorat(C, z0)[1:]),
                    fa.utils.number2float(dtype, z0),
                )
            )
            params.update(
                taylor_rseries_with_zero=(
                    fa.utils.number2float(dtype, fpp.asrpolynomial(fpp.taylorat(C, z0)[1:])),
                    fa.utils.number2float(dtype, z0),
                )
            )
            params.update(taylor_product=Clst)
        if transform == "levin":
            params.update(levin_product=(Nlst, D))
            params.update(
                levin_rseries=(
                    fa.utils.number2float(dtype, Nlst[:-1]),
                    fa.utils.number2float(dtype, rN),
                    fa.utils.number2float(dtype, rD),
                )
            )

    if transform == "taylor":
        params.update(zeros=zeros)
    zeros = fa.utils.number2float(dtype, zeros)

    with mp_ctx.workprec(max_prec):
        ulp_counts = defaultdict(int)
        min_k = 1
        max_error = 0
        result_lst = []
        expected_lst = []
        for z in samples:
            z = -z
            fz = fa.utils.float2fraction(z)
            expected = fa.utils.mpf2float(dtype, mp_ctx.hyper(alpha, beta, fa.utils.float2mpf(mp_ctx, z)))

            if transform == "scipy":
                result = pFq(z)
                # u = fa.utils.diff_ulp(result, expected)
            else:
                z0 = None
                result = pFq(
                    ctx,
                    dtype,
                    z,
                    k=k,
                    m=0,
                    n=0,
                    normalization_index=["with-smallest", "with-largest", "with-maximal-range", -1][1],
                    expansion_length=1,
                    z0=z0,
                    **params,
                )
            result_lst.append(result)
            expected_lst.append(expected)
            u = fa.utils.diff_prec(result, expected)
            # u = abs(abs(u) - abs(fi.machep))
            u = abs(abs(u) - abs(fi.machep))
            if u > 5 and 0:
                print(f"{z=} {result=} {expected=} {type(result)} {u=} {result/expected=}")

            err = abs(result - expected)
            if err > max_error:
                max_error = err
                if 0:
                    print(f"{z=} {result=} {expected=} {err=}")

            ulp_counts[u] += 1
            mark = (str(u) if u else ".") if u < 10 else ("^" if numpy.isfinite(result) else "N")
            print(mark, end="", flush=True)
            # return
        fa.utils.show_ulp(ulp_counts, title=f"{alpha=}, {beta=}")

    return

    import matplotlib.pyplot as plt

    print(len(result_lst))

    result_lst = numpy.array(result_lst)
    expected_lst = numpy.array(expected_lst)

    plt.subplot(211)
    plt.plot(samples, result_lst, label="result")
    plt.plot(samples, expected_lst, label="reference")
    plt.legend()
    plt.subplot(212)
    plt.semilogy(samples, abs(result_lst - expected_lst), label="result - reference")
    plt.legend()

    print(f"wrote hyp0f1.jpg")
    plt.savefig("hyp0f1.jpg")


def test_pFq_taylor_coeffs(dtype):
    import mpmath

    fi = numpy.finfo(dtype)

    alpha, beta = [], [1]
    k = 110
    k1 = 40
    c = 2
    size = 4
    z0 = [-1.4457965, -7.6178155][1]
    z0 = fa.utils.float2fraction(z0)

    C = ghf.pFq_taylor_coeffs(alpha, beta, k, c=c)
    C0 = fa.polynomial.taylorat(C, c * z0, reverse=False, size=size)
    C0 = fa.utils.number2float(dtype, C0)

    T0 = [fa.polynomial.fast_polynomial(c * z0, T) for T in ghf.pFq_taylor_coeffs(alpha, beta, k1, c=c, i=range(size))]
    T0 = fa.utils.number2float(dtype, T0)

    assert C0 == T0


@pytest.mark.parametrize(
    "beta",
    [1, 2],
)
def test_hyp0f1_zeros(dtype, beta):
    fi = numpy.finfo(dtype)

    # Find the index of a hyp0f1 zero point that coincides with its
    # initial estimate within the given floating-point system.
    # Results (beta==1):
    #  float16: 7 (end=start+1), 39 (end=start+50)
    #  float32: 652 (end=start+1)
    #  float64: > 1000000
    #
    # For expansions larger than 1 no such upper limit exist, that is,
    # initial estimate is always inexact for expansions with length >=
    # 2

    def iszero_exact(value):
        if value is None:
            return False
        return value == 0

    def iszero(value):
        if value is None:
            return False
        if value == 0:
            return True
        return fa.utils.number2float(dtype, value) == 0

    size = 2

    # tested with size 1 and 2, zeros index up to 1000000:
    prec = (-fi.negep) * (size + 1) + 5

    niter = {numpy.float16: 4, numpy.float32: 5, numpy.float64: 6}[dtype]
    niter += int(beta)

    for start in range(0, 1000000, 30000):
        zeros1 = ghf.hyp0f1_zeros(beta, start=start, end=start + 1, niter=1100, iszero=iszero, prec=prec)
        zeros2 = ghf.hyp0f1_zeros(beta, start=start, end=start + 1, niter=1100, iszero=iszero_exact, prec=1100)
        zeros3 = ghf.hyp0f1_zeros(beta, start=start, end=start + 1, niter=niter, iszero=iszero, prec=prec)

        zeros1_fp = fa.utils.number2float(dtype, zeros1)
        zeros2_fp = fa.utils.number2float(dtype, zeros2)
        zeros3_fp = fa.utils.number2float(dtype, zeros3)

        assert zeros1_fp == zeros2_fp
        assert zeros1_fp == zeros3_fp

        zeros1_ex = fa.utils.mpf2expansion(dtype, zeros1, length=size, functional=True)
        zeros2_ex = fa.utils.mpf2expansion(dtype, zeros2, length=size, functional=True)
        zeros3_ex = fa.utils.mpf2expansion(dtype, zeros3, length=size, functional=True)

        assert zeros1_ex == zeros2_ex
        assert zeros1_ex == zeros3_ex

        if not numpy.isfinite(zeros1_ex[0][0]):
            break
