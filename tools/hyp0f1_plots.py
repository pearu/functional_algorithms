import mpmath
import numpy
import math
import matplotlib.pyplot as plt
import functional_algorithms as fa
import functional_algorithms.floating_point_algorithms as fpa
import functional_algorithms.generalized_hypergeometric_functions as ghf
import functional_algorithms.polynomial as fpp
import scipy
from collections import defaultdict
import warnings


def main_taylor(dtype):
    alpha, beta = [], [1]
    k = 34
    mp_prec = 110
    mp_ctx = mpmath.mp
    with mp_ctx.workprec(1500):
        roots = fa.utils.number2fraction(ghf.hyp0f1_roots(beta[0], start=0, end=20, niter=100))

    fp_roots = fa.utils.number2float(dtype, roots)
    print(f"{fp_roots=}")

    roots_samples = []
    roots_width = 1
    for r in fp_roots[3:13]:
        roots_samples.append(r)
        for i in range(1, roots_width + 1):
            roots_samples.insert(-i, numpy.nextafter(roots_samples[-i], dtype(numpy.inf)))
        for i in range(1, roots_width + 1):
            roots_samples.append(numpy.nextafter(roots_samples[-1], dtype(-numpy.inf)))

    roots_samples = numpy.array(roots_samples[::-1], dtype=dtype)

    print(f"{roots_samples=}")

    fi = numpy.finfo(dtype)
    min_value, max_value = numpy.sqrt(fi.eps), 1500

    min_value, max_value = 10, fi.max ** (0.5**3)
    min_value, max_value = 10, 10000
    # min_value, max_value = 33, 40
    # min_value, max_value = 17, 20
    # min_value, max_value = 6, 9
    # min_value, max_value = 0, 2
    # min_value, max_value = 1 / fi.eps, min(1 / fi.eps ** 2, fi.max)
    eps = 3e1
    root = -fp_roots[19]
    # min_value, max_value = root - eps, root + eps
    size = 1000
    # size = 5
    samples = fa.utils.real_samples(size=size, dtype=dtype, min_value=min_value, max_value=max_value, include_infinity=False)
    samples = -numpy.array(samples, dtype=dtype)
    samples = roots_samples
    # print(f'{samples.shape=}')
    ctx = fa.utils.NumpyContext(default_constant_type=dtype)

    with mp_ctx.workprec(1500):
        # roots = fa.utils.number2fraction(ghf.hyp0f1_roots(beta[0], start=0, end=10, niter=100))
        expected_mp = [mp_ctx.hyper(alpha, beta, fa.utils.float2mpf(mp_ctx, z)) for z in samples]
        expected = numpy.array([fa.utils.mpf2float(dtype, v) for v in expected_mp], dtype=dtype)

    # print(f'{expected=}')

    # Taylor
    C = ghf.pFq_taylor_coeffs(alpha, beta, k)
    rC = fpp.asrpolynomial(C)
    rC = fa.utils.number2float(dtype, rC)
    with warnings.catch_warnings(action="ignore"):
        result = fpa.rpolynomial(ctx, samples, rC, reverse=False)

    result_levin = None
    if 0:
        # Levin
        N, D = ghf.pFq_levin_coeffs(alpha, beta, k, n=0, gamma=2)
        N, D = ghf.normalize_rational_sequences(dtype, N, D, normalization_index="with-largest")
        rN = fpp.asrpolynomial(N, reverse=True)
        rD = fpp.asrpolynomial(D, reverse=True)
        rN = fa.utils.number2float(dtype, rN)
        rD = fa.utils.number2float(dtype, rD)

        with warnings.catch_warnings(action="ignore"):
            result_levin = fpa.rpolynomial(ctx, samples, rN, reverse=True) / fpa.rpolynomial(ctx, samples, rD, reverse=True)

    # SciPy
    result_scipy = scipy.special.hyp0f1(beta[0], samples).astype(dtype)
    # print(f'{result_scipy=}')

    result_cos = None
    if 0:
        # Asymptotic
        with mp_ctx.workprec(mp_prec):
            JC, JS = ghf.hyp0f1_coeffs_asymptotic(beta[0], max_k=20)

        rJC = fpp.asrpolynomial(JC, reverse=False)
        rJS = fpp.asrpolynomial(JS, reverse=False)
        etype = numpy.float64
        etype = dtype
        sz = numpy.sqrt(-samples.astype(etype))
        sz = sz.astype(etype)
        # omega = sz * dtype(2) - dtype(numpy.pi * beta[0] / 2 - numpy.pi / 4)
        # cos(2 * z - pi * (b / 2 - 1/ 4)) = cos(2 * z) * cos(pi * (b / 2 - 1/4)) + sin(2 * z) * sin(pi * (b / 2 - 1/4))
        # sin(2 * z - pi * (b / 2 - 1/ 4)) = sin(2 * z) * cos(pi * (b / 2 - 1/4)) - cos(2 * z) * sin(pi * (b / 2 - 1/4))

        # Evaluating cos/sin(2 * sz) in higher precsion will considerably
        # improve accuracy of hyp0f1. TODO: implement apmath.cos/sin
        # support.
        cs2z = numpy.cos(sz + sz).astype(dtype)
        sn2z = numpy.sin(sz + sz).astype(dtype)

        csb = etype(numpy.cos(numpy.pi * (beta[0] / 2 - 0.25)))
        snb = etype(numpy.sin(numpy.pi * (beta[0] / 2 - 0.25)))

        cso = cs2z * csb + sn2z * snb
        sno = sn2z * csb - cs2z * snb

        # csz = etype(numpy.power(-samples.astype(etype), -beta[0] / 2 + 1 / 4))
        # ssz = etype(numpy.power(-samples.astype(etype), -beta[0] / 2 - 1 / 4))

        ez = numpy.log(-samples)
        csz = numpy.exp(ez * (-beta[0] / 2 + 1 / 4))
        ssz = numpy.exp(ez * (-beta[0] / 2 - 1 / 4))

        with warnings.catch_warnings(action="ignore"):
            jc = numpy.array(fa.polynomial.rpolynomial(numpy.reciprocal(-samples), rJC), dtype=etype).T
            js = numpy.array(fa.polynomial.rpolynomial(numpy.reciprocal(-samples), rJS), dtype=etype).T

        # result_cos = (cso * jc - sno * js / (sz + sz) ) * ssz
        result_cos = cso * jc * csz - sno * js * ssz
        result_cos = numpy.array(result_cos, dtype=dtype)

        assert result_cos.dtype == dtype, (result_cos.dtype, dtype)

    result_apmath = None
    if 0:
        # APMath
        result_apmath = fa.apmath.hypergeometric(ctx, alpha, beta, [samples], k * 2, functional=True, size=2)
        result_apmath = numpy.array(result_apmath).T
        with mp_ctx.workprec(mp_prec):
            result_apmath_mp = [fa.utils.expansion2mpf(mp_ctx, v) for v in result_apmath]
        result_apmath = numpy.array(list(map(sum, result_apmath)), dtype=dtype)

    result_cos_apmath_mp = None
    if 0:
        # APMath asymptotic
        with mp_ctx.workprec(mp_prec):
            result_cos_apmath = fa.apmath.hypergeometric0f1_asymptotic(
                ctx, beta[0], [samples], functional=True, size=3, mp_ctx=mp_ctx
            )
            result_cos_apmath = numpy.array(result_cos_apmath).T
        with mp_ctx.workprec(mp_prec):
            result_cos_apmath_mp = [fa.utils.expansion2mpf(mp_ctx, v) for v in result_cos_apmath]
        result_cos_apmath = numpy.array(list(map(sum, result_cos_apmath)), dtype=dtype)

    if 0:
        # APMath Taylor
        with mp_ctx.workprec(mp_prec):
            result_taylor_apmath = fa.apmath.hypergeometric0f1_taylor(
                ctx, beta[0], [samples], functional=True, size=2
            )  # , mp_ctx=mp_ctx)
            result_taylor_apmath = numpy.array(result_taylor_apmath).T
        with mp_ctx.workprec(mp_prec):
            result_taylor_apmath_mp = [fa.utils.expansion2mpf(mp_ctx, v) for v in result_taylor_apmath]
        result_taylor_apmath = numpy.array(list(map(sum, result_taylor_apmath)), dtype=dtype)
        # print(f'{result_taylor_apmath=}')
        # print(f'{samples=}')
        # print(f'{fa.utils.number2float(dtype, roots)=}')
        if 0:
            # APMath Taylor
            C = ghf.pFq_taylor_coeffs(alpha, beta, k * 3)
            rC = fpp.asrpolynomial(C)
            rC = [fa.utils.number2expansion(dtype, c_, length=2, functional=True) for c_ in rC]

            result_taylor_apmath = [
                fa.apmath.rpolynomial(ctx, [z_], rC, reverse=False, functional=True, size=2) for z_ in samples
            ]

            with mp_ctx.workprec(mp_prec):
                result_taylor_apmath_mp = [fa.utils.expansion2mpf(mp_ctx, v) for v in result_taylor_apmath]
            result_taylor_apmath = numpy.array(list(map(sum, result_taylor_apmath)), dtype=dtype)

    result_0f1_apmath = None
    if 1:
        # APMath asymptotic
        with mp_ctx.workprec(mp_prec):
            result_0f1_apmath = fa.apmath.hypergeometric0f1(ctx, beta[0], [samples], functional=True, size=2, mp_ctx=mp_ctx)
            result_0f1_apmath = numpy.array(result_0f1_apmath).T
        with mp_ctx.workprec(mp_prec):
            result_0f1_apmath_mp = [fa.utils.expansion2mpf(mp_ctx, v) for v in result_0f1_apmath]
        result_0f1_apmath = numpy.array(list(map(sum, result_0f1_apmath)), dtype=dtype)
        print(f"{result_0f1_apmath=}")
    result_taylor0_apmath = None
    if 0:
        # APMath Taylor at z0
        C = ghf.pFq_taylor_coeffs(alpha, beta, k * 3)
        az0 = roots[3]
        rC = fpp.asrpolynomial(fpp.taylorat(C, az0, reverse=False)[1:], reverse=False)
        rC = [fa.utils.number2expansion(dtype, c_, length=2, functional=True) for c_ in rC]
        az0 = fa.utils.number2expansion(dtype, az0, length=2, functional=True)

        result_taylor0_apmath = fa.apmath.rtaylor(ctx, [samples], rC, az0, reverse=False, functional=True, size=2)
        result_taylor0_apmath = numpy.array(result_taylor0_apmath).T
        with mp_ctx.workprec(mp_prec):
            result_taylor0_apmath_mp = [fa.utils.expansion2mpf(mp_ctx, v) for v in result_taylor0_apmath]
        result_taylor0_apmath = numpy.array(list(map(sum, result_taylor0_apmath)), dtype=dtype)

    # Figure
    if 1:
        plt.figure(figsize=(14, 10), dpi=300)
    else:
        plt.figure(figsize=(8, 10), dpi=300)
        plt.subplot(211)

        plt.plot(samples, result, label=f"taylor(z)")
        if result_levin:
            plt.plot(samples, result_levin, label=f"levin(z)")
        if result_cos:
            plt.plot(samples, result_cos, label=f"asymptote")
        if 0:
            plt.plot(samples, result_apmath, label=f"apmath")
            plt.plot(samples, result_taylor_apmath, label=f"taylor apmath")
        plt.plot(samples, result_scipy, label=f"scipy", linestyle=":")
        plt.plot(samples, expected, label=f"reference", linestyle="--", color="k")

        plt.ylim(min(expected) - 1, max(expected) + 1)
        plt.legend()

    plt.title(f"hyp0f1[beta={beta[0]}], {dtype.__name__}, {k=}")

    if 0:
        plt.subplot(212)
    else:
        plt.ylabel("precision")
    plt.xlabel("z")

    def error(result):
        if isinstance(result[0], numpy.floating):
            r = fa.utils.matching_bits(result, expected)
        elif isinstance(result[0], mp_ctx.mpf):
            with mp_ctx.workprec(mp_prec):
                err = fa.utils.matching_bits(result, expected_mp)
            r = numpy.array(err)
        else:
            assert 0, type(result[0])  # not impl
        print(f"{result=}")
        print(f"{expected=}")
        print(f"{r=}")
        return r

    graphs = []

    linewidth = 2
    if 0:
        graphs.append(((samples, error(result)), dict(label=f"taylor(z)", linewidth=linewidth)))
    if result_levin:
        graphs.append(((samples, error(result_levin)), dict(label=f"levin(z)", linewidth=linewidth)))

    if result_0f1_apmath is not None:
        graphs.append(((samples, error(result_0f1_apmath)), dict(label=f"apmath 0f1(z)", linewidth=linewidth)))

    last_z0 = None
    for z0 in roots:
        break
        if abs(z0) < 2 or abs(z0) > 35 * 2:
            continue
        rC = fa.utils.number2float(dtype, fpp.asrpolynomial(fpp.taylorat(C, z0, reverse=False)[1:], reverse=False))
        z0_ = fa.utils.number2float(dtype, z0)
        result = numpy.array([fpa.rpolynomial(ctx, z - z0_, rC, reverse=False) * (z - z0_) for z in samples], dtype=dtype)
        if 0:
            graphs.append(((samples, error(result)), dict(label=f"taylorat[z0={z0_:1.2f}](z)", linewidth=linewidth)))

        if last_z0 is not None and 0:
            z1 = (last_z0 + z0) / 2
            rC = fa.utils.number2float(dtype, fpp.asrpolynomial(fpp.taylorat(C, z1, reverse=False), reverse=False))
            z1_ = fa.utils.number2float(dtype, z1)
            result = numpy.array([fpa.rpolynomial(ctx, z - z1_, rC, reverse=False) for z in samples], dtype=dtype)
            graphs.append(
                (
                    (samples, error(result)),
                    dict(label=f"taylorat[z1={z1_:1.2f}](z)", linewidth=linewidth, linestyle="--"),
                )
            )

        last_z0 = z0
        last_z0_ = z0_

    if result_cos:
        graphs.append(((samples, error(result_cos)), dict(label=f"asymptote", linewidth=linewidth)))
    if result_cos_apmath_mp is not None:
        error_result_cos_apmath_mp = error(result_cos_apmath_mp)
        err_prec = defaultdict(int)
        for p in error_result_cos_apmath_mp:
            err_prec[p] += 1
        fa.utils.show_prec(err_prec)
        graphs.append(((samples, error_result_cos_apmath_mp), dict(label=f"asymptote apmath [mpf]", linewidth=linewidth)))

        # graphs.append(
        #    (
        #        (samples, error(result_taylor_apmath)),
        #        dict(label=f"taylor apmath [sum]", linewidth=linewidth, linestyle="-."),
        #    )
        # )
        graphs.append(
            (
                (samples, error(result_taylor_apmath_mp)),
                dict(label=f"taylor apmath [mpf]", linewidth=linewidth, linestyle="-."),
            )
        )
    if result_taylor0_apmath is not None:
        graphs.append(
            (
                (samples, error(result_taylor0_apmath_mp)),
                dict(label=f"taylorat[z0={sum(az0):1.2f}](z) apmath [mpf]", linewidth=linewidth),
            )
        )
    if result_apmath is not None:
        # graphs.append(
        #    ((samples, error(result_apmath)), dict(label=f"apmath [sum]", linewidth=linewidth, linestyle="--"))
        # )
        graphs.append(((samples, error(result_apmath_mp)), dict(label=f"apmath [mpf]", linewidth=linewidth, linestyle="--")))
    graphs.append(
        (
            (samples, error(result_scipy)),
            dict(label=f"scipy.special", linewidth=linewidth, linestyle=":", color="k"),
        )
    )

    mn = 1
    for args, kwargs in graphs:
        if (args[1] != 0).any():
            mn = min(mn, args[1][numpy.nonzero(args[1])].min())

    def envelope(x, y):
        eN = 100
        width = abs(x[-1] - x[0]) / eN
        new_hy = []
        new_ly = []
        for i in range(len(x)):
            mask = abs(x - x[i]) <= width / 2
            new_hy.append(numpy.max(y[mask]))
            new_ly.append(numpy.min(y[mask]))
        return x, numpy.array(new_ly, dtype=y.dtype), numpy.array(new_hy, dtype=y.dtype)

    for args, kwargs in graphs:
        if 1:
            x, y = args
            for n in range(len(x) // (2 * roots_width + 1)):
                start = n * (2 * roots_width + 1)
                end = start + (2 * roots_width + 1)
                if n:
                    kwargs.pop("label", None)
                plt.plot(x[start:end], y[start:end], **kwargs)
            continue

        if 0:
            args[1][args[1] == 0] = mn * 1e-1
            args[1][args[1] > 1] = 1
            x, l, h = envelope(*args)
            plt.semilogy(x, h, **kwargs)
        else:
            x, l, h = envelope(*args)
            plt.plot(x, l, **kwargs)

    plt.legend()
    fn = f"hyp0f1_taylor_{dtype.__name__}.jpg"
    plt.savefig(fn)
    print(f"Created {fn}")


if __name__ == "__main__":
    if 1:
        main_taylor(numpy.float16)
    if 1:
        main_taylor(numpy.float32)
    if 1:
        main_taylor(numpy.float64)
