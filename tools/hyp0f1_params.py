import numpy
import functional_algorithms as fa
import functional_algorithms.generalized_hypergeometric_functions as ghf
import mpmath


def main(dtype, mp_ctx):
    fi = numpy.finfo(dtype)
    p = -fi.negep
    ctx = fa.utils.NumpyContext(dtype)
    eps = fi.eps

    root_appoximation_degree = 3
    root_range_end = 10
    b = 1
    k = 80
    size = 2

    roots = fa.utils.number2fraction(ghf.hyp0f1_roots(b, count=root_range_end, niter=100))
    roots_ex = [fa.utils.fraction2expansion(dtype, r) for r in roots]
    roots_fp = numpy.array([fa.utils.fraction2float(dtype, r) for r in roots])

    # from experiments:
    upper_k = {
        numpy.float64: [11, 17, 22, 27, 32, 36, 41, 45, 50, 54, 59, 63, 68, 72, 76, 81, 85, 89, 94, 98, 102, 107],
    }.get(dtype, [])

    last_r = 0
    k1_lst = []
    w_lst = []
    for j in range(len(roots)):
        k0 = upper_k[j] + 5 if j < len(upper_k) else 10 + int(4.6 * j) + 5

        C0 = [
            fa.polynomial.fast_polynomial(roots[j], T)
            for T in ghf.pFq_taylor_coeffs([], [b], k0, c=1, i=range(1, root_appoximation_degree + 1))
        ]
        rC0 = fa.polynomial.asrpolynomial(C0, reverse=False)
        rC0 = [fa.utils.number2expansion(dtype, c_, length=2, functional=True) for c_ in rC0]

        r_fp = fa.utils.fraction2float(dtype, roots[j])
        z = dtype(r_fp)
        z_mp = fa.utils.number2mpf(mp_ctx, r_fp)
        expected_mp = mp_ctx.hyper([], [b], z_mp)
        # result = fa.apmath.rtaylor.impl(ctx, dtype, [z], rC0, roots_ex[j], reverse=False, functional=True, size=size)
        result = fa.apmath.hypergeometric0f1_asymptotic.impl(ctx, dtype, b, [z], functional=True, size=size)
        result_mp = fa.utils.expansion2mpf(mp_ctx, result)
        prec = fa.utils.matching_bits(result_mp, expected_mp)
        print(f"{j=} {rC0=} {prec=}")
        print(f"{fa.utils.fraction2expansion(numpy.float16, fa.utils.float2fraction(rC0[0]))=}")
        z = dtype(r_fp)
        if dtype == numpy.float16:
            z -= dtype(1e-4 * eps * abs(z))
        elif dtype == numpy.float32:
            z -= dtype(1e0 * eps * abs(z))
        else:
            z -= dtype(1e2 * eps * abs(z))
        c = 0
        while True:
            c += 1
            z_mp = fa.utils.number2mpf(mp_ctx, z)
            expected_mp = mp_ctx.hyper([], [b], z_mp)

            result = fa.apmath.rtaylor.impl(ctx, dtype, [z], rC0, roots_ex[j], reverse=False, functional=True, size=size)
            result_mp = fa.utils.expansion2mpf(mp_ctx, result)

            prec = fa.utils.matching_bits(result_mp, expected_mp)
            if dtype == numpy.float16 and prec < p:
                break
            elif dtype != numpy.float16 and prec < p + 2:
                break

            if dtype == numpy.float64:
                z -= dtype(1e8 * eps * abs(z))
            elif dtype == numpy.float32:
                z -= dtype(1e2 * eps * abs(z))
            else:
                z0 = z
                z = numpy.nextafter(z, dtype(0))
                if z0 == z:
                    break

        w = abs(z - dtype(r_fp))
        # print(f'{r_fp=} {w=} {c=} {prec=}')
        last_r = r_fp
        if c == 1:
            break

        w_lst.append(w)

        k1 = k0
        z = dtype(r_fp) - dtype(0.5 * w)
        z_mp = fa.utils.number2mpf(mp_ctx, z)
        expected_mp = mp_ctx.hyper([], [b], z_mp)
        while k1 > 0:
            C0 = [
                fa.polynomial.fast_polynomial(roots[j], T)
                for T in ghf.pFq_taylor_coeffs([], [b], k1, c=1, i=range(1, root_appoximation_degree + 1))
            ]
            rC0 = fa.polynomial.asrpolynomial(C0, reverse=False)
            rC0 = [fa.utils.number2expansion(dtype, c_, length=2, functional=True) for c_ in rC0]

            result = fa.apmath.rtaylor.impl(ctx, dtype, [z], rC0, roots_ex[j], reverse=False, functional=True, size=size)
            result_mp = fa.utils.expansion2mpf(mp_ctx, result)

            prec = fa.utils.matching_bits(result_mp, expected_mp)

            if prec < p + 2:
                break

            k1 -= 1

        k1_lst.append(k1)

    if dtype == numpy.float16:
        w_lst = numpy.array(w_lst, dtype=numpy.float32)
        roots_fp = numpy.array(roots_fp, dtype=numpy.float32)

    print(f"{k1_lst=}")
    # model: k = A + B * i
    A, B = numpy.polynomial.Polynomial.fit(range(len(k1_lst)), k1_lst, deg=1).convert().coef
    print(f">>>>> k = {A} + {B} * i")
    # model: w = A * abs(r) ** B
    # log(w) = log(A) + B * log(abs(r))
    print(f"{w_lst=}")
    lA, B = numpy.polynomial.Polynomial.fit(numpy.log(abs(roots_fp[: len(w_lst)])), numpy.log(w_lst), deg=1).convert().coef
    print(f">>>>> w = {numpy.exp(lA)} * abs(r) ** ({B})")


if __name__ == "__main__":
    mp_ctx = mpmath.mp
    with mp_ctx.workprec(2000):
        main(numpy.float64, mp_ctx)
        # main(numpy.float32, mp_ctx)
        # main(numpy.float16, mp_ctx)
