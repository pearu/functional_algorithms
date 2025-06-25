import numpy
import fractions
import functional_algorithms as fa
import functional_algorithms.generalized_hypergeometric_functions as ghf


def test_zeros_aberth():
    import mpmath

    dtype = numpy.float64
    z0 = fractions.Fraction(1, 3)
    z1 = fractions.Fraction(5, 2)
    z2 = fractions.Fraction(-5, 2)
    z3 = 0

    P = [-z0, 1]
    P = fa.polynomial.multiply(P, [-z1, 1])
    P = fa.polynomial.multiply(P, [-z2, 1])
    P = fa.polynomial.multiply(P, [-z3, 1])
    P = fa.polynomial.multiply(P, [1, 0, -4])

    working_prec = 120
    ctx = mpmath.mp
    with mpmath.workprec(working_prec):
        P = fa.utils.number2mpf(ctx, P)
        r0 = fa.polynomial.zeros_aberth(P, niter=20)
        r0 = sorted(fa.utils.mpf2float(dtype, r0))
        expected = sorted(fa.utils.number2float(dtype, [z0, z1, z2, z3, 0.5, -0.5]))
        assert r0 == expected


def test_zeros_of_hyp0f1():
    dtype = numpy.float64
    import mpmath

    working_prec = 1100  # depends on dtype
    ctx = mpmath.mp
    with mpmath.workprec(working_prec):

        alpha = []
        beta = [1]

        dalpha = [c + 1 for c in alpha]
        dbeta = [c + 1 for c in beta]
        d = ghf.pochhammer(alpha, 1) / ghf.pochhammer(beta, 1)

        def P(x, der=0):
            if der == 0:
                return ctx.hyper(alpha, beta, fa.utils.number2mpf(ctx, x))
            elif der == 1:
                return d * ctx.hyper(dalpha, dbeta, fa.utils.number2mpf(ctx, x))
            else:
                assert 0, der  # unreachable

        # estimate initial zeros from asymptotics
        #   cos(z - a * pi/2 - pi/4) ~ 0L1(,a,-z**2/4)
        #   z - a * pi/2 - pi/4 == pi/2 + k * pi
        #   r == z**2/4
        #   r == (pi/2 + k * pi + a * pi/2 + pi/4)**2 / 4
        #   r == ((k + a/2 + 3/4) * pi)**2 / 4
        zeros_init = [-1 - numpy.pi**2 * k * (k + 1) / 4 - k for k in range(30)]

        # compute zeros
        zeros = fa.polynomial.zeros_aberth(P, niter=12, zeros_init=zeros_init)

        # ensure that all initial zeros have converged
        assert len(zeros) == len(zeros_init)

        # check that we have zeros within dtype precision
        values = fa.utils.mpf2float(dtype, list(map(P, zeros)))
        assert max(map(abs, values)) == 0.0

        # make sure we got all zeros
        dvalues = [P(r, der=1) for r in zeros]
        for i in range(1, len(dvalues)):
            assert dvalues[i] * dvalues[i - 1] < 0


def test_divmod():
    for P, D in [
        ([2, 5, 3, 1, 7], [1, 2, 4]),
        ([2, 5, 3], [1, 2]),
        ([2, 5, 3, 0], [1, 2]),
        (fa.polynomial.multiply([2, 1], [3, 1]), [3, 1]),
        (fa.polynomial.multiply([2, 1], [3, 1]), fa.polynomial.multiply([2, 1], [3, 1])),
    ]:
        P = fa.utils.number2fraction(P)
        D = fa.utils.number2fraction(D)
        Q, R = fa.polynomial.divmod(P, D)
        result = fa.polynomial.add(fa.polynomial.multiply(Q, D), R)
        while P and P[-1] == 0:
            P.pop()
        assert result == P


def test_rpolynomial():
    dtype = numpy.float64
    for C in [
        [1, 2, 3],
        [2, 3, 5, 7, 9, 11],
    ]:
        C = fa.utils.number2fraction(C)
        rC = fa.polynomial.asrpolynomial(C)
        for ax in fa.utils.real_samples(size=20, dtype=dtype, min_value=1e-3, max_value=1000):
            for x in [-ax, ax]:
                f = fa.utils.number2fraction(x)
                expected = fa.polynomial.fast_polynomial(f, C)
                result = fa.polynomial.rpolynomial(f, rC)
                assert result == expected
