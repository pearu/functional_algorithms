import functional_algorithms as fa
import numpy
import math


def fast_exponent_by_squaring(x, n):
    """Evaluate x ** n by squaring."""
    if n == 0:
        return 1
    if n == 1:
        return x
    if n == 2:
        return x * x
    assert n > 0
    r = fast_exponent_by_squaring(x, n // 2)
    r2 = r * r
    return r2 if n % 2 == 0 else r2 * x


def canonical_scheme(k, N):
    return k


def horner_scheme(k, N):
    return 1


def estrin_dac_scheme(k, N):
    import math

    return int(math.log(k))


def balanced_dac_scheme(k, N):
    return k // 2


def fast_polynomial(x, coeffs, reverse=False, scheme=None, _N=None):
    """Evaluate a polynomial

     P(x) = coeffs[N] + coeffs[N - 1] * x + ... + coeffs[0] * x ** N

    when reverse is True, otherwise evaluate a polynomial

      P(x) = coeffs[0] + coeffs[1] * x + ... + coeffs[N] * x ** N

    where `N = len(coeffs) - 1`, using "Fast polynomial evaluation and
    composition" algorithm by G. Moroz.

    scheme is an int-to-int unary function. Examples:

      scheme = lambda k, N: k            # canonical polynomial
      scheme = lambda k, N: 1            # Horner's scheme [default]
      scheme = lambda k, N: int(log(k))  # Estrin' DAC scheme
      scheme = lambda k, N: k // 2       # balanced DAC scheme

    Reference:
      https://hal.science/hal-00846961v3
    """

    if reverse:
        return fast_polynomial(x, list(reversed(coeffs)), reverse=False, scheme=scheme)

    if scheme is None:
        scheme = horner_scheme
        alt_scheme = balanced_dac_scheme
    else:
        alt_scheme = scheme

    N = len(coeffs) - 1
    if _N is None:
        _N = N

    if N == 0:
        return coeffs[0]

    if N == 1:
        return coeffs[0] + coeffs[1] * x

    if len(coeffs) > 500:
        # to avoid maximal recursion depth exceeded exception
        d = alt_scheme(N, _N)
    else:
        d = scheme(N, _N)

    if d == 0:
        # evaluate reduced polynomial as it is
        s = coeffs[0]
        for i in range(1, N):
            s += coeffs[i] * fast_exponent_by_squaring(x, i)
        return s

    a = fast_polynomial(x, coeffs[d:], reverse=reverse, scheme=scheme, _N=_N)
    b = fast_polynomial(x, coeffs[:d], reverse=reverse, scheme=scheme, _N=_N)
    xd = fast_exponent_by_squaring(x, d)
    return a * xd + b


def asrpolynomial(coeffs, reverse=False):
    if reverse:
        return asrpolynomial(coeffs[::-1], reverse=False)[::-1]
    rcoeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        rcoeffs.append(coeffs[i] / coeffs[i - 1])
    return rcoeffs


def rpolynomial(x, rcoeffs, reverse=False):
    """Evaluate a polynomial

      P(x) = coeffs[0] + coeffs[1] * x + ... + coeffs[N] * x ** N

    where coefficients are given as ratios of coefficients and their
    following counterparts:

      rcoeffs[i] = coeffs[i] / coeffs[i - 1]

    where we define coeffs[-1] == 1.
    """
    if reverse:
        return rpolynomial(x, rcoeffs[::-1], reverse=False)

    r = 1
    for rc in reversed(rcoeffs[1:]):
        r = 1 + rc * x * r
    return r * rcoeffs[0]


def multiply(P, Q, reverse=False):
    """Return the coefficients of the polynomial expression
    polynomial(P, x) * polynomial(Q, x)

    P(x) = sum(P[i] * x^i, i=0..N)
    Q(x) = sum(Q[j] * x^j, j=0..M)

    P(x) * Q(x) = sum(P[i] * Q[j] * x^{i+j}, i=0..M, j=0..M)
    """
    if not isinstance(Q, list):
        return multiply(P, [Q], reverse=reverse)
    if not isinstance(P, list):
        return multiply([P], Q, reverse=reverse)

    if reverse:
        return list(reversed(multiply(list(reversed(P)), list(reversed(Q)), reverse=False)))

    lst = [0] * (len(P) + len(Q) - 1)
    for i, p in enumerate(P):
        for j, q in enumerate(Q):
            k = i + j
            lst[k] += p * q
    return lst


def add(P, Q, reverse=False):
    """Return the coefficients of the polynomial expression
    polynomial(P, x) + polynomial(Q, x)
    """
    if not isinstance(Q, list):
        return add(P, [Q], reverse=reverse)
    if not isinstance(P, list):
        return add([P], Q, reverse=reverse)

    if reverse:
        return list(reversed(add(list(reversed(P)), list(reversed(Q)), reverse=False)))

    lst = []
    for i in range(min(len(P), len(Q))):
        lst.append(P[i] + Q[i])
    if len(P) < len(Q):
        lst.extend(Q[len(P) :])
    elif len(P) > len(Q):
        lst.extend(P[len(Q) :])
    return lst


def divmod(P, D, reverse=False):
    """Given polynomials P and D, return polynomials Q and R such that

    P = Q * D + R
    """
    if reverse:
        Q, R = divmod(list(reversed(P)), list(reversed(D)), reverse=False)
        return list(reversed(Q)), list(reversed(R))

    P = P[:]
    while P and P[-1] == 0:
        P.pop()

    D = D[:]
    while D and D[-1] == 0:
        D.pop()

    if len(P) < len(D):
        return [], P
    n = len(P) - len(D) + 1
    ld = D[-1]
    D = [0] * (len(P) - len(D)) + D

    Q = []
    R = P
    for k in range(n):
        if not R:
            break
        t = R[-1] / ld
        Q.insert(0, t)
        R = add(R, multiply(-t, D[k:], reverse=reverse), reverse=reverse)
        while R and R[-1] == 0:
            R.pop()
    while Q and Q[-1] == 0:
        Q.pop()
    return Q, R


def derivative(P, n=1, reverse=False):
    """Given a polynomial polynomial(P, x), return coefficients of its
    n-th derivative.
    """
    if reverse:
        return list(reversed(derivative(list(reversed(P)), n=n, reverse=False)))

    if n == 0:
        return P

    if n > 1:
        return derivative(derivative(P, n=1, reverse=reverse), n=n - 1, reverse=reverse)

    assert n == 1
    return [P[i] * i for i in range(1, len(P))]


def zeros_aberth(P, niter=10, zeros_init=None, reverse=False, scheme=None, iszero=None):
    """Given a polynomial polynomial(P, x) return all zeros of the given
    polynomial. If P is callable, it must have a signature

      P(x, der=0) -> value

    and it is assumed that the corresponding function has polynomial
    zeros, that is, there exists a polynomial Q and a function F such
    that

      P(x) = Q(x) * F(x)

    This function uses Aberth method for computing the zeros, see
    https://en.wikipedia.org/wiki/Aberth_method
    """
    if isinstance(P, list):
        dP = derivative(P, n=1, reverse=reverse)

        def func(x, der=0):
            if der == 0:
                return fast_polynomial(x, P, reverse=reverse, scheme=scheme)
            elif der == 1:
                return fast_polynomial(x, dP, reverse=reverse, scheme=scheme)
            else:
                assert 0, der  # unreachable

    elif callable(P):
        func = P
    else:
        assert 0, type(P)  # unreachable

    if zeros_init is None:
        zeros_init = list(range(len(P) - 1))
    elif isinstance(zeros_init, int):
        zeros_init = list(range(zeros_init))
    else:
        assert isinstance(zeros_init, list), type(zeros_init)

    if iszero is None:

        def iszero(value):
            return value == 0

    elif callable(iszero):
        pass
    else:
        assert 0, type(iszero)  # unreachable

    zeros = zeros_init[:]
    convergence = [None] * len(zeros_init)
    for n in range(niter):
        has_converged = True
        for i, x in enumerate(zeros_init):
            if iszero(convergence[i]):
                continue
            has_converged = False
            p = func(x, der=0)
            dp = func(x, der=1)
            f = p / dp
            r = 0
            for j, y in enumerate(zeros_init):
                if i != j:
                    r += 1 / (x - y)
            w = f / (1 - f * r)
            zeros[i] -= w
            convergence[i] = zeros[i] - zeros_init[i]
        if has_converged:
            break
        zeros_init = zeros[:]
    zeros = [r for i, r in enumerate(zeros) if iszero(convergence[i])]
    return zeros


def taylorat(P, z0, reverse=False, size=None):
    """Return Taylor coefficients for a polynomial at z = z0 such that

      sum(C_m * (z - z0) ** m, m=0..k) == sum(P_m * z ** m, m=0..k)

    If reverse is True, then return

      taylorat(P[::-1], z0, reverse=False)[::-1]

    Notice that when z0 is a zero of P then C_0 == 0.
    """
    if reverse:
        return taylorat(P[::-1], z0, reverse=False)[::-1]
    if isinstance(z0, (float, numpy.floating)):
        z0 = fa.utils.float2fraction(z0)
    k = len(P) - 1
    C = []
    if size is None:
        size = k + 1
    for m in range(size):
        s = 0
        z0e = 1
        for _, j in enumerate(range(m, k + 1)):
            # e == j - m
            s += P[j] * math.comb(j, m) * z0e
            z0e *= z0
        C.append(s)
    return C
