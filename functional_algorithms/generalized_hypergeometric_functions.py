"""Algorithms for computing generalized hypergeometric functions.

We consider the generalized hypergeometric functions in the form

  pFq(alpha, beta, z) = lim(s_n, n->oo)

where

  s_n = sum(a_k, k=0,...,n)
  a_k = (alpha)_k / (beta)_k * z ** k / k!
  (alpha)_k = prod((alpha[i])_k, i=1..p)
  (beta)_k = prod((beta[i])_k, i=1..q)

  Pochhammer symbol:
    (x)_k = Gamma(x + k) / Gamma(x) = x * (x + 1) * ... * (x + k - 1)
    (x)_0 = 1
    (x)_{-k} = 1 / (x - k)_{k} for k > 0

For the theoretical background of using rational approximations for
evaluating pFq, see

  Slevinsky, R.M. Fast and stable rational approximation of
  generalized hypergeometric functions. Numer Algor 98, 587–624
  (2025). https://doi.org/10.1007/s11075-024-01808-w

  An early version of the paper is also available as

  Richard Mikael Slevinsky.  Fast and stable rational approximation of
  generalized hypergeometric functions.  2023.
  https://doi.org/10.48550/arXiv.2307.06221

The Slevinsky(2025) paper describes two algorithms for computing pFq
that are based on using Drummond' and factorial Levin-type sequence
transformations, respectively, and it defines recursive relations for
computing a family of rational approximations for pFq.

Here we assume that the parameters in alpha and beta are fixed and the
only variable is z. When using a JIT compiler to compile the code that
calls pFq, the z-independent parts of the recursive relations are
resolved at compile time and therefore there is little gain from
implementing the full algorithms provided by Slevinsky(2025) that
assume that alpha and beta are variables as well. Also, when the input
z to pFq is an array, using stopping criteria that depends on z is
suboptimal for concurrent evaluation of pFq. Instead, we'll use a
fixed parameter for the number of iterations for all possible z
values. All these considerations will simplify the algorithms
describled in Slevinsky(2025) considerably but we'll need to
preprocess (see below) the z-independent parts of the recursive
relations provided by Slevinsky(2025) leading to a formulation of the
algorithms that does not need recursive relations at all.

Outline of the rational approximations of pFq using sequence transforms
-----------------------------------------------------------------------

A sequence transformation is defined as a solution of a remainder
estimate

  T_n^{(k)} - s_n = omega_n * z_n^{(k)}

where omega_n != 0.

For Drummond' sequence transformation, z_n^{(k)} is a polynomial with
respect to n that has k parameters.

For factorial Levin-type sequence transformation

  z_n^{(k)} = sum(c_j / (n + gamma)_j, j=0,...,k-1)

where c is a set of k parameters and gamma > 0.

It is assumed that there exists an annihilation operator Delta that
applied k times to the remainder estimate will give a linear system of
equations that can be solved for the k parameters and T_n^{(k)}.
The Slevinsky(2025) paper describes such a solution as a set of
recurrence relations for both above mentioned sequence
transformations. The solution is given in the form

  T_n^{(k)} = N_n^{(k)} / D_n^{(k)}

In the following, we'll use the remainder estimate

  omega_n = Delta(s_n) = a_{n + 1}

where Delta(s_n) is the forward difference operator

  Delta(s_n) = s_{n + 1} - s_n

For Drummond' sequence transformation, we'll have

  N_n^{(k)} = Delta^k (s_n / omega_n)
            = sum(binom(k, j) * (-1) ** (k - j) * s_{n + j} / omega_{n + j}, j=0..k)
  D_n^{(k)} = Delta^k (1 / omega_n)
            = sum(binom(k, j) * (-1) ** (k - j) / omega_{n + j}, j=0..k)

For factorial Levin-type sequence transformation, we'll have

  N_n^{(k)} = Delta^k ((n + gamma)_{k - 1} * s_n / omega_n)
            = sum(binom(k, j) * (-1) ** (k - j) * (n + j + gamma)_{k - 1} * s_{n + j} / omega_{n + j}, j=0..k)
  D_n^{(k)} = Delta^k ((n + gamma)_{k - 1} / omega_n)
            = sum(binom(k, j) * (-1) ** (k - j) * (n + j + gamma)_{k - 1} / omega_{n + j}, j=0..k)

In the following, we'll separate z-independent and z-dependent parts
of N_n^{(k)} and D_n^{(k)}. 

Theorem 1
---------

For a sequence transform, there exists z-independent coefficients N_m and D_m such that

  N_n^{(k)} = sum(N_m / z ** m, m=0..n + k) / z
  D_n^{(k)} = sum(D_m / z ** m, m=0..k) / z ** (n + 1)

Hint 1: use Horner scheme to evaluated the polynomials N_n^{(k)} and D_n^{(k)} over 1 / z.

Hint 2: Pre-computed N_m and D_m can be re-scaled with C * z for
better numerical stability when evaluating N_n^{(k)} / D_n^{(k)}:

  T_n^{(k)} = sum(N_m * C / z ** m, m=0..n + k) / sum(D_m * C / z ** m, m=0..k) / z ** n

that also suggest using n == 0 for balancing the polynomial orders of
the numerator and denominator, respectively.

Proof of Theorem 1:

  Let's assume Drummond' sequence transformation and find

    N_n^{(k)} = sum(binom(k, j) * (-1) ** (k - j) * s_{n + j} / omega_{n + j}, j=0..k)
              = sum(binom(k, j) * (-1) ** (k - j) * sum(a_i / a_{n + j + 1}, i=0..n + j), j=0..k)

  There exists z-independent constant C such that

    a_i / a_{n + j + 1} = C * z ** -(n + j + 1 - i) = C * z ** -(j - i) * z ** -(n + 1)

  where the z exponent (n + j + 1 - i) varies from 1 to n + 1 + k
  because j - i varies from -n (j=0, i=n) to k (j=k, i=0).
  Hence, we can write

    N_n^{(k)} = sum(N_m / z ** m, m = -n..k) / z ** -(n + 1)
              = sum(N_m / z ** m, m = 0..n + k) / z

  Similarly, let's find

    D_n^{(k)} = sum(binom(k, j) * (-1) ** (k - j) / omega_{n + j}, j=0..k)
              = sum(binom(k, j) * (-1) ** (k - j) / a_{n + j + 1}, j=0..k)

  There exists z-independent constant C such that

    1 / a_{n + j + 1} = C * z ** -(n + j + 1)

  where the z exponent (n + 1 + j) varies from n + 1 to n + 1 + k.
  Hence, we can write

    D_n^{(k)} = sum(D_m / z ** m, m = n + 1..n + 1 + k)
              = sum(D_m / z ** m, m = 0..k) / z ** (n + 1)

  Same arguments apply to other types of sequence transformations.

QED
"""

import math
import fractions
import functional_algorithms as fa
import functional_algorithms.floating_point_algorithms as fpa


def pochhammer(x, n):
    r = 1
    if isinstance(x, (tuple, list)):
        for x_ in x:
            r *= pochhammer(x_, n)
    else:
        for i in range(n):
            r *= x + i
    return r


class Reference:

    def __init__(self, alpha, beta, transform="drummond", gamma=2):
        self.alpha = alpha
        self.beta = beta
        self.transform = transform
        self.gamma = gamma

    def omega(self, z, n):
        """Compute omega_n(z)"""
        return self.a(z, n + 1)

    def weight(self, n, k):
        if self.transform == "levin":
            return pochhammer(n + self.gamma, k - 1)
        return 1

    def A(self, k):
        """Compute a_k(z) / z ** k"""
        return fractions.Fraction(pochhammer(self.alpha, k), pochhammer(self.beta, k) * math.factorial(k))

    def a(self, z, k):
        """Compute a_k(z) = A_k * z ** k"""
        return self.A(k) * z**k

    def s(self, z, n):
        """Compute s_n(z)"""
        s = 0
        for k in range(n + 1):
            s += self.a(z, k)
        return s

    def N_ref(self, z, n, k):
        """Compute N_n^{(k)}(z)"""
        s = 0
        for j in range(k + 1):
            s += math.comb(k, j) * (-1) ** (k - j) * self.weight(n + j, k) * self.s(z, n + j) / self.omega(z, n + j)
        return s

    def D_ref(self, z, n, k):
        """Compute D_n^{(k)}(z)"""
        s = 0
        for j in range(k + 1):
            s += math.comb(k, j) * (-1) ** (k - j) * self.weight(n + j, k) / self.omega(z, n + j)
        return s

    def T_ref(self, z, n, k):
        """Compute T_n^{(k)}(z)"""
        return self.N_ref(z, n, k) / self.D_ref(z, n, k)

    def Nm1_ref(self, z, n, k):
        """Compute Nm1_n^{(k)}(z) such that

        T_n^{(k)} - 1 = Nm1_n^{(k)} / D_n^{(k)}
        """
        s = 0
        for j in range(k + 1):
            for i in range(j + 1):
                s += math.comb(k, j) * (-1) ** (k - j) * self.weight(n + j, k) * self.a(z, n + i) / self.omega(z, n + j)
        return s

    def Nm1(self, z, n, k):
        """Compute Nm1_n^{(k)}(z) such that

        T_n^{(k)} - 1 = Nm1_n^{(k)} / D_n^{(k)}
        """

        s = 0
        for m in range(k + 1):
            s1 = 0
            for j in range(k + 1 - m):
                s1 += (
                    math.comb(k, m + j)
                    * (-1) ** (k - j - m)
                    * self.weight(n + m + j, k)
                    * self.A(n + j)
                    / self.A(n + m + 1 + j)
                )
            s += s1 * z**-m
        return s / z

    def Nm1_poly(self, n, k):
        """Compute polynomial for Nm1_n^{(k)}(z) such that

        Nm1_n^{(k)}(z) == polynomial(Nm1, 1 / z) / z
        """
        Nm1 = []
        for m in range(k + 1):
            s1 = 0
            for j in range(k + 1 - m):
                s1 += (
                    math.comb(k, m + j)
                    * (-1) ** (k - j - m)
                    * self.weight(n + m + j, k)
                    * self.A(n + j)
                    / self.A(n + m + 1 + j)
                )
            Nm1.append(s1)
        return Nm1

    def Nsplit(self, z, n, k):
        """Compute s_{n - 1} D_n^{(k)} + Nm1_{n}^((k))"""
        return self.s(z, n - 1) * self.D_ref(z, n, k) + self.Nm1_ref(z, n, k)

    def D(self, z, n, k):
        """Compute D_n^{(k)}(z)"""
        s = 0
        for j in range(k + 1):
            s += math.comb(k, j) * (-1) ** (k - j) * self.weight(n + j, k) / self.A(n + j + 1) * z**-j
        return s * z ** -(n + 1)

    def D_poly(self, n, k):
        """Compute polynomial for D_n^{(k)}(z) such that

        D_n^{(k)}(z) == polynomial(D, 1 / z) / z ** (n + 1)
        """
        D = []
        for j in range(k + 1):
            D.append(math.comb(k, j) * (-1) ** (k - j) * self.weight(n + j, k) / self.A(n + j + 1))
        return D

    def T(self, z, n, k):
        """Compute T_n^{(k)}(z)"""
        nm1 = self.Nm1(z, n, k)
        d = self.D(z, n, k)
        return self.s(z, n - 1) + nm1 / d

    def Tm1(self, z, n, k):
        """Compute T_n^{(k)}(z) - 1"""
        nm1 = self.Nm1(z, n, k)
        d = self.D(z, n, k)
        if n == 1:
            return nm1 / d
        return self.s(z, n - 1) - 1 + nm1 / d


def pFq_taylor_coeffs(a, b, k):
    """Return T such that

    s_k = polynomial(z, T)
    """
    A = lambda k: fractions.Fraction(pochhammer(a, k), pochhammer(b, k) * math.factorial(k))
    T = []
    for j in range(k + 1):
        T.append(A(j))
    return T


def pFq_drummond_coeffs(a, b, k, n=0, normalization_index=None):
    """Return Nm1, D such that

      T_n^{(k)} = s_{n - 1} + polynomial(1 / z, Nm1) / polynomial(1 / z, D) * z ** n

    and

      max(D) == 1
    """
    assert isinstance(k, int)
    assert isinstance(n, int)
    if normalization_index is None:
        normalization_index = k

    A = lambda k: fractions.Fraction(pochhammer(a, k), pochhammer(b, k) * math.factorial(k))
    Nm1 = []
    D = []
    for m in range(k + 1):
        s1 = 0
        for j in range(k + 1 - m):
            s1 += math.comb(k, m + j) * (-1) ** (k - j - m) * A(n + j) / A(n + m + 1 + j)
        Nm1.append(s1)
        D.append(math.comb(k, m) * (-1) ** (k - m) / A(n + m + 1))

    w = D[normalization_index]
    Nm1 = [c / w for c in Nm1]
    D = [c / w for c in D]

    return Nm1, D


def pFq_levin_coeffs(a, b, k, n=0, gamma=2, normalization_index=None):
    """Return Nm1, D such that

    T_n^{(k)} = s_{n - 1} + polynomial(1 / z, Nm1) / polynomial(1 / z, D) * z ** n

    and

      max(D) == 1
    """
    assert isinstance(k, int)
    assert isinstance(n, int)
    if normalization_index is None:
        normalization_index = k
    A = lambda k: fractions.Fraction(pochhammer(a, k), pochhammer(b, k) * math.factorial(k))
    Nm1 = []
    D = []
    for m in range(k + 1):
        s1 = 0
        for j in range(k + 1 - m):
            c = pochhammer(n + m + j + gamma, k - 1)
            s1 += math.comb(k, m + j) * (-1) ** (k - j - m) * c * A(n + j) / A(n + m + 1 + j)
        Nm1.append(s1)
        c = pochhammer(n + m + gamma, k - 1)
        D.append(math.comb(k, m) * (-1) ** (k - m) / A(n + m + 1) * c)

    w = D[normalization_index]
    Nm1 = [c / w for c in Nm1]
    D = [c / w for c in D]

    return Nm1, D


def pFq_minus_one_impl(ctx, dtype, a, b, z, k=None, m=None, normalization_index=None, transform=None, **params):
    """Compute pFq(a, b, z) - 1 using rational approximation using a sequence transformation."""
    # TODO: find optimal k, m, and normalization_index for the given dtype
    if k is None:
        k = 10
    if m is None:
        m = 0
    if normalization_index is None:
        normalization_index = 4
    n = 1
    if transform == "levin" or transform is None:
        Nm1, D = pFq_levin_coeffs(a, b, k, n=n, normalization_index=normalization_index, **params)
    elif transform == "drummond":
        Nm1, D = pFq_drummond_coeffs(a, b, k, n=n, normalization_index=normalization_index, **params)
    elif transform == "taylor":
        C = pFq_taylor_coeffs(a, b, k)
        C = [ctx.constant(c, z) for c in C[1:]]
        return fpa.laurent(ctx, z, C, m=1, reverse=False, scheme=fpa.horner_scheme)
    else:
        assert 0, transform  # unreachable
    Nm1 = [ctx.constant(c, z) for c in Nm1]
    D = [ctx.constant(c, z) for c in D]
    numer = fpa.laurent(ctx, z, Nm1, m=m + n, reverse=True, scheme=fpa.horner_scheme)
    denom = fpa.laurent(ctx, z, D, m=m, reverse=True, scheme=fpa.horner_scheme)
    return numer / denom


def pFq_impl(ctx, dtype, a, b, z, k=None, m=None, normalization_index=None, transform=None, **params):
    """Compute pFq(a, b, z) using rational approximation using a sequence transformation."""
    one = ctx.constant(1, z)
    return (
        pFq_minus_one_impl(
            ctx, dtype, a, b, z, k=k, m=m, normalization_index=normalization_index, transform=transform, **params
        )
        + one
    )
