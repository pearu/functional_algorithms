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
the numerator and denominator, respectively. Taking C = z ** (n + k) leads to

  T_n^{(k)} = sum(N_m * z ** (n + k - m), m=0..n + k) / sum(D_m * z ** (k - m), m=0..k)

which is a rational function with coefficients in reverse order and
this form avoids the need to compute the reciprocal of z.

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


Lemma 1
-------

Consider a sum

  P^{(k)} = sum(P_j * z ** j, j=0..k)

around z = z0:

  P = sum(P_j * (z - z0 + z0) ** j, j=0..k)
    = sum(P_j * sum(binom(j, m) (z-z0)^m * z0^{j-m} * [m <= j], m=0..k), j=0..k)

    = sum(sum(P_j * binom(j, m) * z0^{j-m}, j=m..k) * (z - z0)^m, m=0..k)
    = sum(C_m * (z - z0)^m, m=0..k)

where

  C_m = sum(P_j * binom(j, m) * z0^{j-m}, j=m..k)

Lemma 2
-------

  (x)_(j + i) = Gamma(x + j + i) / Gamma(x)
              = Gamma(x + i + j) / Gamma(x + i) * Gamma(x + i) / Gamma(x)
              = (x + i)_j * (x)_i

Lemma 3
-------

  pFq(a, b, c * z) = pFq(a, b, c * (z - z0 + z0))
                   = sum_{i=0..oo} T_{i} * (z - z0) ** i

where

  T_{i} = pFq(a + i, b + i, c * z0) * c ** i / i! * pochhammer(a, i) / pochhammer(b, i)

Proof:

    pFq(a, b, c * z) = 
      = sum_{k=0..oo} pochhammer(a, k) / pochhammer(b, k) * c ** k / k! * z ** k
      = sum_{k=0..oo} pochhammer(a, k) / pochhammer(b, k) * c ** k / k! * (z - z0 + z0) ** k
      = sum_{k=0..oo} pochhammer(a, k) / pochhammer(b, k) * c ** k / k! * sum_{i=0..k} k! / i! / (k-i)! z0 ** (k - i) * (z - z0) ** i
      = sum_{k=0..oo, i=0..k} pochhammer(a, k) / pochhammer(b, k) * c ** k / i! / (k-i)! z0 ** (k - i) * (z - z0) ** i
      = sum_{i=0..oo} T_{i} * (z - z0) ** i

    Let's find

      sum_{k=0..oo, i=0..k} T_{k, i} * xi ** i = sum_{k=0..oo, i=0..oo} T_{k, i} * xi ** i * [i <= k]
          = sum_{i=0..oo, k=0..oo} T_{k, i} * [i <= k] * xi ** i
          = sum_{i=0..oo, k=i..oo} T_{k, i} * xi ** i
          = sum_{i=0..oo, j=0..oo} T_{j + i, i} * xi ** i
 
    Hence,

      T_i = sum(j=0..oo) pochhammer(a, j + i) / pochhammer(b, j + i) * c ** (j + i) / i! / j! * z0 ** j
          = sum(j=0..oo) pochhammer(a + i, j) * pochhammer(a, i) / pochhammer(b + i, j) / pochammer(b, i) * c ** (j + i) / i! / j! * z0 ** j
          = c ** i / i! * pochhammer(a, i) / pochhammer(b, i) * sum(j=0..oo) pochhammer(a + i, j) / pochhammer(b + i, j) * c ** j / j! * z0 ** j
          = pFq(a + i, b + i, c * z0) * c ** i / i! * pochhammer(a, i) / pochhammer(b, i)

QED
"""

import math
import fractions
import numpy
import mpmath
import functional_algorithms as fa
import functional_algorithms.floating_point_algorithms as fpa
import functional_algorithms.polynomial as fpp


def pochhammer(x, n):
    r = 1
    if isinstance(x, (tuple, list)):
        for x_ in x:
            r *= pochhammer(x_, n)
    else:
        if isinstance(x, (numpy.floating, float)):
            assert x == numpy.round(x), x  #  todo: use gamma
            x = fa.utils.float2fraction(x)

        for i in range(n):
            r *= x + i
    return r


def taylorat(P, z0, reverse=False):
    """
    Return Taylor coefficients for a polynomial at z = z0 such that

      sum(C_m * (z - z0) ** m, m=0..k) == sum(P_m * z ** m, m=0..k)

    If reverse is True, then return

      taylorat(P[::-1], z0, reverse=False)[::-1]
    """
    if reverse:
        return taylorat(P[::-1], z0, reverse=False)[::-1]
    if isinstance(z0, (float, numpy.floating)):
        z0 = fa.utils.float2fraction(z0)
    k = len(P) - 1
    C = []
    for m in range(k + 1):
        s = 0
        for j in range(m, k + 1):
            s += P[j] * math.comb(j, m) * z0 ** (j - m)
        C.append(s)
    return C


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
        if self.transform.startswith("levin"):
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

    def N(self, z, n, k):
        """Compute N_n^{(k)}(z) such that

        T_n^{(k)} = N_n^{(k)} / D_n^{(k)}
        """
        s = 0
        for q in reversed(range(k + 1)):
            s1 = 0
            for j in range(q, k + 1):
                s1 += math.comb(k, j) * (-1) ** (k - j) * self.weight(n + j, k) * self.A(j - q) / self.A(n + j + 1)
            s += s1 * z ** -(q + 1 + n)  # -k - n -1 ... -n - 1
        for q in range(n):
            s1 = 0
            for j in range(k + 1):
                s1 += math.comb(k, j) * (-1) ** (k - j) * self.weight(n + j, k) * self.A(q + 1 + j) / self.A(n + j + 1)
            s += s1 * z ** (q - n)  # -n ... -1
        return s

    def N_poly(self, n, k):
        """Compute polynomial for N_n^{(k)}(z) such that

        N_n^{(k)}(z) == polynomial(N, 1 / z) / z
        """
        N = []
        s = 0
        for q in reversed(range(n)):
            s1 = 0
            for j in range(k + 1):
                s1 += math.comb(k, j) * (-1) ** (k - j) * self.weight(n + j, k) * self.A(q + 1 + j) / self.A(n + j + 1)
            N.append(s1)
        for q in range(k + 1):
            s1 = 0
            for j in range(q, k + 1):
                s1 += math.comb(k, j) * (-1) ** (k - j) * self.weight(n + j, k) * self.A(j - q) / self.A(n + j + 1)
            N.append(s1)

        return N

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
        return self.N(z, n, k) / self.D(z, n, k)

    def Tm1(self, z, n, k):
        """Compute T_n^{(k)}(z) - 1"""
        if n == 1:
            d = self.D(z, n, k)
            nm1 = self.Nm1(z, n, k)
            return nm1 / d
        return self.T(z, n, k) - 1


def pFq_taylor_coeffs(a, b, k, c=1, i=0):
    """Return T such that

      s_n(c * z) = polynomial(z, T)

    We have

      s_n(c * (z - z0 + z0)) = polynomial(c * (z - z0), T0)

    where

      T0 = [polynomial(c * z0, pFq_taylor_coeffs(a, b, k0, c=c, i=i)) for i in range(k)]
    """
    if isinstance(i, (range, list)):
        return [pFq_taylor_coeffs(a, b, k, c=c, i=i_) for i_ in i]

    def A(k):
        if i:
            return fractions.Fraction(
                pochhammer([a_ + i for a_ in a], k) * c ** (k + i) * pochhammer(a, i),
                pochhammer([b_ + i for b_ in b], k) * pochhammer(b, i) * math.factorial(k) * math.factorial(i),
            )
        return fractions.Fraction(pochhammer(a, k) * c**k, pochhammer(b, k) * math.factorial(k))

    return [A(j) for j in range(k + 1)]


def normalize_rational_sequences(dtype, a, b, normalization_index=None):
    """Normalize sequences a and b that define a dtype rational function."""
    if normalization_index is None:
        return a, b

    if normalization_index == "with-maximal-range":
        max_db = None
        index = None
        for i in range(len(b)):
            w = b[i]
            if w == 0:
                continue
            db = fa.utils.number2float(dtype, [b_ / w for b_ in b])
            if [b_ for b_ in db if not numpy.isfinite(b_)]:
                continue
            if max_db is None or len([b_ for b_ in db if b_ != 0]) > len(max_db):
                max_db = db
                index = i
        assert index is not None
        normalization_index = index
    elif normalization_index in {"with-smallest", "with-largest"}:
        max_db = None
        index = None
        for i in range(len(b)):
            w = b[i]
            if w == 0:
                continue
            db = fa.utils.number2float(dtype, [b_ / w for b_ in b])
            if [b_ for b_ in db if not numpy.isfinite(b_)]:
                continue
            index = i
            if normalization_index == "with-smallest":
                break
        assert index is not None
        normalization_index = index
    elif normalization_index == "middle-non-zero":
        nz_b_indices = [i for i, c in enumerate(b) if c]
        normalization_index = nz_b_indices[len(nz_b_indices) // 2]
    else:
        normalization_index = dict(middle=len(b) // 2).get(normalization_index, normalization_index)
    w = b[normalization_index]
    return [c / w for c in a], [c / w for c in b]


def pFq_drummond_coeffs(a, b, k, n=0):
    """Return N, D such that

      T_n^{(k)} = polynomial(1 / z, N) / polynomial(1 / z, D) * z ** n

    and

      max(D) == 1
    """
    assert isinstance(k, int)
    assert isinstance(n, int)

    A = lambda k: fractions.Fraction(pochhammer(a, k), pochhammer(b, k) * math.factorial(k))
    N = []
    D = []

    for q in reversed(range(n)):
        s1 = 0
        for j in range(k + 1):
            s1 += math.comb(k, j) * (-1) ** (k - j) * A(q + 1 + j) / A(n + j + 1)
        N.append(s1)

    for q in range(k + 1):
        s1 = 0
        for j in range(q, k + 1):
            s1 += math.comb(k, j) * (-1) ** (k - j) * A(j - q) / A(n + j + 1)
        N.append(s1)
        D.append(math.comb(k, q) * (-1) ** (k - q) / A(n + q + 1))

    return N, D


def pFqm1_drummond_coeffs(a, b, k, n=0):
    """Return Nm1, D such that

      T_n^{(k)} = s_{n - 1} + polynomial(1 / z, Nm1) / polynomial(1 / z, D) * z ** n

    and

      max(D) == 1
    """
    assert isinstance(k, int)
    assert isinstance(n, int)

    A = lambda k: fractions.Fraction(pochhammer(a, k), pochhammer(b, k) * math.factorial(k))
    Nm1 = []
    D = []
    for m in range(k + 1):
        s1 = 0
        for j in range(k + 1 - m):
            s1 += math.comb(k, m + j) * (-1) ** (k - j - m) * A(n + j) / A(n + m + 1 + j)
        Nm1.append(s1)
        D.append(math.comb(k, m) * (-1) ** (k - m) / A(n + m + 1))
    return Nm1, D


def pFq_levin_coeffs(a, b, k, n=0, gamma=2):
    """Return N, D such that

      T_n^{(k)} = polynomial(1 / z, N) / polynomial(1 / z, D) * z ** n

    and

      max(D) == 1
    """
    assert isinstance(k, int)
    assert isinstance(n, int)
    A = lambda k: fractions.Fraction(pochhammer(a, k), pochhammer(b, k) * math.factorial(k))
    N = []
    D = []

    for q in reversed(range(n)):
        s1 = 0
        for j in range(k + 1):
            c = pochhammer(n + j + gamma, k - 1)
            s1 += math.comb(k, j) * (-1) ** (k - j) * c * A(q + 1 + j) / A(n + j + 1)
        N.append(s1)

    for q in range(k + 1):
        s1 = 0
        for j in range(q, k + 1):
            c = pochhammer(n + j + gamma, k - 1)
            s1 += math.comb(k, j) * (-1) ** (k - j) * c * A(j - q) / A(n + j + 1)
        N.append(s1)
        c = pochhammer(n + q + gamma, k - 1)
        D.append(math.comb(k, q) * (-1) ** (k - q) * c / A(n + q + 1))

    return N, D


def pFqm1_levin_coeffs(a, b, k, n=0, gamma=2):
    """Return Nm1, D such that

    T_n^{(k)} = s_{n - 1} + polynomial(1 / z, Nm1) / polynomial(1 / z, D) * z ** n

    and

      max(D) == 1
    """
    assert isinstance(k, int)
    assert isinstance(n, int)
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
        D.append(math.comb(k, m) * (-1) ** (k - m) * c / A(n + m + 1))

    return Nm1, D


def remove_leading_zeros(C):
    if C[0] == 0:
        for i, c in enumerate(C):
            if c != 0:
                return C[i:]
        return C[:1]
    return C


def hyp0f1_closest_zero(beta, z1, niter=None, iszero=None, prec=None):
    """Return the zero of hyp0f1(beta, z) that is closest to z1."""
    alpha = []
    beta = [beta]

    dalpha = [c + 1 for c in alpha]
    dbeta = [c + 1 for c in beta]
    d = pochhammer(alpha, 1) / pochhammer(beta, 1)

    if prec is None:
        prec = 1100  # depends on dtype

    mp_ctx = mpmath.mp

    if niter is None:
        niter = 12

    def P(x, der=0):
        if der == 0:
            return mp_ctx.hyper(alpha, beta, fa.utils.number2mpf(mp_ctx, x))
        elif der == 1:
            return d * mp_ctx.hyper(dalpha, dbeta, fa.utils.number2mpf(mp_ctx, x))
        else:
            assert 0, der  # unreachable

    with mp_ctx.workprec(prec):
        zeros_init = [fa.utils.number2mpf(mp_ctx, z1)]
        zeros = fpp.zeros_aberth(P, niter=niter, zeros_init=zeros_init, iszero=iszero)
    return zeros


def hyp0f1_zeros(beta, start=0, end=10, niter=None, iszero=None, prec=None):
    """Return the zeros of hyp0f1(beta, z)."""
    alpha = []
    beta = [beta]

    dalpha = [c + 1 for c in alpha]
    dbeta = [c + 1 for c in beta]
    d = pochhammer(alpha, 1) / pochhammer(beta, 1)

    if prec is None:
        prec = 1100  # depends on dtype

    mp_ctx = mpmath.mp

    if niter is None:
        niter = 12

    def P(x, der=0):
        if der == 0:
            return mp_ctx.hyper(alpha, beta, fa.utils.number2mpf(mp_ctx, x))
        elif der == 1:
            return d * mp_ctx.hyper(dalpha, dbeta, fa.utils.number2mpf(mp_ctx, x))
        else:
            assert 0, der  # unreachable

    """
    For large negative x,
      hyp0f1(b, x) ~ cos(2 * sqrt(-x) - beta * pi / 2 + pi / 4)
    that gives initial estimate to hyp0f1(b, x) zeros:
      -x = ((2 * beta + 1) / 8 + k / 2) ** 2 * pi ** 2
    where k = 0, 1, ...
    """

    with mp_ctx.workprec(prec):
        one = mp_ctx.mpf(1)
        zeros_init = [-(((2 * beta[0] + one) / 4 + k) ** 2 / 4) * mp_ctx.pi**2 for k in range(start, end)]
        if niter > 0:
            zeros = fpp.zeros_aberth(P, niter=niter, zeros_init=zeros_init, iszero=iszero)
        else:
            zeros = zeros_init
    return zeros


def hyp0f1(
    ctx, dtype, z, k=None, m=None, n=None, normalization_index=None, transform=None, expansion_length=1, z0=None, **params
):
    """
    J_a(x) = (x/2) ^ a / a! hyper([], [a], -x^2/4)

    For x >> abs(a**2 - 1/4):
    J_a(x) = sqrt(2 / (pi * x)) * cos(x - a*pi/2 - pi/4)


    hyp0f1(-x^2/4) = sqrt(2 / (pi * x)) * cos(x - pi/4)
    z = -x^2/4
    sqrt(-z) * 2 = x
    hyp0f1(-x^2/4) = sqrt(2 / (pi * x)) * cos(x - pi/4)

    For z < 0 and -z >> 1 / sqrt(8)
    hyp0f1(z) = sqrt(1 / (pi * sqrt(-z))) * cos(2 * sqrt(-z) - pi/4)
    """
    if z < -20 and 0:
        az = abs(z)
        sz = numpy.sqrt(az)
        sz2 = sz + sz
        return (numpy.cos(sz2) + numpy.sin(sz2)) / dtype(numpy.sqrt(sz2 * (numpy.pi)))

    if k is None:
        # abs(z) <= 10000, n=0, gamma=3
        if z0 is not None:
            z_ = z - z0
        else:
            z_ = z
        if transform is None or transform.startswith("levin"):
            k = int(1 + 13 * abs(z_) ** 0.25)  # float64
            # k = int(1 + 12 * abs(z) ** 0.25)  # float32
            # k = int(1 + 11 * abs(z) ** 0.25)  # float16
        elif transform.startswith("drummond"):
            k = int(1 + 26 * abs(z_) ** 0.25)  # float64
            # k = int(1 + 24 * abs(z) ** 0.25)  # float16, float32
        elif transform.startswith("taylor"):
            k = int(1 + 29 * abs(z_) ** 0.25)  # float64
            # k = int(1 + 28 * abs(z) ** 0.25)  # float16, float32
        else:
            assert 0, transform  # unreachable
    if m is None:
        m = 0
    if n is None:
        if transform.endswith("-m1"):
            n = 1
        else:
            n = 0
    a, b = [], [1]
    if transform == "levin" or transform is None:
        levin_rseries = params.get("levin_rseries")

        if levin_rseries is not None and 1:
            zP, rN, rD = levin_rseries
            numer = fpa.rpolynomial(ctx, z, rN, reverse=True)
            denom = fpa.rpolynomial(ctx, z, rD, reverse=True)
            for P in zP:
                numer *= fpa.laurent(ctx, z, P, m=0, reverse=True, scheme=fpa.horner_scheme)
            return numer / denom

        levin_product = params.get("levin_product")
        if levin_product is not None:
            Nlst, D = levin_product
            r = None
            for N in Nlst:
                r1 = fpa.laurent(ctx, z, N, m=0, reverse=True, scheme=fpa.horner_scheme)
                if r is None:
                    r = r1
                else:
                    r = r * r1
            d = fpa.laurent(ctx, z, D, m=0, reverse=True, scheme=fpa.horner_scheme)
            return r / d
        N, D = pFq_levin_coeffs(a, b, k, n=n, **params)
        assert len(N) == len(D) + n
    elif transform == "levin-m1":
        N, D = pFqm1_levin_coeffs(a, b, k, n=n, **params)
        assert len(N) == len(D)
    elif transform == "drummond":
        N, D = pFq_drummond_coeffs(a, b, k, n=n, **params)
        assert len(N) == len(D) + n
    elif transform == "drummond-m1":
        N, D = pFqm1_drummond_coeffs(a, b, k, n=n, **params)
        assert len(N) == len(D)
    elif transform.startswith("taylor"):
        taylor_series_with_zero = params.get("taylor_series_with_zero")
        taylor_rseries_with_zero = params.get("taylor_rseries_with_zero")
        taylor_rseries = params.get("taylor_rseries")
        taylor_series = params.get("taylor_series")
        if taylor_rseries_with_zero is not None and 1:
            rtC, z0 = taylor_rseries_with_zero
            return fpa.rpolynomial(ctx, z - z0, rtC, reverse=False) * (z - z0)
        if taylor_series_with_zero is not None and 1:
            tC, z0 = taylor_series_with_zero
            return fpa.fast_polynomial(ctx, z - z0, tC, reverse=False) * (z - z0)
        if taylor_series is not None and 1:
            return fpa.fast_polynomial(ctx, z, taylor_series, reverse=False)
        if taylor_rseries is not None:
            return fpa.rpolynomial(ctx, z, taylor_rseries, reverse=False)
        taylor_product = params.get("taylor_product")
        if taylor_product is not None:
            r = None
            for C in taylor_product:
                r1 = fpa.laurent(ctx, z, C, m=0, reverse=False, scheme=fpa.horner_scheme)
                if r is None:
                    r = r1
                else:
                    r = r * r1
            return r
        C = pFq_taylor_coeffs(a, b, k)
        if z0 is not None:
            z0 = ctx.constant(z0, z)
            dz = z - z0
            C = taylorat(C, z0, reverse=False)
        else:
            dz = z

        Clst = list(
            map(list, zip(*[fa.utils.number2expansion(dtype, c, length=expansion_length, functional=True) for c in C]))
        )
        Clst = [remove_leading_zeros(C_) for C_ in Clst]
        r = ctx.constant(0, z)
        for C_ in Clst:
            r += fpa.laurent(ctx, dz, C_, m=0, reverse=False, scheme=fpa.horner_scheme)

        return r
    else:
        assert 0, transform  # unreachable

    if z0 is not None:
        z0 = ctx.constant(z0, z)
        dz = z - z0
        D = taylorat(D, z0, reverse=True)
        N = taylorat(N, z0, reverse=True)
    else:
        dz = z

    N, D = normalize_rational_sequences(dtype, N, D, normalization_index=normalization_index)

    if issubclass(dtype, numpy.floating):
        Dlst = list(
            map(list, zip(*[fa.utils.number2expansion(dtype, c, length=expansion_length, functional=True) for c in D]))
        )
        Nlst = list(
            map(list, zip(*[fa.utils.number2expansion(dtype, c, length=expansion_length, functional=True) for c in N]))
        )

        Dlst = [remove_leading_zeros(D_) for D_ in Dlst]
        Nlst = [remove_leading_zeros(N_) for N_ in Nlst]

        zero_count = 0
        non_finite_count = 0
        count = 0
        for D_ in Dlst + Nlst:
            for c in D_:
                if c == 0:
                    zero_count += 1
                elif not numpy.isfinite(c):
                    non_finite_count += 1
                else:
                    count += 1
        assert non_finite_count == 0, (count, non_finite_count, zero_count)
    else:
        Dlst = [D]
        Nlst = [N]

    denom = ctx.constant(0, z)
    for D_ in Dlst[::-1]:
        D_ = [ctx.constant(c, z) for c in D_]
        t = fpa.laurent(ctx, dz, D_, m=m, reverse=True, scheme=fpa.horner_scheme)
        denom += t

    numer = ctx.constant(0, z)
    for N_ in Nlst[::-1]:
        N_ = [ctx.constant(c, z) for c in N_]
        numer += fpa.laurent(ctx, dz, N_, m=m, reverse=True, scheme=fpa.horner_scheme)

    # return numer / denom

    zeros = fa.utils.number2float(dtype, hyp0f1_zeros(1, end=5))
    # print(f'{zeros=}')
    zeros = fa.utils.number2fraction(zeros)

    # print(f'{N=}')
    N1, R1 = fpp.divmod(N, [1, -zeros[0]], reverse=True)
    N2, R2 = fpp.divmod(N1, [1, -zeros[1]], reverse=True)
    N3, R3 = fpp.divmod(N2, [1, -zeros[2]], reverse=True)

    N1 = fa.utils.number2float(dtype, N1)
    N2 = fa.utils.number2float(dtype, N2)
    N3 = fa.utils.number2float(dtype, N3)
    # print(f'{fa.utils.number2float(dtype, R1)=}')

    # print(f'{N=}')

    numer1 = fpa.laurent(ctx, dz, N1, m=m, reverse=True, scheme=fpa.horner_scheme)
    numer2 = fpa.laurent(ctx, dz, N2, m=m, reverse=True, scheme=fpa.horner_scheme)
    numer3 = fpa.laurent(ctx, dz, N3, m=m, reverse=True, scheme=fpa.horner_scheme)
    znumer1 = fpa.laurent(ctx, dz, [1, -zeros[0]], m=m, reverse=True, scheme=fpa.horner_scheme)
    znumer2 = fpa.laurent(ctx, dz, [1, -zeros[1]], m=m, reverse=True, scheme=fpa.horner_scheme)
    znumer3 = fpa.laurent(ctx, dz, [1, -zeros[2]], m=m, reverse=True, scheme=fpa.horner_scheme)

    # print(f'{numer1,denom=} {znumer1=} {numer1/denom * znumer1=}')
    # return znumer2 * znumer1 * numer2 / denom
    return znumer3 * znumer2 * znumer1 * numer3 / denom
    return numer1 / denom * znumer1
    numer = ctx.constant(0, z)
    for N_ in Nlst[::-1]:
        N_ = [ctx.constant(c, z) for c in N_]
        numer += fpa.laurent(ctx, dz, N_, m=m, reverse=True, scheme=fpa.horner_scheme)

    return numer / denom


def pFq_impl(
    ctx,
    dtype,
    a,
    b,
    z,
    k=None,
    m=None,
    n=None,
    normalization_index=None,
    transform=None,
    expansion_length=1,
    z0=None,
    **params,
):
    """Compute pFq(a, b, z) using rational approximation using a sequence transformation."""
    if k is None:
        # abs(z) <= 10000, n=0, gamma=3
        if z0 is not None:
            z_ = z - z0
        else:
            z_ = z
        if transform is None or transform.startswith("levin"):
            k = int(1 + 13 * abs(z_) ** 0.25)  # float64
            # k = int(1 + 12 * abs(z) ** 0.25)  # float32
            # k = int(1 + 11 * abs(z) ** 0.25)  # float16
        elif transform.startswith("drummond"):
            k = int(1 + 26 * abs(z_) ** 0.25)  # float64
            # k = int(1 + 24 * abs(z) ** 0.25)  # float16, float32
        elif transform.startswith("taylor"):
            k = int(1 + 29 * abs(z_) ** 0.25)  # float64
            # k = int(1 + 28 * abs(z) ** 0.25)  # float16, float32
        else:
            assert 0, transform  # unreachable
    if m is None:
        m = 0
    if n is None:
        if transform.endswith("-m1"):
            n = 1
        else:
            n = 0
    if transform == "levin" or transform is None:
        N, D = pFq_levin_coeffs(a, b, k, n=n, **params)
        assert len(N) == len(D) + n
    elif transform == "levin-m1":
        N, D = pFqm1_levin_coeffs(a, b, k, n=n, **params)
        assert len(N) == len(D)
    elif transform == "drummond":
        N, D = pFq_drummond_coeffs(a, b, k, n=n, **params)
        assert len(N) == len(D) + n
    elif transform == "drummond-m1":
        N, D = pFqm1_drummond_coeffs(a, b, k, n=n, **params)
        assert len(N) == len(D)
    elif transform.startswith("taylor"):
        C = pFq_taylor_coeffs(a, b, k)
        if z0 is not None:
            z0 = ctx.constant(z0, z)
            dz = z - z0
            C = taylorat(C, z0, reverse=False)
        else:
            dz = z
        Clst = list(
            map(list, zip(*[fa.utils.number2expansion(dtype, c, length=expansion_length, functional=True) for c in C]))
        )
        Clst = [remove_leading_zeros(C_) for C_ in Clst]
        r = ctx.constant(0, z)
        for C_ in Clst:
            r += fpa.laurent(ctx, dz, C_, m=0, reverse=False, scheme=fpa.horner_scheme)
        if transform == "taylor-m1":
            return r - ctx.constant(1, z)
        return r
    else:
        assert 0, transform  # unreachable

    flag = not True

    if z0 is not None:
        z0 = ctx.constant(z0, z)
        dz = z - z0
        D = taylorat(D, z0, reverse=True)
        if flag and transform.endswith("-m1"):
            N = N + [ctx.constant(0, z)] * n
        N = taylorat(N, z0, reverse=True)
    else:
        dz = z

    N, D = normalize_rational_sequences(dtype, N, D, normalization_index=normalization_index)

    if issubclass(dtype, numpy.floating):
        Dlst = list(
            map(list, zip(*[fa.utils.number2expansion(dtype, c, length=expansion_length, functional=True) for c in D]))
        )
        Nlst = list(
            map(list, zip(*[fa.utils.number2expansion(dtype, c, length=expansion_length, functional=True) for c in N]))
        )

        Dlst = [remove_leading_zeros(D_) for D_ in Dlst]
        Nlst = [remove_leading_zeros(N_) for N_ in Nlst]

        zero_count = 0
        non_finite_count = 0
        count = 0
        for D_ in Dlst + Nlst:
            for c in D_:
                if c == 0:
                    zero_count += 1
                elif not numpy.isfinite(c):
                    non_finite_count += 1
                else:
                    count += 1
        assert non_finite_count == 0, (count, non_finite_count, zero_count)
    else:
        Dlst = [D]
        Nlst = [N]

    denom = ctx.constant(0, z)
    for D_ in Dlst[::-1]:
        D_ = [ctx.constant(c, z) for c in D_]
        t = fpa.laurent(ctx, dz, D_, m=m, reverse=True, scheme=fpa.horner_scheme)
        denom += t

    numer = ctx.constant(0, z)
    for N_ in Nlst[::-1]:
        N_ = [ctx.constant(c, z) for c in N_]
        numer += fpa.laurent(ctx, dz, N_, m=m, reverse=True, scheme=fpa.horner_scheme)

    if transform.endswith("-m1"):
        if n == 0:
            return numer / denom - ctx.constant(1, z)
        if flag:
            f = ctx.constant(1, z)
        else:
            f = fpa.fast_exponent_by_squaring(ctx, z, n)
        if n == 1:
            return numer / denom * f
        C = pFq_taylor_coeffs(a, b, n - 1)
        C = [ctx.constant(c, z) for c in C[1:]]
        sn1 = fpa.laurent(ctx, z, C, m=1, reverse=False, scheme=fpa.horner_scheme)
        return sn1 + numer / denom * f
    else:
        return numer / denom
