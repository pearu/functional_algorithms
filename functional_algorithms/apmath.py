"""Arbitrary precision arithmetics.

References:

1. Douglas M. Priest. On Properties of Floating Point Arithmetics:
Numerical Stability and the Cost of Accurate Computations. PhD
Thesis. 1992.
https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4ba0df0935641e440c0853a55d1410bbe0cd459b

2. Yozo Hida, Xiaoye S. Li, David H. Bailey. Library for Double-Double
and Quad-Double Arithmetic. 2007.
https://web.mit.edu/tabbott/Public/quaddouble-debian/qd-2.3.4-old/docs/qd.pdf

3. Yozo Hida, Xiaoye S. Li David H. Bailey. Algorithms for Quad-Double
Precision Floating Point Arithmetic.
https://portal.nersc.gov/project/sparse/xiaoye-web/arith15.pdf

4. Jonathan Richard Shewchuk. Adaptive Precision Floating-Point
Arithmetic and Fast Robust Geometric Predicates. Discrete &
Computational Geometry 18(3):305–363, October 1997.
https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf

5. Jean-Michel Muller, Valentina Popescu, Ping Tak Peter Tang.  A new
multiplication algorithm for extended precision using floating-point
expansions.  ARITH23, Jul 2016, Santa Clara, United States.
ffhal-01298195 https://hal.science/hal-01298195

6. Sylvie Boldo, Mioara Joldes, Jean-Michel Muller, Valentina
Popescu. Formal Verification of a FloatingPoint Expansion
Renormalization Algorithm. 8th International Conference on Interactive
Theorem Proving (ITP’2017), Sep 2017, Brasilia, Brazil.
https://hal.science/hal-01512417

7. Mioara Joldes, Olivier Marty, Jean-Michel Muller, Valentina
Popescu. Arithmetic algorithms for extended precision using
floating-point expansions. IEEE Transactions on Computers, 2016, 65
(4), pp.1197 - 1210. ff10.1109/TC.2015.2441714ff. ffhal-01111551v2
https://hal.science/hal-01111551v2

8. Nicolas Brisebarre, Guillaume Hanrot, Jean-Michel Muller, Paul
Zimmermann. Correctly-rounded evaluation of a function: why, how, and
at what cost?. 2025. �hal-04474530v3� HAL Id: hal-04474530
https://hal.science/hal-04474530v3

"""

import numpy
import math
import fractions
import functools

import functional_algorithms as fa
from . import floating_point_algorithms as fpa
from . import generalized_hypergeometric_functions as ghf


def _merge(ctx, x, y, cmp):
    if not x:
        return y
    if not y:
        return x
    f = cmp(ctx, x[0], y[0])
    r1 = _merge(ctx, x[1:], y, cmp)
    r2 = _merge(ctx, x, y[1:], cmp)
    assert len(r1) == len(r2)
    lst = [ctx.select(f, x[0], y[0])]
    for a, b in zip(r1, r2):
        lst.append(ctx.select(f, a, b))
    return lst


def mergesort(ctx, lst, mth="<"):
    """Return a sorted list of expressions using mergesort method."""
    if len(lst) < 2:
        return lst
    n = len(lst) // 2
    if mth == "<":

        def cmp(ctx, x, y):
            return ctx.lt(x, y)

    elif mth == "a<a":

        def cmp(ctx, x, y):
            return ctx.lt(abs(x), abs(y))

    elif mth == ">":

        def cmp(ctx, x, y):
            return ctx.lt(y, x)

    elif mth == "a>a":

        def cmp(ctx, x, y):
            return ctx.lt(abs(y), abs(x))

    else:
        raise NotImplementedError(f"comparison method '{mth}'")
    return _merge(ctx, mergesort(ctx, lst[:n], mth=mth), mergesort(ctx, lst[n:], mth), cmp=cmp)


def two_sum(ctx, a, b):
    return fpa.add_2sum(ctx, a, b, fast=False)


def quick_two_sum(ctx, a, b):
    return fpa.add_2sum(ctx, a, b, fast=True)


def split(ctx, a):
    return fpa.split_veltkamp(ctx, a, scale=True)


@fpa.make_api()
def two_prod(ctx, dtype, a, b, scale=True):
    return fpa.mul_dekker(ctx, a, b, scale=scale, dtype=dtype)


@fpa.make_api()
def two_prod_mod4(ctx, dtype, a, b, scale=True):
    hi, lo = fpa.mul_dekker(ctx, a, b, scale=scale, dtype=dtype)
    qh = quotient4(ctx, hi, dtype=dtype)
    ql = quotient4(ctx, lo, dtype=dtype)
    return hi - qh, lo - ql


def vecsum(ctx, seq, fast=False):
    """ """
    s = seq[-1]
    e_lst = []
    for i in reversed(range(len(seq) - 1)):
        if fast:
            s, e = quick_two_sum(ctx, seq[i], s)
        else:
            s, e = two_sum(ctx, seq[i], s)
        e_lst.insert(0, e)
    e_lst.insert(0, s)
    return e_lst


def vecsumerr(ctx, seq, fast=False):
    """ """
    g_lst = []
    s = seq[0]
    for i in range(len(seq) - 1):
        if fast:
            g_i, s = quick_two_sum(ctx, s, seq[i + 1])
        else:
            g_i, s = two_sum(ctx, s, seq[i + 1])
        g_lst.append(g_i)
    g_lst.append(s)
    return g_lst


def renormalize(ctx, seq, functional=False, fast=False, size=None, dtype=None):
    """Convert a list of possibly overlapping items to a list of
    non-overlapping items.

    The renormalization algorithm originates from Priest's PhD thesis
    (1992) and there exists several modifications of it. Here we'll
    use the renormalization algorithm from
    https://hal.science/hal-01512417 that is provided with a formal
    proof and is more efficient as it proved that that the third step
    of the algorithm in ffhal-01111551v2 is unnecessary.

    Absolute values of the input must be a decreasing sequence,
    excluding zero items. If the input does not satisfy this criteria,
    the output will be inaccurate when fast is True. If fast is False,
    the output may have overlapping items. If so, calling renormalize
    twice is required.

    If functional is True, a functional algorithm of the
    renormalization is returned, that is, the input items can be
    symbolic expressions.

    The length of the output is equal or less than the length of the
    input. If functional is True, the lengths of input and output are
    equal and output may contain zero items.

    The length of the output will not exceed size when specified.

    When dtype is specified, the length of the output will not exceed
    the maximal size of expansion that a given floating-point system
    enables.
    """
    # VecSum:
    e_lst = vecsum(ctx, seq, fast=fast)
    # VecSumErrBranch:
    if functional:
        zero = ctx.constant(0, seq[0])

    f_lst = []
    eps_i = e_lst[0]
    for i in range(len(seq) - 1):
        if fast:
            f_j, eps_ip1 = quick_two_sum(ctx, eps_i, e_lst[i + 1])
        else:
            f_j, eps_ip1 = two_sum(ctx, eps_i, e_lst[i + 1])
        if functional:
            p = ctx.ne(eps_ip1, zero)
            f_lst.append(ctx.select(p, f_j, zero))
            eps_i = ctx.select(p, eps_ip1, f_j)
        elif ctx._is_nonzero(eps_ip1):
            f_lst.append(f_j)
            eps_i = eps_ip1
        else:
            eps_i = f_j
    if functional:
        p = ctx.ne(eps_i, zero)
        f_lst.append(ctx.select(p, eps_i, zero))
        # TODO: replace it with f_lst.append(eps_i) ?
    elif ctx._is_nonzero(eps_i):
        f_lst.append(eps_i)

    if dtype is not None:
        # max_size == (fi.maxexp - fi.minexp - fi.machep) // (-fi.negep - 1)
        max_size = {numpy.float16: 4, numpy.float32: 12, numpy.float64: 40}[dtype]
        if size is None:
            size = max_size
        else:
            size = min(size, max_size)

    if functional:
        if 0:
            f_lst = move_zeros_to_right(ctx, f_lst, size=size)
        else:
            return nztopk(ctx, f_lst, size if size is not None else len(f_lst))
    if size is not None:
        f_lst = f_lst[:size]
        assert len(f_lst) <= size or not functional, (len(f_lst), size)
    return f_lst


def nztopk(ctx, seq, k):
    """Return top k non-zero elements of a sequence.

    Using a functional friendly algorithm.
    """
    if k == 0 or len(seq) == 0:
        return []
    elif len(seq) == 1:
        return seq
    elif len(seq) == 2:
        flag = seq[0] == ctx.constant(0, seq[0])
        if k == 1:
            return [ctx.select(flag, seq[1], seq[0])]
        return [ctx.select(flag, seq[1], seq[0]), ctx.select(flag, seq[0], seq[1])]

    zero = ctx.constant(0, seq[0])
    izero = ctx.constant(0)
    ione = ctx.constant(1, izero)

    isnzero = [a != zero for a in seq]
    nzcount = [izero]
    for b in isnzero[:-1]:
        nzcount.append(nzcount[-1] + ctx.select(b, ione, izero))

    result = []
    for i in range(min(k, len(seq))):
        lst = [ctx.select(ctx.logical_and(isnzero[j], nzcount[j] == i), seq[j], zero) for j in range(i, len(seq))]
        result.append(sum(lst[:-1], lst[-1]))

    return result


def move_zeros_to_right(ctx, seq, size=None):
    """Move zeros in a sequence of unequal items to the right end.

    Using a functional friendly algorithm.
    """
    if size == 0 or len(seq) == 0:
        return []
    elif len(seq) == 1:
        return seq
    elif len(seq) == 2:
        flag = seq[0] == ctx.constant(0)
        if size == 1:
            return [ctx.select(flag, seq[1], seq[0])]
        result = [ctx.select(flag, seq[1], seq[0]), ctx.select(flag, seq[0], seq[1])]
    else:
        zero = ctx.constant(0)
        first = seq[-1]
        for i in reversed(range(len(seq) - 1)):
            first = ctx.select(seq[i] != zero, seq[i], first)

        if size == 1:
            return [first]

        rest = []
        for i in range(1, len(seq)):
            item = seq[i - 1]
            for j in reversed(range(i)):
                item = ctx.select(seq[j] == first, seq[i], item)
            rest.append(item)
        result = [first] + move_zeros_to_right(ctx, rest, size=size - 1 if size is not None else None)
    return result[:size] if size is not None else result


def negate(ctx, seq):
    """Negate an FP expansion."""
    return [-x for x in seq]


@fpa.make_api(mp_func=lambda mp_ctx, x, y: x + y)
def add(ctx, dtype, seq1, seq2, functional=False, fast=False, size=None):
    """Add two FP expansions."""
    return renormalize(ctx, seq1 + seq2, functional=functional, fast=fast, size=size, dtype=dtype)


@fpa.make_api(mp_func=lambda mp_ctx, x, y: x - y)
def subtract(ctx, dtype, seq1, seq2, functional=False, fast=False, size=None):
    """Subtract two FP expansions."""
    return renormalize(ctx, seq1 + negate(ctx, seq2), functional=functional, fast=fast, size=size, dtype=dtype)


@fpa.make_api(mp_func=lambda mp_ctx, x, y: x * y)
def multiply(ctx, dtype, seq1, seq2, functional=False, fast=False, size=None, base=None, scale=True):
    """Multiply two FP expansions.

    Warning: when fast=True, the result will be very likely inaccurate.
    """
    if not (functional or (seq1 and seq2)):
        return []
    base_lst = [ctx.constant(base**i2) for i2 in range(len(seq2))] if base is not None else None
    r_0, e_0 = two_prod(ctx, seq1[0], seq2[0], dtype=dtype)
    r_lst = [r_0]
    e_lst = [e_0]
    for n in range(1, len(seq1) + len(seq2)):
        p_lst = []
        ne_lst = []
        for i1 in range(len(seq1)):
            i2 = n - i1
            if i2 >= 0 and i2 < len(seq2):
                if base is None:
                    p_i, e_i = two_prod(ctx, seq1[i1], seq2[i2], scale=scale, dtype=dtype)
                else:
                    p_i, e_i = two_prod(ctx, seq1[i1] / base_lst[i2], seq2[i2], scale=scale, dtype=dtype)
                p_lst.append(p_i)
                ne_lst.append(e_i)
        lst = vecsum(ctx, p_lst + e_lst, fast=fast)
        r_lst.append(lst[0])
        e_lst = lst[1:] + ne_lst
    lst = r_lst + e_lst
    return renormalize(ctx, lst, functional=functional, fast=fast, size=size, dtype=dtype)


@fpa.make_api(mp_func=None)
def multiply_mod4(ctx, dtype, seq1, seq2, functional=False, fast=False, size=None, base=None, scale=True):
    """Multiply two FP expansions modulo 4.

    Warning: when fast=True, the result will be very likely inaccurate.

    If base is specified then seq2 represents a scaled FP expansion of y:

      y = sum(c / base ** i for i, c in enumerate(seq2))

    Use base that is a power of two to minimize numerical errors.
    """
    base_lst = [ctx.constant(base**i2) for i2 in range(len(seq2))] if base is not None else None
    r_0, e_0 = two_prod_mod4(ctx, seq1[0], seq2[0], dtype=dtype)
    r_lst = [r_0]
    e_lst = [e_0]
    for n in range(1, len(seq1) + len(seq2)):
        p_lst = []
        ne_lst = []
        for i1 in range(len(seq1)):
            for i2 in range(len(seq2)):
                if i1 + i2 == n:
                    if base_lst is None:
                        p_i, e_i = two_prod_mod4(ctx, seq1[i1], seq2[i2], dtype=dtype)
                    else:
                        p_i, e_i = two_prod_mod4(ctx, seq1[i1] / base_lst[i2], seq2[i2], scale=scale, dtype=dtype)
                    p_lst.append(p_i)
                    ne_lst.append(e_i)
        lst = vecsum(ctx, p_lst + e_lst, fast=fast)
        r_lst.append(lst[0])
        e_lst = lst[1:] + ne_lst
    lst = r_lst + e_lst
    result = renormalize(ctx, lst, functional=functional, fast=fast, size=size, dtype=dtype)

    # ensure that abs(result[0]) <= 2
    result = add(
        ctx, result, [-quotient4(ctx, result[0], dtype=dtype)], functional=functional, fast=fast, size=size, dtype=dtype
    )

    return result


@fpa.make_api(mp_func=None)
def quotient4(ctx, dtype, x):
    """Return quotient of module 4 such that

    x = q + r

    where abs(r) <= 2 and q is an integer with multiple of 4.
    """
    two = fpa.next(ctx, ctx.constant(2), up=False, dtype=dtype)
    four = ctx.constant(4)
    q = ctx.floor((abs(x) + two) / four) * four
    q = ctx.select(x < 0, -q, q)
    return q


@fpa.make_api(mp_func=lambda mp_ctx, x: x * x)
def square(ctx, dtype, seq, functional=False, fast=False, size=None, scale=True):
    """Square of an FP expansion."""
    r_0, e_0 = two_prod(ctx, seq[0], seq[0], dtype=dtype)
    r_lst = [r_0]
    e_lst = [e_0]
    for n in range(1, len(seq) * 2):
        p_lst = []
        ne_lst = []
        for i1 in range(len(seq)):
            i2 = n - i1
            if i1 > i2 or i2 >= len(seq):
                continue
            elif i1 < i2:
                p_i, e_i = two_prod(ctx, seq[i1], seq[i2], scale=scale, dtype=dtype)
                p_i += p_i
                e_i += e_i
            else:
                p_i, e_i = two_prod(ctx, seq[i1], seq[i1], scale=scale, dtype=dtype)
            p_lst.append(p_i)
            ne_lst.append(e_i)
        lst = vecsum(ctx, p_lst + e_lst, fast=fast)
        r_lst.append(lst[0])
        e_lst = lst[1:] + ne_lst
    lst = r_lst + e_lst
    return renormalize(ctx, lst, functional=functional, fast=fast, size=size, dtype=dtype)


@fpa.make_api(mp_func=lambda mp_ctx, x: 1 / x)
def reciprocal(ctx, dtype, seq, functional=False, size=None, niter=None):
    """Reciprocal of an FP expansion."""
    if size is None:
        size = len(seq)
    q = max(len(seq), size).bit_length() - 1
    if niter is None:
        niter = q + 1
    two = ctx.constant(2)
    x_lst = [ctx.reciprocal(seq[0])]
    for i in range(niter):
        v_lst = multiply(ctx, x_lst, seq[: 2 ** (i + 1)], functional=functional, size=2 ** (i + 1), dtype=dtype)
        if v_lst:
            w_lst = subtract(ctx, [two], v_lst, functional=functional, size=2 ** (i + 1), dtype=dtype)
        else:
            w_lst = [two]
        x_lst = multiply(ctx, x_lst, w_lst, functional=functional, size=2 ** (i + 1), dtype=dtype)

    if functional:
        x_lst = x_lst[:size]

    if functional:
        assert len(x_lst) == size, (len(x_lst), size)
        return x_lst

    return x_lst[:size]


@fpa.make_api(mp_func=lambda mp_ctx, x, n, d: (1 / x ** (n / d) if x else x + mp_ctx.inf))
def rpower_fraction(ctx, dtype, seq, n, d, functional=False, size=None, niter=None):
    """Reciprocal power of an FP expansion.

    Returns x ** (- n / d).

    Algorithm
    ---------

    First, eliminate common factors in n and d.

    f(x) = 1 / x ** d - a ** n
    f'(x) = -d / x ** (d + 1)

    x_(i + 1) = x_{i} - f(x_{i}) / f'(y_{i})
              = x_{i} * (d + 1 - a ** n * x ** d) / d
              = x_{i} * (d + 1 - (a * x) ** n * x ** (d - n)) / d  if n < d
              = x_{i} * (d + 1 - (a * x) ** d * a ** (n - d)) / d  if n > d
    """
    assert isinstance(n, int) and n > 0
    assert isinstance(d, int) and d > 0
    assert n < d, (n, d)
    cf = int(numpy.gcd(n, d))
    n //= cf
    d //= cf

    c = ctx.constant(d + 1)
    f = ctx.constant(1 / d)
    nd = ctx.constant(n / d)
    if size is None:
        size = len(seq)
    if niter is None:
        # For better accuracy, niter is here one larger than suggested
        # in https://perso.ens-lyon.fr/jean-michel.muller/07118139.pdf
        # for n / d == 1 / 2.
        niter = max(size, len(seq)).bit_length()

    x_lst = [ctx.exp(-nd * ctx.log(seq[0]))]
    for i in range(niter):
        xpq_lst = power(ctx, x_lst, d - n, functional=functional, size=2 ** (i + 1), dtype=dtype)
        assert len(xpq_lst) <= 2 ** (i + 1)
        v_lst = multiply(ctx, x_lst, seq[: 2 ** (i + 1)], functional=functional, size=2 ** (i + 1), dtype=dtype)
        v_lst = power(ctx, v_lst, n, functional=functional, size=2 ** (i + 1), dtype=dtype)
        assert len(v_lst) <= 2 ** (i + 1)
        if v_lst:
            w_lst = multiply(ctx, xpq_lst, v_lst, functional=functional, size=2 ** (i + 1), dtype=dtype)
            y_lst = subtract(ctx, [c], w_lst, functional=functional, size=2 ** (i + 1), dtype=dtype)
        else:
            w_lst = []
            y_lst = [c]
        z_lst = multiply(ctx, x_lst, y_lst, functional=functional, size=2 ** (i + 1), dtype=dtype)

        # Warning: assume d is power of two!
        assert d % 2 == 0, d
        x_lst = [z * f for z in z_lst]

    x_lst = x_lst[:size]
    if functional:
        assert len(x_lst) <= size, (len(x_lst), size)

    return x_lst


@fpa.make_api(mp_func=lambda mp_ctx, x, n, d: x ** (n / d))
def power_fraction(ctx, dtype, seq, n, d, functional=False, size=None, niter=None):
    """Reciprocal power of an FP expansion.

    Returns x ** (n / d).

    Algorithm
    ---------

    x ** (n / d) = x ** (-1 + n / d + 1)
                 = x * x ** (-(d - n) / d)
    """
    # TODO: # reduced size may give a better accuracy, see sqrt
    r = rpower_fraction(ctx, dtype, seq, d - n, d, functional=functional, size=size, niter=niter, dtype=dtype)
    return multiply(ctx, seq, r, functional=functional, size=size, dtype=dtype)


@fpa.make_api(mp_func=lambda mp_ctx, x: (1 / mp_ctx.sqrt(x) if x else x + mp_ctx.inf))
def rsqrt(ctx, dtype, seq, functional=False, size=None, niter=None):
    """Reciprocal square root of an FP expansion.

    Algorithm
    ---------

    f(y) = 1 / y **2 - x
    f'(y) = -2 / y ** 3

    y_(i + 1) = y_{i} - f(y_{i}) / f'(y_{i})
              = y_{i} + (1 / y_{i} ** 2 - x) y_{i} ** 3 / 2
              = y_{i} * (2 + 1 - x * y_{i} ** 2) / 2
              = y_{i} * (3 - x * y_{i} ** 2) / 2

    converges to y = 1 / sqrt(x)
    """
    return rpower_fraction(ctx, seq, 1, 2, functional=functional, size=size, dtype=dtype)


@fpa.make_api(mp_func=lambda mp_ctx, x: mp_ctx.sqrt(x))
def sqrt(ctx, dtype, seq, functional=False, size=None):
    """Square root of an FP expansion."""

    sizem1 = max(1, size - 1) if size is not None else None  # reduced size gives better accuracy
    rr = rsqrt(ctx, seq, functional=functional, size=size, dtype=dtype)
    r = multiply(ctx, seq, rr, functional=functional, size=size, dtype=dtype)

    # Correction from Eq 8 in https://www.davidhbailey.com/dhbpapers/mpfun2020.pdf:
    #   rr = rsqrt(seq)
    #   r = seq * rr
    #   sqrt(seq) = r + 1/2 * (seq - r ** 2) * rr
    #
    # This correction is useful only when r ** 2 does not overflow. To
    # avoid the overflow, we rewrite the correction as follows:
    #   sqrt(seq) = r + 1/2 * (1 - r * rr) * r
    #             = 1/2 * (3 - r * rr) * r
    #
    if 1:
        # sqrt(seq) = r + 1/2 * (1 - r * rr) * r
        # sqrt(seq) = 1/2 * (3 - r * rr) * r
        half = ctx.constant(0.5)
        r2 = multiply(ctx, r, rr, functional=functional, size=sizem1, dtype=dtype)
        a = subtract(ctx, [ctx.constant(3)], r2, functional=functional, size=sizem1, dtype=dtype)
        a = [a_ * half for a_ in a]
        r = multiply(ctx, a, r, functional=functional, size=size, dtype=dtype)
    else:
        # sqrt(seq) = r + 1/2 * (seq - r ** 2) * rr
        half = ctx.constant(0.5)
        r2 = square(ctx, r, functional=functional, size=sizem1, dtype=dtype)
        rr = [r_ * half for r_ in rr]
        a = subtract(ctx, seq, r2, functional=functional, size=sizem1, dtype=dtype)
        b = multiply(ctx, rr, a, functional=functional, size=sizem1, dtype=dtype)
        r1 = add(ctx, r, b, functional=functional, size=size, dtype=dtype)
        limit = 63460  # upper limit where squaring above does not overflow, specific to float16
        r = where(ctx, seq[0] < limit, r1, r)
    return r


@fpa.make_api(mp_func=lambda mp_ctx, x, n: x**n)
def power(ctx, dtype, seq, n, functional=False, size=None, scale=True):
    """
    n-th power of an FP expansion.
    """
    if isinstance(n, list):
        if not [n_ for n_ in n[1:] if n_ != 0]:
            n = n[0]
    if n == 0:
        return [ctx.constant(1)]
    elif n == 1:
        assert size is None or len(seq) <= size, (size, len(seq))  # not impl
        return seq
    elif n == 2:
        return square(ctx, seq, functional=functional, size=size, scale=scale, dtype=dtype)
    elif isinstance(n, int):
        if n > 0:
            r = power(ctx, seq, n // 2, functional=functional, size=size, scale=scale, dtype=dtype)
            sq = square(ctx, r, functional=functional, size=size, scale=scale, dtype=dtype)
            if n % 2 == 0:
                return sq
            return multiply(ctx, sq, seq, functional=functional, size=size, scale=scale, dtype=dtype)
        else:
            rseq = reciprocal(ctx, seq, functional=functional, size=size, dtype=dtype)
            return power(ctx, rseq, n, functional=functional, size=size, dtype=dtype)
    elif n == 0.5:
        return sqrt(ctx, seq, functional=functional, size=size, dtype=dtype)
    elif n == -0.5:
        return rsqrt(ctx, seq, functional=functional, size=size, dtype=dtype)
    elif isinstance(n, fractions.Fraction):
        numer, denom = n.numerator, n.denominator
        if max(abs(numer), denom) < 5:
            if numer < 0:
                return rpower_fraction(ctx, seq, -numer, denom, functional=functional, size=size, dtype=dtype)
            return power_fraction(ctx, seq, numer, denom, functional=functional, size=size, dtype=dtype)
        n = numer / denom

    if isinstance(n, (float, numpy.floating)):
        n = [n]
    elif isinstance(n, list):
        # remove trailing zeros:
        while n and n[-1] == 0:
            n = n[:-1]
    else:
        raise TypeError(f"power exponent argument must be a number or a list of numbers, got {type(n)}")

    lseq = logarithm(ctx, seq, functional=functional, size=size, dtype=dtype)
    m = multiply(ctx, lseq, n, functional=functional, size=size, dtype=dtype)
    return exponential(ctx, m, functional=functional, size=size, dtype=dtype)


@fpa.make_api(mp_func=None)
def hypergeometric(ctx, dtype, a, b, seq, niter, functional=False, size=None):
    """
    Generalized hypergeometic series on an FP expansion:

      sum(prod((a[i])_n, i=0..p) / prod((b[i])_n), i=0..q * z**n / n!, n=0,1,...,niter-1)

    where
      p = len(a) - 1
      p = len(b) - 1
      (k)_n = 1 if n == 0 else (k)_{n-1} * (k + n - 1)
      n! = 1 if n == 0 else (n-1)! * n
      a and b are lists of integers or Fraction instances.
    """
    r = hypergeometric_minus_one(ctx, a, b, seq, niter, functional=functional, size=size, dtype=dtype)
    return add(ctx, [ctx.constant(1)], r, functional=functional, size=size, dtype=dtype)


@fpa.make_api(mp_func=None)
def hypergeometric_minus_one(ctx, dtype, a, b, seq, niter, functional=False, size=None):
    """
    Generalized hypergeometic series on an FP expansion minus one:

      sum(prod((a[i])_n, i=0..p) / prod((b[i])_n), i=0..q * z**n / n!, n=1,...,niter-1)

    where
      p = len(a) - 1
      p = len(b) - 1
      (k)_n = 1 if n == 0 else (k)_{n-1} * (k + n - 1)
      n! = 1 if n == 0 else (n-1)! * n
      a and b are lists of integers or Fraction instances.
    """
    import functional_algorithms as fa

    rcoeffs = []
    for n in range(1, niter):
        numer_ = 1
        denom_ = 1
        for a_ in a:
            numer_ *= a_ + n - 1
        for b_ in b:
            denom_ *= b_ + n - 1
        denom_ *= n

        if not numer_:
            break

        rc = fa.utils.fraction2expansion(dtype, fractions.Fraction(numer_, denom_))
        if not rc:
            break
        rcoeffs.append(renormalize(ctx, rc, functional=False, dtype=dtype))

    # hypergeometric series evaluation using Horner' scheme as
    #
    #   sum(c[n] * z ** n, n=0..niter-1)
    #   = c[0] + c[1] * z + c[2] * z ** 2 + c[3] * z ** 3 + ...
    #   = c[0] + (c[1] + (c[2] + (c[3] + ...) * z) * z) * z
    #   = fma(fma(fma(fma(..., z, c[3]), z, c[2]), z, c[1]), z, c[0])
    #
    # is inaccurate because c[n] is a rapidly decreasing sequence and
    # the given dtype may not have enough exponent range to represent
    # very small coefficients.
    #
    # In the following, we'll use the property of geometric series that
    # the ratio of neighboring coefficients is a rational number so
    # that we have
    #
    #   c[n] * z ** n == (c[n-1] * z ** (n-1)) * z * R(n)
    #   c[0] = 1
    #
    # Hence
    #
    #   sum(c[n] * z ** n, n=0..niter-1)
    #   = c[0] + c[0] * z * R(1) + c[0] * z * R(1) * z * R(2) + ...
    #   = c[0] * (1 + z * R(1) * (1 + z * R(2) * (1 + z * R(3) * (1 + ...))))
    #   = 1 + z * R(1) * (1 + z * R(2) * (1 + z * R(3) * (1 + ...)))
    #
    # where R(n) is a slowly varying sequence in n. For instance, for
    # float16 hypergeometric([], [], z), R(n) = 1 / n is nonzero for n
    # over 2 ** 24, that is, the maximal value for user-specified
    # niter is practically unlimited.
    def rhorner(rcoeffs, z):
        r = multiply(ctx, rcoeffs[0], z, functional=functional, size=size, dtype=dtype) or [dtype(0)]
        if len(rcoeffs) > 1:
            h = add(ctx, [dtype(1)], rhorner(rcoeffs[1:], z), functional=functional, size=size, dtype=dtype)
            r = multiply(ctx, r, h, functional=functional, size=size, dtype=dtype)
        return r

    return rhorner(rcoeffs, seq)


@functools.lru_cache(typed=True)
def hypergeometric0f1_asymptotic_parameters(dtype, b, max_k=20, prec=110, size=2):
    """Return rC and rS such that

    JC(z) = rpolynomial(1 / z / 4, rC) * (b - 1)! * sqrt(pi)
    JS(z) = rpolynomial(1 / z / 4, rS) * (b - 1)! * sqrt(pi) * 2
    """
    import mpmath
    import functional_algorithms.generalized_hypergeometric_functions as ghf

    mp_ctx = mpmath.mp

    with mp_ctx.workprec(prec):
        a = b - 1
        half = fractions.Fraction(1, 2)
        quarter = fractions.Fraction(1, 4)
        # The Taylor coefficients of
        #  hyper([1/2 - a, 1/2 + a], [], -1/2 * 1/z)
        # are the Taylor coefficients of JC(I * z) and JS(I * z)
        T = ghf.pFq_taylor_coeffs([half - a, half + a], [], max_k * 2 + 1, c=-half)
        sqpigb = fa.utils.number2fraction(mp_ctx.sqrt(mp_ctx.pi) * mp_ctx.gamma(b))
        T = [c / sqpigb for c in T]
        C, S = [c * (-quarter) ** i for i, c in enumerate(T[::2])], [c * (-quarter) ** i / 2 for i, c in enumerate(T[1::2])]

        if isinstance(b, (int, fractions.Fraction)):
            nbp4 = -fractions.Fraction(2 * b - 1, 4)
            nbm4 = -fractions.Fraction(2 * b + 1, 4)
        else:
            b = mp_ctx.mpf(b)
            nbp4 = -(2 * b - 1) / 4
            nbm4 = -(2 * b + 1) / 4

        snb = mp_ctx.sin(-mp_ctx.pi * nbp4)
        csb = mp_ctx.cos(mp_ctx.pi * nbp4)

    rC = fa.polynomial.asrpolynomial(C, reverse=False)
    rS = fa.polynomial.asrpolynomial(S, reverse=False)

    rC = fa.utils.number2expansion(dtype, rC, length=size, functional=True, base=None)
    rS = fa.utils.number2expansion(dtype, rS, length=size, functional=True, base=None)

    snb = fa.utils.number2expansion(dtype, snb, length=size, functional=True, base=None)
    csb = fa.utils.number2expansion(dtype, csb, length=size, functional=True, base=None)

    if not isinstance(nbp4, fractions.Fraction):
        nbp4 = fa.utils.number2expansion(dtype, nbp4, length=size, functional=True, base=None)
        nbm4 = fa.utils.number2expansion(dtype, nbm4, length=size, functional=True, base=None)

    return dict(rC=rC, rS=rS, snb=snb, csb=csb, nbp4=nbp4, nbm4=nbm4)


@functools.lru_cache(typed=True)
def log_of_two(ctx, dtype, size=None, base=None):
    """Return FP expansion of log(2)."""
    import mpmath

    if size is None:
        size = 2

    fi = numpy.finfo(dtype)
    # max_prec = {numpy.float16: 21, numpy.float32: 140, numpy.float64: 1055}[dtype]
    max_prec = -fi.negep * size
    mp_ctx = mpmath.mp
    with mp_ctx.workprec(max_prec):
        return fa.utils.mpf2expansion(dtype, mp_ctx.log(2), length=size, functional=True, base=base)


@functools.lru_cache(typed=True)
def reciprocal_log_of_two(ctx, dtype, size=None, base=None):
    """Return FP expansion of log(2)."""
    import mpmath

    if size is None:
        size = 2

    fi = numpy.finfo(dtype)
    # max_prec = {numpy.float16: 24, numpy.float32: 138, numpy.float64: 1052}[dtype]
    max_prec = -fi.negep * size + 2
    mp_ctx = mpmath.mp
    with mp_ctx.workprec(max_prec):
        return fa.utils.mpf2expansion(dtype, 1 / mp_ctx.log(2), length=size, functional=True, base=base)


@functools.lru_cache(typed=True)
def _argument_reduction_exponential_parameters(ctx, dtype, size=None, base=None):
    import mpmath

    fi = numpy.finfo(dtype)
    rlog2_size = 3
    rlog2 = reciprocal_log_of_two(ctx, dtype, size=rlog2_size, base=base)
    log2 = log_of_two(ctx, dtype, size=size + 1)
    prec = -fi.negep
    limit = dtype(0.7 * 2**prec)

    max_prec = -fi.negep * size + 2
    mp_ctx = mpmath.mp
    with mp_ctx.workprec(max_prec):
        largest_log2 = fa.utils.mpf2float(dtype, mp_ctx.log(2) * fa.utils.float2mpf(mp_ctx, fi.max))
        smallest_integer_log2 = fa.utils.mpf2float(dtype, mp_ctx.log(2) * 2 ** (prec - 1))

    return dict(
        rlog2=rlog2,
        log2=log2,
        limit=limit,
        largest=fi.max,
        largest_log2=largest_log2,
        smallest_integer_log2=smallest_integer_log2,
    )


@fpa.make_api(mp_func=None)
def argument_reduction_exponential(ctx, dtype, seq, size=None, functional=False, base=None, scale=True):
    """Return k, rseq such that

    x = k * log(2) + sum(rseq)

    where seq is FP expansion of x, k is integral, abs(sum(rseq)) <=
    0.5005 when abs(x) < log(2) * largest.

    Let's assume x / log(2) > 2 ** (prec - 1). Then x / log(2) is
    integer. So, k = x / log(2) and sum(rseq) is 0. However, if x /
    log(2) > largest, k cannot be represented as a finite value. Then
    we'll set k = largest (which is also an integer) and sum(rseq) = x
    - largest * log(2).
    """
    zero = ctx.constant(0)
    params = _argument_reduction_exponential_parameters(ctx, dtype, size=size, base=base)
    rlog2 = params["rlog2"]
    log2 = params["log2"]

    y = multiply(ctx, seq, rlog2, size=1, functional=functional, base=base, scale=scale, dtype=dtype)[0]
    huge = abs(seq[0]) >= params["largest_log2"]
    large = abs(seq[0]) >= params["smallest_integer_log2"]

    sign = ctx.select(seq[0] < 0, ctx.constant(-1), ctx.constant(1))
    k = ctx.select(huge, sign * params["largest"], ctx.select(large, seq[0] * rlog2[0], ctx.trunc(y)))
    kcorr = ctx.select(large, dtype(0), ctx.trunc((y - k) * rlog2[0] + ctx.constant(0.5)))
    k = k + kcorr
    # scale = True is required for moderately large abs(x) that is
    # smaller than log(2) ** 2 ** (prec - 1) but still large enough causing
    # overflow in k * log(2).
    nklog2 = multiply(ctx, log2, [-k], size=size + 1, functional=functional, scale=True, dtype=dtype)
    rseq = add(ctx, seq, nklog2, size=size, functional=functional, dtype=dtype)
    rseq_huge = seq[0] - sign * params["largest_log2"]
    rseq = [ctx.select(huge, rseq_huge if i == 0 else zero, ctx.select(large, zero, c)) for i, c in enumerate(rseq)]
    return k, rseq


@functools.lru_cache(typed=True)
def _exponential_parameters(dtype, functional=False, size=None, base=None):
    if size is None:
        size = 2

    fi = numpy.finfo(dtype)
    # prec = -fi.negep
    # q = int(round(prec ** (2 / 5)))  # see exponential doc
    """
1000000 samples:
  float16, q=0, n=6: ULP diff 1 == 227
  float16, q=0, n=7: ULP diff 1 == 89
  float16, q=1, n=5: ULP diff 1 == 144
  float16, q=1, n=6: ULP diff 1 == 33
  float16, q=1, n=7: ULP diff 1 == 38
  float16, q=2, n=5: ULP diff 1 == 44
  float16, q=2,3, n=6: ULP diff 1 == 36
  float16, q=2, n=7: ULP diff 1 == 36

  float32, q=2, n=7: ULP diff 1 == 3040
  float32, q=2, n=8: ULP diff 1 == 2978
  float32, q=2, n=9,10: ULP diff 1 == 2976
  float32, q=3, n=7: ULP diff 0 == 995043
  float32, q=3, n=8: ULP diff 0 == 995042

  float64, q=16, n=6: ULP diff 1 == 0
  float64, q=2, n=8: ULP diff 0 == 995211
  float64, q=3, n=8: ULP diff 0 == 996766
  float64, q=4, n=8: ULP diff 0 == 998608
  float64, q=5, n=8: ULP diff 0 == 999946
  float64, q=6, n=8: ULP diff 0 == 1000001
  float64, q=7, n=8: ULP diff 0 == 1000001
  float64, q=8, n=8: ULP diff 0 == 1000001
  float64, q=6, n=7: ULP diff 0 == 999637
  float64, q=7, n=7: ULP diff 0 == 999995
  float64, q=8, n=7: ULP diff 0 == 1000001
  float64, q=8, n=6: ULP diff 0 == 999730
    """
    q, n = {numpy.float16: (1, 6), numpy.float32: (2, 8), numpy.float64: (6, 8)}[dtype]
    C = [fractions.Fraction(1)]
    for i in range(1, n):
        c = fractions.Fraction(1, math.factorial(i))
        if base is not None:
            c *= base**i
        C.append(c)
    rC = fa.polynomial.asrpolynomial(C, reverse=False)

    rC = fa.utils.number2expansion(dtype, rC, length=size, functional=functional)
    C = fa.utils.number2expansion(dtype, C, length=size, functional=functional)
    return dict(rC=rC, q=q, C=C, log_largest=numpy.log(fi.max), log_smallest=numpy.log(fi.smallest_subnormal))


@fpa.make_api(mp_func=lambda mp_ctx, *args: mp_ctx.exp(*args))
def exponential(ctx, dtype, seq, functional=False, size=None, scale=False):
    """Return FP expansion of exp(x)

    Algorithm
    ---------

    exp(x) = exp(k * log(2) + r) = exp2(k) * exp(r)

    where abs(r) < log(2) * 0.6, k is integral, and exp(r) is
    evaluated using Taylor series. For faster convergence, we'll use

      exp(r * 2 ** -q * 2 ** q) = exp(r / 2 ** q) ** (2 ** q)

    where q = round(prec ** (2 / 5))
    [https://www.davidhbailey.com/dhbpapers/mpfun2020.pdf] but we'll
    use heuristics that leads to more accurate results [see
    exponential_parameters].
    """
    zero = ctx.constant(0)
    inf = ctx.constant("inf")

    base = None
    params = _exponential_parameters(dtype, functional=functional, size=1, base=base)
    q = params["q"]

    k, rseq = argument_reduction_exponential(ctx, seq, functional=functional, size=size, scale=scale, dtype=dtype)
    if q:
        q2 = ctx.constant(2**q, zero)
        rseq2 = [v / q2 for v in rseq]
    else:
        rseq2 = rseq

    # using rpolynomial is slightly more accurate but twice slower:
    # eq2 = rpolynomial(ctx, rseq2, params['rC'], functional=functional, size=size + 1 if q else size, scale=scale)
    eq2 = polynomial(
        ctx, rseq2, params["C"], functional=functional, size=size if q else size, scale=scale or True, base=base, dtype=dtype
    )

    if q:
        er = power(ctx, eq2, 2**q, functional=functional, size=size, scale=scale, dtype=dtype)
    else:
        er = eq2

    # split k into k2 + k3 to avoid overflow from exp2(k) when k is
    # large while abs(v) < 1:
    khalf = ctx.trunc(k / dtype(2))  # ensure that k2, k3 will be powers of
    # two, otherwise the multiplication
    # operations below will be inaccurate
    k2 = ctx.exp2(khalf)
    k3 = ctx.exp2(k - khalf)
    r = [k2 * v * k3 for v in er]
    if functional or size is not None:
        r = r[:size]

    large = seq[0] > params["log_largest"]
    small = seq[0] <= params["log_smallest"]

    f = ctx.logical_or(small, large)
    r = [ctx.select(small, zero, ctx.select(large, inf, r[0]))] + [ctx.select(f, zero, c) for c in r[1:]]
    return r


@fpa.make_api(mp_func=lambda mp_ctx, *args: mp_ctx.log(*args))
def logarithm(ctx, dtype, seq, functional=False, size=None, niter=None):
    """Return FP expansion of log(x).

    Algorithm
    ---------

    Given y_0 = [ctx.log(seq[0])], we'll use the following iterative
    scheme to find the logarithm of x:

      y_{k + 1} = y_k + multiply(seq, exponential(-y_k)) - 1
    """
    if niter is None:
        niter = {numpy.float16: 2, numpy.float32: 1, numpy.float64: 1}[dtype]
    y0 = ctx.log(seq[0])
    neg_one = ctx.constant(-1)
    y1 = [y0]
    for i in range(niter):
        e1 = exponential(ctx, negate(ctx, y1), functional=functional, size=size + 1, dtype=dtype)
        m = multiply(ctx, seq, e1, functional=functional, size=size, dtype=dtype)
        y1 = add(ctx, y1, m + [neg_one], functional=functional, size=size, dtype=dtype)
    return y1


@functools.lru_cache(typed=True)
def _hypergeometric0f1_zeros(dtype, b, start, end, size=2):
    """Return hypergeometric0f1 zeros indexed with range(start, end) as mpmath floats.

    size represents a precision that is achievable with a floating
    point expansion with the given size.
    """
    fi = numpy.finfo(dtype)
    prec = (-fi.negep) * (size + 1) + 5

    niter = {numpy.float16: 4, numpy.float32: 5, numpy.float64: 6}[dtype]
    niter += int(b)

    def iszero(value):
        if value is None:
            return False
        if value == 0:
            return True
        return fa.utils.number2float(dtype, value) == 0

    return ghf.hyp0f1_zeros(b, start=start, end=end, niter=niter, iszero=iszero, prec=prec)


@functools.lru_cache(typed=True)
def _hypergeometric0f1_taylor_parameters(dtype, b, functional):
    import functional_algorithms.generalized_hypergeometric_functions as ghf

    main_zero_index = 1

    k = {numpy.float16: 16, numpy.float32: 25 + 4, numpy.float64: 15 + 50}[dtype]

    C = ghf.pFq_taylor_coeffs([], [b], k)  # TODO: use i=range(k)

    zeros = fa.utils.number2fraction(ghf.hyp0f1_zeros(b, end=main_zero_index + 1, niter=100))
    C = fa.polynomial.taylorat(C, zeros[main_zero_index], reverse=False)[1:]

    rC = fa.polynomial.asrpolynomial(C, reverse=False)
    rC = [fa.utils.number2expansion(dtype, c_, length=2, functional=functional) for c_ in rC]

    main_zero = fa.utils.number2expansion(dtype, zeros[main_zero_index], length=2, functional=functional)

    return dict(rC=rC, main_zero=main_zero)


@fpa.make_api(mp_func=lambda mp_ctx, b, x: mp_ctx.hyper([], [b], x))
def hypergeometric0f1_taylor(ctx, dtype, b, seq, functional=False, size=None):
    """Return FP expansion of hypergeometric0f1(b, x).

    Algorithm
    ---------

    We'll use Taylor series of hypergeometic0f1 about its third zero
    that provides maximal range where the absolute error <= 1 ULPs

      -25 <= x <= 0       [float16]

    except in the surrounding of its other zeros.

    Use hypergeometric0f1_zeros_correction to extend the maximal
    accuracy to the neighboring points of hypergeometric0f1 zeros.

    For small values of x, use hypergeometric0f1_asymptotic that
    ensures high accuracy results for large abs(x).
    """

    params = _hypergeometric0f1_taylor_parameters(dtype, b, functional=functional)

    return rtaylor(ctx, seq, params["rC"], params["main_zero"], reverse=False, functional=functional, size=size, dtype=dtype)


@functools.lru_cache(typed=True)
def _hypergeometric0f1_zeros_correction_parameters(dtype, b, zero_indices=None, functional=False, size=None):
    import functional_algorithms.generalized_hypergeometric_functions as ghf

    if zero_indices is None:
        if dtype == numpy.float16:
            # max zero range end for float16 is 163
            zero_indices = [0, (2, 12)]
        else:
            # There will be dULP > 1 errors around all zeros of
            # 0f1. Users can enable the correction of zeros by
            # specifying the zeros indices around which exact results
            # are requited.
            zero_indices = []

    if size is None:
        size = 2

    fi = numpy.finfo(dtype)
    prec = (-fi.negep) * (size + 1) + 5

    niter = {numpy.float16: 4, numpy.float32: 5, numpy.float64: 6}[dtype]
    niter += int(b)

    # The estimates to k and zero_window_width are computed using the
    # script tools/hyper0f1_params.py, b == 1, max number of zeros ==
    # 50 The b correction term is found for max number of zeros == 20.

    zeros = []
    zero_window_width = []
    zeros_index = []
    for index in zero_indices:
        if isinstance(index, int):
            zero_range_start = index
            zero_range_end = index + 1
        elif isinstance(index, tuple):
            zero_range_start, zero_range_end = index
        else:
            assert 0, type(index)  # not impl
        zeros_index += list(range(zero_range_start, zero_range_end))

        zeros_ = fa.utils.number2fraction(
            ghf.hyp0f1_zeros(b, start=zero_range_start, end=zero_range_end, niter=niter, prec=prec)
        )
        zeros += zeros_

        if dtype == numpy.float16:
            zero_window_width += [dtype(0.0198 * fa.utils.number2float(dtype, abs(r)) ** 0.65) for r in zeros_]
        elif dtype == numpy.float32:
            zero_window_width += [dtype(0.0063 * fa.utils.number2float(dtype, abs(r)) ** 0.63) for r in zeros_]
        elif dtype == numpy.float64:
            zero_window_width += [dtype(9e-6 * fa.utils.number2float(dtype, abs(r)) ** 0.61) for r in zeros_]
        else:
            assert 0, dtype  # unreachable

    zero_appoximation_degree = 3

    rpolynomials_at_zeros = []
    for i, r in enumerate(zeros):
        index = zeros_index[i]
        if dtype == numpy.float16:
            k = int(zero_appoximation_degree + 3 + (index + 1) * 5.44 + 2 * b)
        elif dtype == numpy.float32:
            k = int(zero_appoximation_degree + 8 + (index + 1) * 4.28 + 2 * b)
        elif dtype == numpy.float64:
            k = int(zero_appoximation_degree + 15 + (index + 1) * 4.33 + 2 * b)
        else:
            assert 0, dtype  # unreachable

        C0 = [
            fa.polynomial.fast_polynomial(r, T)
            for T in ghf.pFq_taylor_coeffs([], [b], k, c=1, i=range(1, zero_appoximation_degree + 1))
        ]
        rC0 = fa.polynomial.asrpolynomial(C0, reverse=False)
        rC0 = [fa.utils.number2expansion(dtype, c_, length=2, functional=functional) for c_ in rC0]
        rpolynomials_at_zeros.append(rC0)

    zeros = [fa.utils.number2expansion(dtype, r, length=2, functional=functional) for r in zeros]
    return dict(zeros=zeros, rpolynomials_at_zeros=rpolynomials_at_zeros, zero_window_width=zero_window_width)


@fpa.make_api(mp_func=lambda mp_ctx, b, x, y: y)
def hypergeometric0f1_zeros_correction(ctx, dtype, b, seq, rseq, zero_indices=None, functional=False, size=None):
    """Return FP expansion of hypergeometric0f1(b, x) result that is
    accuracy corrected around its zeros.

    zero_indices specifies zeros of hypergeometric0f1(b, x) that will
    be accuracy corrected.

    Algorithm
    ---------

    Around each specified zero, we'll apply cubic Taylor series
    approximation.
    """
    if isinstance(zero_indices, list):
        # allow zero_indices to be a part of a cache key
        zero_indices = tuple(zero_indices)

    params = _hypergeometric0f1_zeros_correction_parameters(
        dtype, b, zero_indices=zero_indices, functional=functional, size=size
    )

    zeros = params["zeros"]
    rpolynomials_at_zeros = params["rpolynomials_at_zeros"]
    zero_window_width = params["zero_window_width"]

    result = rseq
    for i in range(len(zeros)):
        rC0 = rpolynomials_at_zeros[i]
        dseq0 = subtract(ctx, seq, zeros[i], functional=functional, size=size, dtype=dtype)
        result0 = rtaylor(ctx, seq, rC0, zeros[i], reverse=False, functional=functional, size=size, dtype=dtype)
        result = ctx.select(abs(dseq0[0]) < zero_window_width[i], result0, result)

    return result


@fpa.make_api(mp_func=lambda mp_ctx, b, x: mp_ctx.hyper([], [b], x))
def hypergeometric0f1_asymptotic(ctx, dtype, b, seq, enable_largest_correction=False, functional=False, size=None):
    """Generalized hypergeometic series hypergeometric([], [b], seq) for
    large negative FP expansion.

    Algorithm
    ---------

    We have

      J_a(z) = (z/2) ^ a / a! hyp0f1(1 + a, -z^2/4)

    and for z -> oo,

      J_a(z) ~ sqrt(2 / pi / z) * (cos(omega) * JC(z) - sin(omega) * JS(z))

    where

      omega = z - a * pi / 2 - pi / 4
      JC(z) = sum((-1)**k * (1/2 - a)_{2*k} (1/2 + a)_{2*k} / (-2)^{2*k} / (2*k)! / z^{2*k}, k=0..oo)
      JS(z) = -sum((-1)**k * (1/2 - a)_{2*k+1} (1/2 + a)_{2*k+1} / (-2)^{2*k+1} / (2*k+1)! / z^{2*k+1}, k=0..oo)

    Reference: https://dlmf.nist.gov/10.17

    Let's define

      b = 1 + a
      x = -z^2 / 4

    then

      hyp0f1(b, x) = (b - 1)! * (-x)^(-b/2 + 1/4) / sqrt(pi) * (cos(omega) * JC(2 * sqrt(-x)) - sin(omega) * JS(2 * sqrt(-x)))
      omega = 2 * sqrt(-x) - b * pi / 2 + pi / 4

    for x -> -oo.
    """

    prec = 110

    params = hypergeometric0f1_asymptotic_parameters(dtype, b, max_k=20, prec=prec, size=size)

    # -z
    nseq = negate(ctx, seq)

    # 1 / (-z)
    rz = reciprocal(ctx, nseq, functional=functional, size=size, dtype=dtype)

    # Increasing size_sqrt improves the accuracy of results for large
    # arguments (abs(x) > sqrt(largest)). However, the number of extra
    # operations will increase the number of operations accordingly.
    if enable_largest_correction:
        # maximal accuracy up to largest
        size_sqrt = {numpy.float16: size + 2, numpy.float32: size + 2, numpy.float64: size + 9}[dtype]
    else:
        # maximal accuracy up to sqrt(largest)
        size_sqrt = {numpy.float16: size + 1, numpy.float32: size + 1, numpy.float64: size + 4}[dtype]
    sz = sqrt(ctx, nseq, functional=functional, size=size_sqrt, dtype=dtype)  # accuracy bottleneck

    sz2 = [s_ * dtype(2) for s_ in sz]

    # sine/cosine of 2 * sqrt(-z)
    sn2z, cs2z = sine_cosine(ctx, sz2, functional=functional, size=size)  # accuracy bottleneck

    snb, csb = params["snb"], params["csb"]

    cso = add(
        ctx,
        multiply(ctx, cs2z, csb, functional=functional, size=size, dtype=dtype),
        multiply(ctx, sn2z, snb, functional=functional, size=size, dtype=dtype),
        functional=functional,
        size=size + 1,
        dtype=dtype,
    )  # accuracy bottleneck, hence using size+1
    sno = subtract(
        ctx,
        multiply(ctx, sn2z, csb, functional=functional, size=size, dtype=dtype),
        multiply(ctx, cs2z, snb, functional=functional, size=size, dtype=dtype),
        functional=functional,
        size=size,
        dtype=dtype,
    )

    # (-x) ** (-(2 * b + 1) / 4), (-x) ** (-(2 * b - 1) / 4)
    ssz = power(ctx, nseq, params["nbm4"], functional=functional, size=size, dtype=dtype)  # accuracy bottleneck
    csz = power(ctx, nseq, params["nbp4"], functional=functional, size=size, dtype=dtype)  # accuracy bottleneck

    js = rpolynomial(ctx, rz, params["rS"], functional=functional, size=size, dtype=dtype)
    jc = rpolynomial(ctx, rz, params["rC"], functional=functional, size=size, dtype=dtype)

    js = multiply(ctx, js, ssz, functional=functional, size=size, dtype=dtype)
    jc = multiply(ctx, jc, csz, functional=functional, size=size, dtype=dtype)

    r = subtract(
        ctx,
        multiply(ctx, cso, jc, functional=functional, size=size, dtype=dtype),
        multiply(ctx, sno, js, functional=functional, size=size, dtype=dtype),
        functional=functional,
        size=size,
        dtype=dtype,
    )

    return r


@fpa.make_api(mp_func=lambda mp_ctx, b, x: mp_ctx.hyper([], [b], x))
def hypergeometric0f1(ctx, dtype, b, seq, zero_indices=None, enable_largest_correction=False, functional=False, size=None):
    """Return FP expansion of hypergeometric0f1(b, x) result.

    Algorithm
    ---------

    For small abs(x) values (the limit depends on dtype), we'll use
    Taylor series around the third zero of hypergeometric0f1(b,
    x). The order of series depends on the dtype.

    For large abs(x) values, we'll use the asymptotic series of
    hypergeometric0f1.

    For x values close to hypergeometric0f1(b, x) zeros, one can
    specify zero_indices that enables cubic Taylor series correction
    around the specified zeros zeros.
    """
    r_a = hypergeometric0f1_asymptotic(
        ctx, b, seq, enable_largest_correction=enable_largest_correction, functional=functional, size=size, dtype=dtype
    )
    r_t = hypergeometric0f1_taylor(ctx, b, seq, functional=functional, size=size, dtype=dtype)
    # TODO: a2t_z likely depends on the value of b
    # float16: a2t_z is between the 2-nd and 3-rd zeros of 0f1
    # float32: a2t_z is between the 3-rd and 4-th zeros of 0f1
    # float64: a2t_z is between the 10-th and 11-th zeros of 0f1
    a2t_z = {numpy.float64: 320 - 50, numpy.float32: 44, numpy.float16: 21}[dtype]
    r = ctx.select(seq[0] < -a2t_z, r_a, r_t)
    r = hypergeometric0f1_zeros_correction(
        ctx, b, seq, r, zero_indices=zero_indices, functional=functional, size=size, dtype=dtype
    )
    return r


@fpa.make_api(mp_func=None)
def polynomial(ctx, dtype, seq, coeffs, reverse=False, functional=False, fast=False, size=None, base=None, scale=True):
    """Evaluate a polynomial

      P(x) = coeffs[0] + coeffs[1] * x + ... + coeffs[N] * x ** N
           = coeffs[0] + x * (coeffs[1] + ... + coeffs[N] * x ** (N - 1))
           = coeffs[0] + x * (coeffs[1] + ... + x * (coeffs[N - 1] + x * coeffs[N]))

    using Horner' scheme.
    """
    if reverse:
        return polynomial(
            ctx,
            seq,
            coeffs[::-1],
            reverse=False,
            functional=functional,
            fast=fast,
            size=size,
            base=base,
            scale=scale,
            dtype=dtype,
        )

    c = coeffs[-1]
    if not isinstance(c, list):
        c = [c]
    r = c
    for c in reversed(coeffs[:-1]):
        if not isinstance(c, list):
            c = [c]
        r = multiply(ctx, r, seq, functional=functional, fast=fast, size=size, base=base, scale=scale, dtype=dtype)
        r = add(ctx, c, r, functional=functional, fast=fast, size=size, dtype=dtype)
    return r


@fpa.make_api(mp_func=None)
def rpolynomial(ctx, dtype, seq, rcoeffs, reverse=False, functional=False, fast=False, size=None, base=None, scale=True):
    """Evaluate a polynomial

      P(x) = coeffs[0] + coeffs[1] * x + ... + coeffs[N] * x ** N

    where coefficients are given as ratios of coefficients and their
    following counterparts:

      rcoeffs[i] = coeffs[i] / coeffs[i - 1]

    where we define coeffs[-1] == 1 and seq is expansion of x.
    """
    if reverse:
        return rpolynomial(
            ctx, seq, rcoeffs[::-1], reverse=False, functional=functional, fast=fast, size=size, scale=scale, dtype=dtype
        )

    one = ctx.constant(1)
    r = None
    for i, rc in enumerate(reversed(rcoeffs[1:])):
        if not isinstance(rc, list):
            rc = [rc]
        rcx = multiply(ctx, seq, rc, functional=functional, fast=fast, size=size, base=base, scale=scale, dtype=dtype)
        if r is not None:
            rcx = multiply(ctx, rcx, r, functional=functional, fast=fast, size=size, scale=scale, dtype=dtype)
        r = add(ctx, [one], rcx, functional=functional, fast=fast, size=size, dtype=dtype)
    rc = rcoeffs[0]
    if not isinstance(rc, list):
        rc = [rc]
    if r is None:
        return rc
    return multiply(ctx, r, rc, functional=functional, fast=fast, size=size, base=base, scale=scale, dtype=dtype)


@fpa.make_api(mp_func=None)
def rtaylor(ctx, dtype, seq, rcoeffs, seq0, reverse=False, functional=False, fast=False, size=None):
    dseq = subtract(ctx, seq, seq0, functional=functional, fast=fast, size=size, dtype=dtype)
    p = rpolynomial(ctx, dseq, rcoeffs, reverse=reverse, functional=functional, fast=fast, size=size, dtype=dtype)
    return multiply(ctx, p, dseq, functional=functional, fast=fast, size=size, dtype=dtype)


@functools.lru_cache(typed=True)
def two_over_pi(ctx, dtype, size=None, base=None):
    """Return FP expansion of 2 / pi."""
    import mpmath

    if size is None:
        size = 2

    fi = numpy.finfo(dtype)
    # max_prec = {numpy.float16: 24, numpy.float32: 149, numpy.float64: 1074}[dtype]
    max_prec = -fi.negep * size + 16
    mp_ctx = mpmath.mp
    with mp_ctx.workprec(max_prec):
        return fa.utils.mpf2expansion(dtype, 2 / mp_ctx.pi, length=size, functional=True, base=base)


@functools.lru_cache(typed=True)
def pi_over_two(ctx, dtype, size=None, base=None):
    """Return FP expansion of 2 / pi."""
    import mpmath

    if size is None:
        size = 2

    fi = numpy.finfo(dtype)
    # max_prec = {numpy.float16: 24, numpy.float32: 149, numpy.float64: 1074}[dtype]
    max_prec = -fi.negep * size + 16
    mp_ctx = mpmath.mp
    with mp_ctx.workprec(max_prec):
        result = fa.utils.mpf2expansion(dtype, mp_ctx.pi / 2, length=size, functional=True, base=base)

    return result


@fpa.make_api(mp_func=None)
def argument_reduction(ctx, dtype, seq, size=None, functional=False, base=None, scale=True):
    """Return k, rseq such that

    x = 2 * pi * N + k * pi / 2 + sum(rseq)

    where seq is FP expansion of x, k in [-1, 0, 1, 2], sum(rseq) <=
    0.5005, and N is integral that is not computed.
    """
    if size is None:
        size = 2

    def two_over_pi_func(dtype):
        two_over_pi_size = {numpy.float16: 4, numpy.float32: 7, numpy.float64: 21}[dtype]
        return two_over_pi(ctx, dtype, size=two_over_pi_size, base=base)

    pi_over_two_base = base

    def pi_over_two_func(dtype):
        # Using same size as input seq as we don't need more accuracy
        # than that.
        return pi_over_two(ctx, dtype, size=size, base=pi_over_two_base)

    largest = fpa.get_largest(ctx)
    two_over_pi_seq = fpa.switch_largest(ctx, largest, two_over_pi_func)
    pi_over_two_seq = fpa.switch_largest(ctx, largest, pi_over_two_func)

    x_2opi = multiply_mod4(ctx, seq, two_over_pi_seq, size=size, functional=functional, base=base, scale=scale, dtype=dtype)

    k = ctx.round(x_2opi[0])  # -2, -1, 0, 1, 2

    rseq = multiply(
        ctx,
        add(ctx, x_2opi, [-k], size=size, functional=functional),
        pi_over_two_seq,
        base=pi_over_two_base,
        functional=functional,
        size=size,
        scale=False,
        dtype=dtype,
    )
    k = ctx.select(k == -2, -k, k)  # -1, 0, 1, 2

    qpi = ctx.constant(numpy.pi / 4)
    rseq = where(ctx, abs(seq[0]) < qpi, seq, rseq)

    return k, rseq


def where(ctx, mask, iftrue, iffalse):
    """Copy from iftrue or iffalse depending on the mask truth value."""
    zero = ctx.constant(0)
    iftrue_extra = [zero] * max(0, len(iffalse) - len(iftrue))
    iffalse_extra = [zero] * max(0, len(iftrue) - len(iffalse))
    return [ctx.select(mask, x, y) for x, y in zip(iftrue + iftrue_extra, iffalse + iffalse_extra)]


def sine_coeffs(n):
    """Return Sn such that

    sine(x) ~ x * polynomial(x * x, Sn)
    """
    assert isinstance(n, int)
    lst = [fractions.Fraction(1)]
    for i in range(1, n):
        lst.append(-lst[-1] * fractions.Fraction(1, 2 * i * (2 * i + 1)))
    return lst


def cosine_minus_one_coeffs(n):
    """Return Cs such that

    cosine(x) ~ 1 + x * x * polynomial(x * x, Cs)
    """
    assert isinstance(n, int)
    lst = [fractions.Fraction(-1, 2)]
    for i in range(2, n):
        lst.append(-lst[-1] * fractions.Fraction(1, 2 * i * (2 * i - 1)))
    return lst


@functools.lru_cache(typed=True)
def sine_expansion_coeffs(dtype, n, size=2):
    coeffs = sine_coeffs(n)
    return fa.utils.number2expansion(dtype, coeffs, length=size, functional=False)


@functools.lru_cache(typed=True)
def cosine_minus_one_expansion_coeffs(dtype, n, size=2):
    coeffs = cosine_minus_one_coeffs(n)
    return fa.utils.number2expansion(dtype, coeffs, length=size, functional=False)


@functools.lru_cache(typed=True)
def sine_expansion_rcoeffs(dtype, n, size=2, base=None):
    coeffs = sine_coeffs(n)
    coeffs = fa.polynomial.asrpolynomial(coeffs, reverse=False)
    return fa.utils.number2expansion(dtype, coeffs, length=size, functional=False, base=base)


@functools.lru_cache(typed=True)
def cosine_minus_one_expansion_rcoeffs(dtype, n, size=2, base=None):
    coeffs = cosine_minus_one_coeffs(n)
    coeffs = fa.polynomial.asrpolynomial(coeffs, reverse=False)
    return fa.utils.number2expansion(dtype, coeffs, length=size, functional=False, base=base)


@fpa.make_api(mp_func=lambda mp_ctx, x: mp_ctx.sin(x))
def sine(ctx, dtype, seq, n=None, functional=False, fast=False, size=None, base=None, scale=True):
    return sine_impl(ctx, dtype, "sin", seq, n=n, functional=functional, fast=fast, size=size, base=base, scale=scale)


@fpa.make_api(mp_func=lambda mp_ctx, x: mp_ctx.cos(x))
def cosine(ctx, dtype, seq, n=None, functional=False, fast=False, size=None, base=None, scale=True):
    return sine_impl(ctx, dtype, "cos", seq, n=n, functional=functional, fast=fast, size=size, base=base, scale=scale)


@fpa.make_api(mp_func=lambda mp_ctx, x: mp_ctx.cos(x) - 1)
def cosine_minus_one(ctx, dtype, seq, n=None, functional=False, fast=False, size=None, base=None, scale=True):
    return sine_impl(ctx, dtype, "cosm1", seq, n=n, functional=functional, fast=fast, size=size, base=base, scale=scale)


@fpa.make_api(mp_func=lambda mp_ctx, x: (mp_ctx.sin(x), mp_ctx.cos(x)))
def sine_cosine(ctx, dtype, seq, n=None, functional=False, fast=False, size=None, base=None, scale=True):
    return sine_impl(ctx, dtype, "sincos", seq, n=n, functional=functional, fast=fast, size=size, base=base, scale=scale)


def sine_impl(ctx, dtype, target, seq, n=None, functional=False, fast=False, size=None, base=None, scale=None):
    """Sine of FP expansion."""

    if size is None:
        size = 2
    if n is None:
        """
        number of samples = 61441
        float coefficients Sn, Cs

        n == 3, float16
        ---------------
        apmath.sine:
          ULP difference 0: 59249
          ULP difference 1: 2192
        apmath.sine:
          precision 50: 1
          precision 43.9: 2
          precision 43.5: 2
          precision 43.1: 2
          precision in [11.7..39.7]: 60716
          precision 11.5: 100
          precision 11.2: 204
          precision 11.0: 412
          precision 9.4: 2

        n == 4, float16
        ---------------
        apmath.sine:
          ULP difference 0: 61127
          ULP difference 1: 314
        apmath.sine:
          precision 50: 1
          precision 43.9: 2
          precision 43.5: 2
          precision 43.1: 2
          precision in [12.0..39.7]: 61424
          precision 11.5: 2
          precision 11.2: 2
          precision 11.0: 4
          precision 9.4: 2

        n == 5, float16
        ---------------
        apmath.sine:
          ULP difference 0: 61145
          ULP difference 1: 296
        apmath.sine:
          <same as n==4, float16>
        apmath.cosine:
          ULP difference 0: 61151
          ULP difference 1: 288
          ULP difference 2: 2

        n == 5, float16, with Sn, Cs expansion rcoefficients
        ---------------------------------------------------
        apmath.sine:
          ULP difference 0: 61409
          ULP difference 1: 32
        apmath.sine:
          precision 50: 1
          precision 43.9: 2
          precision 43.5: 2
          precision 43.1: 2
          precision in [12.0..40.5]: 61424
          precision 11.5: 2
          precision 11.2: 2
          precision 11.0: 4
          precision 9.4: 2
        apmath.cosine:
          ULP difference 0: 61419
          ULP difference 1: 20
          ULP difference 2: 2
        apmath.cosine:
          precision 50: 1
          precision 48.3: 2
          precision 48.1: 2
          precision 48.0: 4
          precision in [13.0..44.8]: 61424
          precision 12.0: 2
          precision 11.0: 2
          precision 10.2: 2
          precision 9.0: 2

        n == 5, float16, with Sn, Cs expansion coefficients
        ---------------------------------------------------
        apmath.sine:
          ULP difference 0: 61405
          ULP difference 1: 36
        apmath.sine:
          precision 50: 1
          precision 43.9: 2
          precision 43.5: 2
          precision 43.1: 2
          precision in [12.0..40.5]: 61424
          precision 11.5: 2
          precision 11.2: 2
          precision 11.0: 4
          precision 9.4: 2
        apmath.cosine:
          ULP difference 0: 61417
          ULP difference 1: 22
          ULP difference 2: 2
        apmath.cosine:
          precision 50: 1
          precision 48.3: 2
          precision 48.1: 2
          precision 48.0: 4
          precision in [13.0..44.8]: 61424
          precision 12.0: 2
          precision 11.0: 2
          precision 10.2: 2
          precision 9.0: 2

        n == 5, float16, base == 128 [no scale]
        ----------------------------
        apmath.sine:
          ULP difference 0: 61143
          ULP difference 1: 298
        apmath.sine:
          precision 50: 1
          precision 43.9: 2
          precision 43.5: 2
          precision 43.1: 2
          precision in [13.0..39.7]: 61420
          precision 12.0: 6
          precision 11.2: 2
          precision 11.0: 4
          precision 9.4: 2


        n == 6, float16
        ---------------
        <same as n==5, float16>

        n == 4, float32
        ---------------
        failure due to ulp <= 1 assert

        n == 5, float32
        ---------------
        apmath.sine:
          ULP difference 0: 60653
          ULP difference 1: 788
        apmath.sine:
          precision 250: 477
          precision 249.0: 378
          precision 248.0: 98
          precision 247.0: 122
          precision in [25.4..246.9]: 59646
          precision 25.3: 48
          precision 25.2: 100
          precision 25.1: 202
          precision 25.0: 370

        n == 6, float32
        ---------------
        apmath.sine:
          ULP difference 0: 61201
          ULP difference 1: 240
        apmath.sine:
          precision 250: 477
          precision 249.0: 378
          precision 248.0: 98
          precision 247.0: 122
          precision in [28.5..246.9]: 59334
          precision 28.3: 76
          precision 28.2: 132
          precision 28.1: 290
          precision 28.0: 534

        n == 6, float32, with Sn, Cs expansion coefficients
        ---------------------------------------------------
        apmath.sine:
          ULP difference 0: 61437
          ULP difference 1: 4
        apmath.sine:
          precision 250: 477
          precision 249.0: 378
          precision 248.0: 98
          precision 247.0: 122
          precision in [33.5..246.9]: 59564
          precision 33.4: 48
          precision 33.3: 114
          precision 33.1: 224
          precision 33.0: 416
        apmath.cosine:
          ULP difference 0: 61437
          ULP difference 1: 4
        apmath.cosine:
          precision 250: 243
          precision 249.0: 356
          precision 248.0: 110
          precision 247.0: 124
          precision in [33.5..246.9]: 59796
          precision 33.4: 58
          precision 33.3: 120
          precision 33.1: 218
          precision 33.0: 416


        n == 7, float32
        ---------------
        <same as n == 6, float32>

        n == 8, float64
        ---------------
        failure due to ulp <= 1 assert

        n == 9, float64
        ---------------
        apmath.sine:
          ULP difference 0: 61247
          ULP difference 1: 194
        apmath.sine:
          precision 1350: 10453
          precision 1349.0: 44
          precision 1348.0: 6
          precision 1347.0: 14
          precision in [57.4..1346.0]: 49692
          precision 57.3: 14
          precision 57.2: 54
          precision 57.1: 244
          precision 57.0: 920

        n == 9, float64, with Sn, Cs expansion coefficients
        ----------------------------------------------------
        apmath.sine:
          ULP difference 0: 61423
          ULP difference 1: 18
        apmath.sine:
          precision 1350: 10453
          precision 1349.0: 44
          precision 1348.0: 6
          precision 1347.0: 14
          precision in [59.0..1346.0]: 50814
          precision 58.3: 4
          precision 58.2: 6
          precision 58.1: 16
          precision 58.0: 84
        apmath.cosine:
          ULP difference 0: 61431
          ULP difference 1: 10
        apmath.cosine:
          precision 1350: 10421
          precision 1349.0: 52
          precision 1348.0: 16
          precision 1347.0: 16
          precision in [59.1..1346.0]: 50428
          precision 59.0: 402
          precision 58.2: 4
          precision 58.1: 14
          precision 58.0: 88

        n == 10, float64
        ----------------
        apmath.sine:
          ULP difference 0: 61241
          ULP difference 1: 200
        apmath.sine:
          precision 1350: 10453
          precision 1349.0: 44
          precision 1348.0: 6
          precision 1347.0: 14
          precision in [57.4..1346.0]: 49718
          precision 57.3: 14
          precision 57.2: 54
          precision 57.1: 244
          precision 57.0: 894

        n == 10, float64, with Sn, Cs expansion coefficients
        ----------------------------------------------------
        apmath.sine:
          ULP difference 0: 61441
        apmath.sine:
          precision 1350: 10453
          precision 1349.0: 44
          precision 1348.0: 6
          precision 1347.0: 14
          precision in [68.5..1346.0]: 50386
          precision 68.3: 10
          precision 68.2: 68
          precision 68.1: 222
          precision 68.0: 238
        apmath.cosine:
          ULP difference 0: 61441
        apmath.cosine:
          precision 1350: 10421
          precision 1349.0: 52
          precision 1348.0: 16
          precision 1347.0: 16
          precision in [68.4..1346.0]: 50482
          precision 68.3: 2
          precision 68.2: 38
          precision 68.1: 190
          precision 68.0: 224
        """
        n = {numpy.float16: 5, numpy.float32: 7, numpy.float64: 10}[dtype]

    if base is None:
        base = {numpy.float16: 32, numpy.float32: 32, numpy.float64: 32}[dtype]
    if scale is None:
        scale = {numpy.float16: base < 128, numpy.float32: base < 8192, numpy.float64: base < 268435456}[dtype]

    coeffs_size = 2
    rcoeffs_base = None

    k, rseq = argument_reduction(ctx, seq, size=size, functional=functional, base=base, scale=scale, dtype=dtype)
    sq_rseq = square(ctx, rseq, functional=functional, fast=False, size=size, dtype=dtype)

    if 1:
        rSn = sine_expansion_rcoeffs(dtype, n, size=coeffs_size, base=rcoeffs_base)
        sn = rpolynomial(
            ctx, sq_rseq, rSn, reverse=False, functional=functional, fast=False, size=size, base=rcoeffs_base, scale=False
        )
    else:
        Sn = sine_expansion_coeffs(dtype, n, size=coeffs_size)
        sn = polynomial(ctx, sq_rseq, Sn, reverse=False, functional=functional, fast=False, size=size, dtype=dtype)
    sn = multiply(ctx, rseq, sn, functional=functional, fast=False, size=size, base=None, scale=False, dtype=dtype)

    if 1:
        rCs = cosine_minus_one_expansion_rcoeffs(dtype, n, size=coeffs_size, base=rcoeffs_base)
        cs = rpolynomial(
            ctx,
            sq_rseq,
            rCs,
            reverse=False,
            functional=functional,
            fast=False,
            size=size,
            base=rcoeffs_base,
            scale=False,
            dtype=dtype,
        )
    else:
        Cs = cosine_minus_one_expansion_coeffs(dtype, n, size=coeffs_size)
        cs = polynomial(ctx, sq_rseq, Cs, reverse=False, functional=functional, fast=False, size=size, dtype=dtype)
    cs = multiply(ctx, sq_rseq, cs, functional=functional, fast=False, size=size, base=None, scale=False, dtype=dtype)
    if target != "cosm1":
        cs = add(ctx, cs, [ctx.constant(1)], functional=functional, size=size, dtype=dtype)

    if target == "sin":
        return ctx.select(k == 0, sn, ctx.select(k == 1, cs, ctx.select(k == 2, negate(ctx, sn), negate(ctx, cs))))
    elif target in {"cos", "cosm1"}:
        return ctx.select(k == -1, sn, ctx.select(k == 0, cs, ctx.select(k == 1, negate(ctx, sn), negate(ctx, cs))))
    elif target == "sincos":
        return (
            ctx.select(k == 0, sn, ctx.select(k == 1, cs, ctx.select(k == 2, negate(ctx, sn), negate(ctx, cs)))),
            ctx.select(k == -1, sn, ctx.select(k == 0, cs, ctx.select(k == 1, negate(ctx, sn), negate(ctx, cs)))),
        )
    else:
        assert 0, target  # unreachable
