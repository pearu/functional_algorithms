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
"""

from . import floating_point_algorithms as fpa


def _merge(ctx, x, y):
    if not x:
        return y
    if not y:
        return x
    f = ctx.lt(x[0], y[0])
    r1 = _merge(ctx, x[1:], y)
    r2 = _merge(ctx, x, y[1:])
    assert len(r1) == len(r2)
    lst = [ctx.select(f, x[0], y[0])]
    for a, b in zip(r1, r2):
        lst.append(ctx.select(f, a, b))
    return lst


def mergesort(ctx, lst):
    """Return a sorted list of expressions using mergesort method."""
    if len(lst) < 2:
        return lst
    n = len(lst) // 2
    return _merge(ctx, mergesort(ctx, lst[:n]), mergesort(ctx, lst[n:]))


def two_sum(ctx, a, b):
    return fpa.add_2sum(ctx, a, b, fast=False)


def quick_two_sum(ctx, a, b):
    return fpa.add_2sum(ctx, a, b, fast=True)


def split(ctx, a):
    return fpa.split_veltkamp(ctx, a)


def two_prod(ctx, a, b):
    return fpa.mul_dekker(ctx, a, b)


def renormalize(ctx, seq, functional=False, fast=False):
    """Convert a list of possibly overlapping items to a list of
    non-overlapping items.

    The renormalization algorithm originates from Priest's PhD thesis
    (1992) and there exists different modifications of it. Here, we'll
    use the renormalization algorithm from
    https://hal.science/hal-01512417 that is provided with a formal
    proof and is slightly simpler from other modifications.

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
    """
    # VecSum:
    s = seq[-1]
    zero = ctx.constant(0, s)
    e_lst = []
    for i in reversed(range(len(seq) - 1)):
        if fast:
            s, e = quick_two_sum(ctx, seq[i], s)
        else:
            s, e = two_sum(ctx, seq[i], s)
        e_lst.insert(0, e)
    e_lst.insert(0, s)

    # VecSumErrBranch:
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
    elif ctx._is_nonzero(eps_i):
        f_lst.append(eps_i)
    return f_lst
