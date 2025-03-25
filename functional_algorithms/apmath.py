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

from . import floating_point_algorithms as fpa


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
        cmp = lambda ctx, x, y: ctx.lt(x, y)
    elif mth == "a<a":
        cmp = lambda ctx, x, y: ctx.lt(abs(x), abs(y))
    if mth == ">":
        cmp = lambda ctx, x, y: ctx.lt(y, x)
    elif mth == "a>a":
        cmp = lambda ctx, x, y: ctx.lt(abs(y), abs(x))
    else:
        raise NotImplementedError(f"comparison method '{mth}'")
    return _merge(ctx, mergesort(ctx, lst[:n], mth=mth), mergesort(ctx, lst[n:], mth), cmp=cmp)


def two_sum(ctx, a, b):
    return fpa.add_2sum(ctx, a, b, fast=False)


def quick_two_sum(ctx, a, b):
    return fpa.add_2sum(ctx, a, b, fast=True)


def split(ctx, a):
    return fpa.split_veltkamp(ctx, a)


def two_prod(ctx, a, b):
    return fpa.mul_dekker(ctx, a, b)


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


def renormalize(ctx, seq, functional=False, fast=False, size=None):
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
    elif ctx._is_nonzero(eps_i):
        f_lst.append(eps_i)
    if size is not None:
        return f_lst[:size]
    return f_lst


def negate(ctx, seq):
    """Negate an FP expansion."""
    return [-x for x in seq]


def add(ctx, seq1, seq2, functional=False, fast=False, size=None):
    """Add two FP expansions."""
    return renormalize(ctx, seq1 + seq2, functional=functional, fast=fast, size=size)


def subtract(ctx, seq1, seq2, functional=False, fast=False, size=None):
    """Subtract two FP expansions."""
    return renormalize(ctx, seq1 + negate(ctx, seq2), functional=functional, fast=fast, size=size)


def multiply(ctx, seq1, seq2, functional=False, fast=False, size=None):
    """Multiply two FP expansions.

    Warning: when fast=True, the result will be very likely inaccurate.
    """
    r_0, e_0 = two_prod(ctx, seq1[0], seq2[0])
    r_lst = [r_0]
    e_lst = [e_0]
    for n in range(1, len(seq1) + len(seq2)):
        p_lst = []
        ne_lst = []
        for i1 in range(len(seq1)):
            for i2 in range(len(seq2)):
                if i1 + i2 == n:
                    p_i, e_i = two_prod(ctx, seq1[i1], seq2[i2])
                    p_lst.append(p_i)
                    ne_lst.append(e_i)
        lst = vecsum(ctx, p_lst + e_lst, fast=fast)
        r_lst.append(lst[0])
        e_lst = lst[1:] + ne_lst
    lst = r_lst + e_lst
    return renormalize(ctx, lst, functional=functional, fast=fast, size=size)


def square(ctx, seq, functional=False, fast=False, size=None):
    """Square of an FP expansion.

    (x0 + x1 + x2) ** 2
      = x0 ** 2 + 2 * x0 * x1 + (x1 ** 2 + 2 * x0 * x2) + 2 * x1 * x2 + x2 ** 2
    """
    r_0, e_0 = two_prod(ctx, seq[0], seq[0])
    r_lst = [r_0]
    e_lst = [e_0]
    for n in range(1, len(seq) * 2):
        p_lst = []
        ne_lst = []
        for i1 in range(len(seq)):
            i2 = n - i1
            if i1 > i2:
                continue
            elif i1 < i2:
                p_i, e_i = two_prod(ctx, seq[i1], seq[i2])
                p_i += p_i
                e_i += e_i
            else:
                p_i, e_i = two_prod(ctx, seq[i1], seq[i1])
            p_lst.append(p_i)
            ne_lst.append(e_i)
        lst = vecsum(ctx, p_lst + e_lst, fast=fast)  # should fast be always False?
        r_lst.append(lst[0])
        e_lst = lst[1:] + ne_lst
    return renormalize(ctx, r_lst + e_lst, functional=functional, fast=fast, size=size)


def reciprocal(ctx, seq, functional=False, size=None):
    """Reciprocal of an FP expansion.

    The size of the output FP expansion must be a power of two.
    """
    if size is None:
        size = len(seq)

    q = size.bit_length() - 1
    assert 2**q == size

    two = ctx.constant(2, seq[0])
    x_lst = [ctx.reciprocal(seq[0])]
    for i in range(0, q):
        v_lst = multiply(ctx, x_lst, seq[: 2 ** (i + 1)], functional=functional, size=2 ** (i + 1))
        w_lst = subtract(ctx, [two], v_lst, functional=functional, size=2 ** (i + 1))
        x_lst = multiply(ctx, x_lst, w_lst, functional=functional, size=2 ** (i + 1))

    assert len(x_lst) == size
    return x_lst


def sqrt(ctx, seq, functional=False, size=None):
    """Square root of an FP expansion.

    The size of the output FP expansion must be a power of two.
    """
    if size is None:
        size = len(seq)

    q = size.bit_length() - 1
    assert 2**q == size

    three = ctx.constant(3, seq[0])
    half = ctx.constant(0.5, seq[0])
    x_lst = [ctx.sqrt(seq[0])]
    for i in range(0, q):
        v_lst = multiply(ctx, x_lst, seq[: 2 ** (i + 1)], functional=functional, size=2 ** (i + 1))
        w_lst = multiply(ctx, x_lst, v_lst, functional=functional, size=2 ** (i + 1))
        y_lst = subtract(ctx, [three], v_lst, functional=functional, size=2 ** (i + 1))
        z_lst = multiply(ctx, x_lst, y_lst, functional=functional, size=2 ** (i + 1))
        x_lst = [z * half for z in z_lst]

    assert len(x_lst) == size
    return x_lst
