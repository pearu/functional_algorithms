"""References:

- Emulation of 3Sum, 4Sum, the FMA and the FD2 instructions in
  rounded-to-nearest floating-point arithmetic. Stef Graillat,
  Jean-Michel Muller. https://hal.science/hal-04624238/
"""


def get_largest(ctx, x: float):
    largest = ctx.constant("largest", x)
    if hasattr(largest, "reference"):
        largest = largest.reference("largest")
    return largest


def get_largest_log(ctx, x: float):
    """Get largest value such that exp(largest) is finite."""
    import numpy

    def get_value(dtype):
        return float(numpy.nextafter(numpy.log(numpy.finfo(dtype).max), dtype(0)))

    largest = ctx.constant("largest", x)
    fp64 = ctx.constant(get_value(numpy.float64), largest)
    fp32 = ctx.constant(get_value(numpy.float32), largest)
    fp16 = ctx.constant(get_value(numpy.float16), largest)
    r = ctx(ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16)))
    if hasattr(r, "reference"):
        r = r.reference("largest_log")
    return r


def get_smallest_log(ctx, x: float):
    """Get smallest value such that exp(smallest) is non-zero."""
    import numpy

    def get_value(dtype):
        return float(numpy.nextafter(numpy.log(numpy.finfo(dtype).smallest_normal), dtype(0)))

    largest = ctx.constant("largest", x)
    fp64 = ctx.constant(get_value(numpy.float64), largest)
    fp32 = ctx.constant(get_value(numpy.float32), largest)
    fp16 = ctx.constant(get_value(numpy.float16), largest)
    r = ctx(ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16)))
    if hasattr(r, "reference"):
        r = r.reference("smallest_log")
    return r


def get_veltkamp_splitter_constant(ctx, largest: float):  # deprecate
    """Return 2 ** s + 1 where s = ceil(p / 2) and s is the precision of
    the floating point number system.

    Using `largest` to detect the floating point type: float16,
    float32, or float64.
    """
    return get_veltkamp_splitter_constants(ctx, largest)[0]


def get_veltkamp_splitter_constants(ctx, largest: float):
    """Return Veltkamp splitter constants:

      V = 2 ** s + 1
      N = 2 ** s
      invN = 2 ** -s

    where s = ceil(p / 2) and s is the precision of the floating point
    number system.

    Using `largest` to detect the floating point type: float16,
    float32, or float64.
    """

    fp64 = ctx.constant(2 ** (54 // 2) + 1, largest)
    fp32 = ctx.constant(2 ** (24 // 2) + 1, largest)
    fp16 = ctx.constant(2 ** (12 // 2) + 1, largest)
    C = ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16))
    fp64 = ctx.constant(2 ** (54 // 2), largest)
    fp32 = ctx.constant(2 ** (24 // 2), largest)
    fp16 = ctx.constant(2 ** (12 // 2), largest)
    N = ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16))
    fp64 = ctx.constant(0.5 ** (54 // 2), largest)
    fp32 = ctx.constant(0.5 ** (24 // 2), largest)
    fp16 = ctx.constant(0.5 ** (12 // 2), largest)
    invN = ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16))
    if hasattr(C, "reference"):
        C = C.reference("veltkamp_C", force=True)
        N = N.reference("veltkamp_N", force=True)
        invN = invN.reference("veltkamp_invN", force=True)
    return C, N, invN


def get_tripleword_splitter_constants(ctx, largest: float):
    """Return Veltkamp splitter constants C1 and C2 for splitting a
    floating point number at p-th and (2 * p)-th bit such that
    2 * p is less that the precision of given floating point type p_dtype:

    dtype   | p_dtype | p     | p_dtype and p relation
    --------+---------+-------+------------------------
    float64 | 53      | 24    | 53 = 24 + 24 + 5
    float32 | 24      | 11    | 24 = 10 + 10 + 4
    float16 | 11      |  4    | 11 =  4 +  4 + 3

    The argument `largest` is used to detect the floating point type:
    float16, float32, or float64.
    """
    fp64 = ctx.constant(2**24 + 1, largest)
    fp32 = ctx.constant(2**10 + 1, largest)
    fp16 = ctx.constant(2**4 + 1, largest)
    C1 = ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16))
    # using 47 to maximize the accuracy in
    # test_argument_reduction_trigonometric[float64]
    fp64 = ctx.constant(2**48 + 1, largest)
    fp32 = ctx.constant(2**20 + 1, largest)
    fp16 = ctx.constant(2**8 + 1, largest)
    C2 = ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16))
    if hasattr(C1, "reference"):
        C1 = C1.reference("tripleword_splitter_C1", force=True)
        C2 = C2.reference("tripleword_splitter_C2", force=True)
    return C1, C2


def get_is_power_of_two_constants(ctx, largest: float):
    """Return Q, P constants for is_power_of_two.

    Using `largest` to detect the floating point type: float16,
    float32, or float64.
    """
    fp64 = ctx.constant(1 << (53 - 1), largest)
    fp32 = ctx.constant(1 << (24 - 1), largest)
    fp16 = ctx.constant(1 << (11 - 1), largest)
    Q = ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16)).reference("Qispowof2", force=True)

    fp64 = ctx.constant(1 << (53 - 1) + 1, largest)
    fp32 = ctx.constant(1 << (24 - 1) + 1, largest)
    fp16 = ctx.constant(1 << (11 - 1) + 1, largest)
    P = ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16)).reference("Pispowof2", force=True)
    return Q, P


def next(ctx, x: float, up=True):
    """Return

      nextafter(x, (1 if up else -1) * inf)

    using floating-point operations.

    Assumes:
    - x is normal, finite, and positive.
    - division/multiplication rounds to nearest.
    """
    largest = ctx.constant("largest", x)
    if hasattr(largest, "reference"):
        largest = largest.reference("largest")

    fp64 = ctx.constant(1 - 1 / (1 << 53), largest)
    fp32 = ctx.constant(1 - 1 / (1 << 24), largest)
    fp16 = ctx.constant(1 - 1 / (1 << 11), largest)
    c = ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16))
    if hasattr(c, "reference"):
        c = c.reference("Cnextup", force=True)

    result = ctx.select(x > 0, x / c, x * c) if up else ctx.select(x < 0, x / c, x * c)
    return result


def nextup(ctx, x: float):
    return next(ctx, x, up=True)


def nextdown(ctx, x: float):
    return next(ctx, x, up=False)


def split_veltkamp(ctx, x, C=None, scale=False):
    """Veltkamp splitter:

      x = xh + xl

    where the coefficient C defines the bit-location of the floating
    point number splitting. For instance, with

      C = 1 + 2 ** ceil(p / 2)

    where p is the precision of the floating point system, xh and xl
    significant parts fit into p / 2 bits.

    It is assumed that the aritmetical operations use rounding to
    nearest and C * x does not overflow. If scale is True, large
    abs(x) values are normalized with `(C - 1)` to increase the domain
    of appicability.

    Domain of applicability:

      abs(x) <= largest * (1 - 1 / C)  if scale is True
      abs(x) <= largest / C            otherwise

    """
    if C is None:
        C, N, invN = get_veltkamp_splitter_constants(ctx, get_largest(ctx, x))
    elif scale:
        one = ctx.constant(1, x)
        N = C - one
        invN = one / N

    if scale:
        N = ctx.select(abs(x) < one, one, N)
        invN = ctx.select(abs(x) < one, one, invN)

    x_n = x * invN if scale else x

    g = C * x_n
    d = g - x_n
    xh = g - d
    xl = x_n - xh

    return (xh * N, xl * N) if scale else (xh, xl)


def mul_dw(ctx, x, y, xh, xl, yh, yl):
    xyh = x * y
    t1 = (-xyh) + xh * yh
    t2 = t1 + xh * yl
    t3 = t2 + xl * yh
    xyl = t3 + xl * yl
    return xyh, xyl


def mul_dekker(ctx, x, y, C=None):
    """Dekker product:

      x * y = xyh + xyl

    where xyh = RN(x * y) and

      C = 1 + 2 ** ceil(p // 2),

    p is the precision of floating point system.

    It is assumed that no overflow occurs in computations.

    Domain of applicability (approximate):
      -986     <= x, y <= 1007, abs(x * y) < 62940 for float16
      -7.5e33  <= x, y <= 8.3e34                   for float32
      -4.3e299 <= x, y <= 1.3e300                  for float64
      x * y is finite.

    Accuracy:
      (xyh, xyl) is exact,
      xyh + xyl maximal allowed ULP difference is 2.

    Note:
      `x * y` is more accurate that `xyh + xyl`. So, using mul_dekker
      makes only sense when the accuracy of the pair (xyh, xyl) is
      taken into account.
    """
    if C is None:
        largest = get_largest(ctx, x)
        C = get_veltkamp_splitter_constant(ctx, largest)
    xh, xl = split_veltkamp(ctx, x, C)
    yh, yl = split_veltkamp(ctx, y, C)
    return mul_dw(ctx, x, y, xh, xl, yh, yl)


def add_2sum(ctx, x, y, fast=False):
    """Add x and y using 2sum or fast2sum algorithm:

    x + y = s + t

    where s = RN(s + y).

    Domain of applicability:
      abs(x), abs(y) < largest / 2

    Accuracy:
      (s, t) is exact.
      s + t maximal allowed ULP difference is 1.

    Note:
      `x + y` has the same accuracy as `s + t`. So, using add_2sum
      makes only sense when the accuracy of the pair (s, t) is taken
      into account.
    """
    s = x + y
    z = s - x
    if fast:
        t = y - z
    else:
        t = (x - (s - z)) + (y - z)
    return s, t


def is_power_of_two(ctx, x, Q, P, invert=False):
    """Check if x is a power of two.

    Q = 2 ** (p - 1)
    P = 2 ** (p - 1) + 1
    p is the precision of the floating point system.

    Domain of applicability:
      2 ** -24   <= abs(x) < 2 ** 6       for float16
      2 ** -129  <= abs(x) < 2 ** 105     for float32
      2 ** -1074 <= abs(x) < 2 ** 972     for float64

    Accuracy: exact.
    """
    L = P * x
    R = Q * x
    D = L - R
    if invert:
        return D != x
    return D == x


def add_3sum(ctx, x, y, z, Q, P, three_over_two):
    """Add x, y, and z using 3sum algorithm:

    x + y + z = s + e + t

    If s is a nested 3-tuple of ternary conditional operator arguments
    then its value is defined by the following if-blocks:

      if s[0]:                     # if not power_of_two(w)
        result = s[1]              #   result = s1
      elif s[2][0]:                # elif s2 == zh
        result = s[2][1]           #   result = zh
      elif s[2][2][0]:             # elif t == 0:
        result = s[2][2][1]        #   result = s1
      elif s[2][2][2][0]:          # elif g < 0:
        result = s[2][2][2][1]     #   result = zh
      else:                        # else:
        result = s[2][2][2][2]     #   result = s2

    and the same rule applies to e and t.

    Domain of applicability:
      abs(x), abs(y), abs(z) < largest / 4

    Accuracy:
      (s, e, t) is exact.
      s + (e + t) maximal allowed ULP difference is 1.

    Note:
      The accuracy of `s + (e + t)` is higher than that of `x + y + z`.
    """
    xh, xl = add_2sum(ctx, x, y)
    sh, sl = add_2sum(ctx, xh, z)
    vh, vl = add_2sum(ctx, xl, sl)
    zh, zl = add_2sum(ctx, sh, vh, fast=True)
    w = vl + zl
    s1 = zh + w
    d = w - zl
    t = vl - d
    wp = three_over_two * w
    s2 = zh + wp
    g = t * w
    s = ctx.select(
        is_power_of_two(ctx, w, Q, P, invert=True),
        s1,
        ctx.select(s2 == zh, zh, ctx.select(t == 0, s1, ctx.select(g < 0, zh, s2))),
    )
    a = s - zh
    e = w - a
    e, t = add_2sum(ctx, e, t, fast=True)
    return s, e, t


def add_dw(ctx, xh, xl, yh, yl, Q, P, three_over_two):
    """Add two double-word numbers:

    xh + xl + yh + yl = s
    """
    sh, sl = add_2sum(ctx, xh, yh)
    th, tl = add_2sum(ctx, xl, yl)
    gh, gl = add_2sum(ctx, sl, th)
    vh, vl = add_2sum(ctx, sh, gh, fast=True)
    wh, wl = add_2sum(ctx, vl, tl, fast=True)
    zh, zl = add_2sum(ctx, vh, wh, fast=True)
    r, e, t = add_3sum(ctx, zl, wl, gl, Q, P, three_over_two)
    t = e + t
    rp = three_over_two * r
    s1 = zh + r
    s2 = zh + rp
    g = t * r
    return ctx.select(
        is_power_of_two(ctx, t, Q, P, invert=True),
        zh,
        ctx.select(s2 == zh, zh, ctx.select(t == 0, s1, ctx.select(g <= 0, zh, s2))),
    )


def add_4sum(ctx, x, y, z, w, Q, P, three_over_two):
    """Add four numbers:

    x + y + z + w = s

    Domain of applicability:
      abs(x), abs(y), abs(z), abs(w) < largest / 4

    Accuracy: maximal allowed ULP difference is 1.

    Note:
      The accuracy of `s` is higher than that of `x + y + z + w`.
    """
    xh, xl = add_2sum(ctx, x, y)
    yh, yl = add_2sum(ctx, z, w)
    return add_dw(ctx, xh, xl, yh, yl, Q, P, three_over_two)


def dot2(ctx, x, y, z, w, C, Q, P, three_over_two):
    """Dot product:

    x * y + z * w = s

    Emulates fused dot-product.

    Domain of applicability:
      abs(x), abs(y), abs(w), abs(z) < sqrt(largest) / 2

    Accuracy: maximal allowed ULP difference is 3.

    Note:
      The accuracy of `s` is higher than that of `x * y + z * w`.
    """
    xh, xl = mul_dekker(ctx, x, y, C)
    yh, yl = mul_dekker(ctx, z, w, C)
    return add_dw(ctx, xh, xl, yh, yl, Q, P, three_over_two)


def mul_add(ctx, x, y, z, C, Q, P, three_over_two):
    """Multiply and add:

    x * y + z = s

    Emulates fused multiply-add.

    Domain of applicability:
      abs(x), abs(y) < sqrt(largest) / 2
      abs(z) < largest / 2

    Accuracy: maximal allowed ULP difference is 2.

    Note:
      The accuracy of `s` is higher than that of `x * y + z`.
    """
    xh, xl = mul_dekker(ctx, x, y, C)

    # Inlined code of add_dw(xh, xl, c, 0):
    sh, sl = add_2sum(ctx, xh, z)
    gh, gl = add_2sum(ctx, sl, xl)
    zh, zl = add_2sum(ctx, sh, gh, fast=True)
    r, t = add_2sum(ctx, zl, gl)
    rp = three_over_two * r
    g = t * r
    s1 = zh + r
    s2 = zh + rp
    return ctx.select(
        is_power_of_two(ctx, t, Q, P, invert=True),
        zh,
        ctx.select(s2 == zh, zh, ctx.select(t == 0, s1, ctx.select(g <= 0, zh, s2))),
    )


def get_log2_doubleword_and_inverse(ctx, largest):
    # The following coefficients are computed using
    # tools/log2_doubleword.py script:

    if 0:
        # p=44, abserr=1.94704509238075e-31, same as p=32
        fp64 = ctx.constant(0.6931471805598903, largest)
        fp64_ = ctx.constant(5.497923018708371e-14, largest)
    elif 0:
        # p=36, abserr=1.0077949135905144e-28
        # float64: ULP differences and counts: 0: 2092780, 1: 8020
        fp64 = ctx.constant(0.6931471805582987, largest)
        fp64_ = ctx.constant(1.6465949582897082e-12, largest)
    elif 1:
        # p=32, abserr=1.1612227229362532e-26, same expm1 accuracy as p=32
        fp64 = ctx.constant(0.6931471803691238, largest)
        fp64_ = ctx.constant(1.9082149292705877e-10, largest)
    elif 0:
        # p=28, abserr=4.00865610552017e-26, same expm1 accuracy as p=32
        # float64: ULP differences and counts: 0: 2092780, 1: 8020
        fp64 = ctx.constant(0.6931471787393093, largest)
        fp64_ = ctx.constant(1.8206359985041462e-09, largest)
    elif 0:
        # p=26, abserr=2.4688171419345863e-25, same as p=32
        fp64 = ctx.constant(0.6931471675634384, largest)
        fp64_ = ctx.constant(1.2996506893889889e-08, largest)
    else:
        assert 0  # unreachable

    fp32 = ctx.constant(0.69314575, largest)  # p=16, abserr=5.497923e-14
    fp32_ = ctx.constant(1.4286068e-06, largest)  # p=16

    fp16 = ctx.constant(0.6875, largest)  # p=4, abserr=1.43e-06
    fp16_ = ctx.constant(0.005646, largest)  # p=4

    ln2hi = ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16))
    ln2lo = ctx.select(largest > 1e308, fp64_, ctx.select(largest > 1e38, fp32_, fp16_))

    ln2 = ctx.constant(0.693147180559945309417, largest)
    ln2inv = ctx.constant(1.4426950408889634074, largest)
    ln2half = ctx.constant(0.34657359027997265471, largest)
    if hasattr(ln2hi, "reference"):
        ln2 = ln2.reference("ln2", force=True)
        ln2hi = ln2hi.reference("ln2hi", force=True)
        ln2lo = ln2lo.reference("ln2lo", force=True)
        ln2inv = ln2inv.reference("ln2inv", force=True)
        ln2half = ln2half.reference("ln2half", force=True)

    return ln2, ln2hi, ln2lo, ln2inv, ln2half


def argument_reduction_exponent(ctx, x):
    """Return r, k, c such that

      x = k * log(2) + (r + c)

    where `k` is integral, `abs(r + c) / log(2) <= 0.51`, and `(r, c)`
    is a double-word representation of the remainder.

    Within the domain of applicability, the upper limit to `k` is
    `log(largest) / log(2)`, that is,

      k <= 2 * bytesize ** 3

    where bytesize is the byte size of floating point numbers.

    Algorithm
    ---------

    Let (ln2hi, ln2lo) be a double-word representation of log(2), that is,

      ln2hi + ln2lo == log(2)

    where ln2hi, ln2lo are positive fixed-width floating point numbers
    with precision p_dtype, evaluation of log(2) and the addition are
    performed with precision p such that

      p > p_dtype.

    In terms of fixed-width floating pointers, we have

      (x, 0) == (k * ln2hi, k * ln2lo) + (r, c)

    that is

      x == k * ln2hi + r
      0 == k * ln2lo + c

    where we require `abs(r + c) <= log(2) * 0.5`. Hence

      k = floor(x / (ln2hi + ln2lo) + 0.5)
      r = x - k * ln2hi
      c = -k * ln2lo

    Domain of applicability:
      abs(x) < log(largest)

    """
    half = ctx.constant(0.5, x)
    ln2, ln2hi, ln2lo, ln2inv, ln2half = get_log2_doubleword_and_inverse(ctx, get_largest(ctx, x))

    # assume x > 0, then
    #             x < ln2 * (k + 0.5)
    # x / ln2 - 0.5 < k
    # that is
    #   k = ceil(x / ln2 - 0.5) = ceil(x / ln2 + 0.5) - 1  = floor(x / ln2 + 0.5)
    #     [iff `x / ln2 + 0.5` is not integer!, otherwise] = floor(x / ln2 + 0.5) + 1
    # however, we can drop the addition by 1 because the absolute values of
    #  x - k * ln2
    #  x - (k + 1) * ln2
    # are both less than log(2) / 2
    #
    # for negative x, we'll have
    #    -x < ln2 * (-k + 0.5)
    #     k < x / ln2 + 0.5
    # k = floor(x / ln2 + 0.5)

    k = ctx.floor(x * ln2inv + half)
    r = x - k * ln2hi
    c = -k * ln2lo
    return k, r, c


def horner(ctx, x, coeffs, reverse=True):
    """Evaluate a polynomial

     P(x) = coeffs[N] + coeffs[N - 1] * x + ... + coeffs[0] * x ** N

    when reverse is True, otherwise evaluate a polynomial

      P(x) = coeffs[0] + coeffs[1] * x + ... + coeffs[N] * x ** N

    where `N = len(coeffs) - 1`, using Horner's scheme.
    """
    N = len(coeffs) - 1
    if reverse:
        s = ctx.constant(coeffs[0], x)
        indices = range(N)
    else:
        s = ctx.constant(coeffs[N], x)
        indices = reversed(range(N))
    for i in indices:
        s = s * x + coeffs[i]
    return s


def compensated_horner(ctx, x, coeffs, reverse=True):
    """Evaluate a polynomial

     P(x) = coeffs[N] + coeffs[N - 1] * x + ... + coeffs[0] * x ** N

    when reverse is True, otherwise evaluate a polynomial

      P(x) = coeffs[0] + coeffs[1] * x + ... + coeffs[N] * x ** N

    where `N = len(coeffs) - 1`, using compensated Horner's scheme.

    Reference:
      https://hal.science/hal-01578486/document
    """
    N = len(coeffs) - 1
    if reverse:
        s = ctx.constant(coeffs[0], x)
        indices = range(N)
    else:
        s = ctx.constant(coeffs[N], x)
        indices = reversed(range(N))
    r = ctx.constant(0, x)
    for i in indices:
        p, pp = mul_dekker(ctx, s, x)
        s, sg = add_2sum(ctx, p, coeffs[i])
        r = r * x + (pp + sg)
    return s, r


def fast_exponent_by_squaring(ctx, x, n):
    """Evaluate x ** n by squaring."""
    if n == 0:
        return ctx.constant(1, x)
    if n == 1:
        return x
    if n == 2:
        return x * x
    assert n > 0
    r = fast_exponent_by_squaring(ctx, x, n // 2)
    if n % 2 == 0:
        return r * r
    return r * r * x


def canonical_scheme(k, N):
    return k


def horner_scheme(k, N):
    return 1


def estrin_dac_scheme(k, N):
    import math

    return int(math.log(k))


def balanced_dac_scheme(k, N):
    return k // 2


def fast_polynomial(ctx, x, coeffs, reverse=True, scheme=None, _N=None):
    """Evaluate a polynomial

     P(x) = coeffs[N] + coeffs[N - 1] * x + ... + coeffs[0] * x ** N

    when reverse is True, otherwise evaluate a polynomial

      P(x) = coeffs[0] + coeffs[1] * x + ... + coeffs[N] * x ** N

    where `N = len(coeffs) - 1`, using "Fast polynomial evaluation and
    composition" algorithm by G. Moroz.

    scheme is an int-to-int unary function. Examples:

      scheme = lambda k, N: k            # canonical polynomial
      scheme = lambda k, N: 1            # Horner's scheme
      scheme = lambda k, N: int(log(k))  # Estrin' DAC scheme
      scheme = lambda k, N: k // 2       # balanced DAC scheme [default]

    Reference:
      https://hal.science/hal-00846961v3
    """

    if reverse:
        return fast_polynomial(ctx, x, list(reversed(coeffs)), reverse=False, scheme=scheme)

    if scheme is None:
        scheme = balanced_dac_scheme

    N = len(coeffs) - 1
    if _N is None:
        _N = N

    if N == 0:
        return ctx.constant(coeffs[0], x)

    if N == 1:
        return ctx.constant(coeffs[0], x) + ctx.constant(coeffs[1], x) * x

    d = scheme(N, _N)

    if d == 0:
        # evaluate reduced polynomial as it is
        s = ctx.constant(coeffs[0], x)
        for i in range(1, N):
            s += coeffs[i] * fast_exponent_by_squaring(ctx, x, i)
        return s

    # P(x) = coeffs[0] + coeffs[1] * x + ... + coeffs[N] * x ** N
    #      = A(x) * x ** d + B(x)
    # where
    #   A(x) = coeffs[d] + coeffs[d + 1] * x + ... + coeffs[N] * x ** (N - d)
    #   B(x) = coeffs[0] + coeffs[1] * x + ... + coeffs[d - 1] * x ** (d - 1)

    a = fast_polynomial(ctx, x, coeffs[d:], reverse=reverse, scheme=scheme, _N=_N)
    b = fast_polynomial(ctx, x, coeffs[:d], reverse=reverse, scheme=scheme, _N=_N)
    xd = fast_exponent_by_squaring(ctx, x, d)
    return a * xd + b


def laurent(ctx, z, C, m, reverse=False, scheme=None):
    """Compute Laurent polynomial

      sum(C[j] * z ** (j + m), j=0..len(C) - 1)

    where m is the exponent of the first term, it could be negative.

    If reverse is True, compute
      sum(C[len(C) - j - 1] * z ** (j + m), j=0..len(C) - 1)
    """
    if m == 0:
        return fast_polynomial(ctx, z, C, reverse=reverse, scheme=scheme)
    elif m > 0:
        return fast_polynomial(ctx, z, C, reverse=reverse, scheme=scheme) * fast_exponent_by_squaring(ctx, z, m)
    elif -m < len(C):
        rz = ctx.reciprocal(z)
        if reverse:
            N = [ctx.constant(0, z)] + C[m:]
            P = C[:m]
        else:
            N = C[:-m] + [ctx.constant(0, z)]
            P = C[-m:]
        return fast_polynomial(ctx, rz, N, reverse=not reverse, scheme=scheme) + fast_polynomial(
            ctx, z, P, reverse=reverse, scheme=scheme
        )
    else:
        rz = ctx.reciprocal(z)
        return fast_polynomial(ctx, rz, C, reverse=not reverse, scheme=scheme) * fast_exponent_by_squaring(
            ctx, rz, -m - len(C) + 1
        )


def split_tripleword(ctx, x, scale=False):
    """Split floating-point value to triple-word so that

    x == x_hi + x_lo + x_rest
    """
    C1, C2 = get_tripleword_splitter_constants(ctx, get_largest(ctx, x))
    x_hi, x_mid = split_veltkamp(ctx, x, C1, scale=scale)
    x_lo, x_rest = split_veltkamp(ctx, x_mid, C2, scale=scale)
    return x_hi, x_lo, x_rest


def mul_mw(ctx, x, y):
    """Return a multiword product of two multiwords.

    A multiword is a list of fixed-precision floating-point values
    with a smaller precision (`p`) that of the corresponding dtype
    (`p_dtype`). Multiword is a descreasing sequence.

    Multiword x represents a multiprecision floating-point value `X`
    such that `X = sum(MP(x))` where `MP` converts fixed-precision
    floating-point value into a multiprecision floating-point value.

    If `p * 2 + 1 < p_dtype` then the resulting multiword is exact.
    """
    lst = [None] * (len(x) + len(y) - 1)
    for i in reversed(range(len(x))):
        for j in reversed(range(len(y))):
            k = i + j
            if lst[k] is None:
                lst[k] = x[i] * y[j]
            else:
                s, t = add_2sum(ctx, lst[k], x[i] * y[j], fast=True)
                lst[k] = s
                # the correction term is required for float64, for
                # shorter types it appears to be zero.  TODO: check if
                # this is always true.
                lst[k + 1] += t
    return lst


def mul_mw_mod4(ctx, x, y):
    zero = ctx.constant(0, x[0])
    one = ctx.constant(1, x[0])
    four = ctx.constant(4, x[0])

    def rem4(v):
        return ctx.trunc(v / four) * four

    total = zero
    rest = zero
    ss = 0
    for k in reversed(range(len(x) + len(y) - 1)):
        s = zero
        acc = zero
        for i in reversed(range(len(x))):
            j = k - i
            if j < 0 or j >= len(y):
                continue
            xy = x[i] * y[j]
            xy = xy - rem4(xy)
            ss += float(x[i]) * float(y[j])
            s, t = add_2sum(ctx, s, xy, fast=True)
            acc += t
        rest += acc
        total, t = add_2sum(ctx, s, total, fast=True)
        total = total - rem4(total)
        rest += t
    k = ctx.round(total)  # % four
    r = total - k  # % one
    return k % four, r, rest


def mw2dw(ctx, x):
    """Return multiword as a double-word.

    Reference:
      https://userpages.cs.umbc.edu/phatak/645/supl/Ng-ArgReduction.pdf
    """
    y = x[-1]
    for i in reversed(range(len(x) - 1)):
        y = y + x[i]
    t = x[0] - y  # note that Ng-ArgReduction.pdf contains a typo
    for i in range(1, len(x)):
        t = t + x[i]
    return y, t


def argument_reduction_trigonometric(ctx, x):
    """Return k, r, t such that

      x = 2 * pi * N + k * pi / 2 + r + t

    where N is some integral, k is in {0, 1, 2, 3}, and abs(r) < pi / 4.

    Reference:
      https://userpages.cs.umbc.edu/phatak/645/supl/Ng-ArgReduction.pdf
    """
    import numpy

    if isinstance(x, (numpy.float64, numpy.float32, numpy.float16)):
        return argument_reduction_trigonometric_impl(ctx, type(x), x)

    largest = get_largest(ctx, x)
    fp64_k, fp64_r, fp64_t = argument_reduction_trigonometric_impl(ctx, numpy.float64, x)
    fp32_k, fp32_r, fp32_t = argument_reduction_trigonometric_impl(ctx, numpy.float32, x)
    fp16_k, fp16_r, fp16_t = argument_reduction_trigonometric_impl(ctx, numpy.float16, x)
    k = ctx.select(largest > 1e308, fp64_k, ctx.select(largest > 1e38, fp32_k, fp16_k))
    r = ctx.select(largest > 1e308, fp64_r, ctx.select(largest > 1e38, fp32_r, fp16_r))
    t = ctx.select(largest > 1e308, fp64_t, ctx.select(largest > 1e38, fp32_t, fp16_t))
    return k, r, t


def argument_reduction_trigonometric_impl(ctx, dtype, x):

    import numpy
    import functional_algorithms as fa

    def make_constant(v):
        return ctx.constant(v, x)

    two = ctx.constant(2, x)
    two_over_pi_max_length = {numpy.float16: None, numpy.float32: None, numpy.float64: None}[dtype]
    two_over_pi_mw = list(map(make_constant, fa.utils.get_two_over_pi_multiword(dtype, max_length=two_over_pi_max_length)))
    pi_over_two_prec = {numpy.float16: 4, numpy.float32: 20, numpy.float64: 40}[dtype]
    pi_over_two, pi_over_two_lo = map(
        make_constant, fa.utils.get_pi_over_two_multiword(dtype, prec=pi_over_two_prec, max_length=2)
    )
    x_tw = split_tripleword(ctx, x, scale=True)
    k, y, t = mul_mw_mod4(ctx, x_tw, two_over_pi_mw)
    r_hi, tt = add_2sum(ctx, y * pi_over_two, y * pi_over_two_lo + t * pi_over_two, fast=True)
    r_lo = t * pi_over_two_lo + tt
    r = ctx.select(abs(x) < pi_over_two / two, x, r_hi)
    t = ctx.select(abs(x) < pi_over_two / two, ctx.constant(0, x), r_lo)
    return k, r, t


def sine_dw(ctx, x, y):
    # Algorithm copied from stdlib-js/math-base-special-kernel-sin
    S1 = ctx.constant(-1.66666666666666324348e-01, x)
    S2 = ctx.constant(8.33333333332248946124e-03, x)
    S3 = ctx.constant(-1.98412698298579493134e-04, x)
    S4 = ctx.constant(2.75573137070700676789e-06, x)
    S5 = ctx.constant(-2.50507602534068634195e-08, x)
    S6 = ctx.constant(1.58969099521155010221e-10, x)

    zero = ctx.constant(0, x)
    half = ctx.constant(0.5, x)
    z = x * x
    w = z * z
    r = S2 + z * (S3 + z * S4) + (z * w * (S5 + z * S6))
    v = z * x
    r0 = x + v * (S1 + z * r)
    # r1 = x - (((z * (half * y - v * r)) - y) - v * S1)
    r1 = x - (((z * ((half * y) - (v * r))) - y) - (v * S1))
    return ctx.select(y == zero, r0, r1)


def cosine_dw(ctx, x, y):
    # Algorithm copied from stdlib-js/math-base-special-kernel-cos
    # https://svnweb.freebsd.org/base/release/12.2.0/lib/msun/src/k_cos.c?view=markup
    half = ctx.constant(0.5, x)
    one = ctx.constant(1, x)
    z = x * x
    w = z * z

    S1 = ctx.constant(0.0416666666666666, x)
    S2 = ctx.constant(-0.001388888888887411, x)
    S3 = ctx.constant(0.00002480158728947673, x)

    S4 = ctx.constant(-2.7557314351390663e-7, x)
    S5 = ctx.constant(2.087572321298175e-9, x)
    S6 = ctx.constant(-1.1359647557788195e-11, x)
    r = z * (S1 + (z * (S2 + (z * S3))))
    r += w * w * (S4 + (z * (S5 + (z * S6))))

    hz = half * z
    w = one - hz
    return w + (((one - w) - hz) + ((z * r) - (x * y)))
