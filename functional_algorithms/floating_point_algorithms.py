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


def get_veltkamp_splitter_constant(ctx, largest: float):
    """Return 2 ** s + 1 where s = ceil(p / 2) and s is the precision of
    the floating point number system.

    Using `largest` to detect the floating point type: float16,
    float32, or float64.
    """
    fp64 = ctx.constant(2 ** (54 // 2) + 1, largest)
    fp32 = ctx.constant(2 ** (24 // 2) + 1, largest)
    fp16 = ctx.constant(2 ** (12 // 2) + 1, largest)
    r = ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16))
    if hasattr(r, "reference"):
        r = r.reference("veltkamp_splitter_constant", force=True)
    return r


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


def split_veltkamp(ctx, x, C):
    """Veltkamp splitter:

      x = xh + xl

    where the coefficient C defines the bit-location of the floating
    point number splitting. For instance, with

      C = 1 + 2 ** ceil(p / 2)

    where p is the precision of the floating point system, xh and xl
    significant parts fit into p / 2 bits.

    It is assumed that the aritmetical operations use rounding to
    nearest and C * x does not overflow.

    Domain of applicability (approximate):

      -986     <= x <= 1007       for float16
      -7.5e33  <= x <= 8.3e34     for float32
      -4.3e299 <= x <= 1.3e300    for float64
    """
    g = C * x
    d = x - g
    xh = g + d
    xl = x - xh
    return xh, xl


def mul_dekker(ctx, x, y, C):
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
    xh, xl = split_veltkamp(ctx, x, C)
    yh, yl = split_veltkamp(ctx, y, C)
    xyh = x * y
    t1 = (-xyh) + xh * yh
    t2 = t1 + xh * yl
    t3 = t2 + xl * yh
    xyl = t3 + xl * yl
    return xyh, xyl


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

    fp64 = ctx.constant(0.6931471805582987, largest)  # p=36, abserr=1.0077949135905144e-28
    fp64 = ctx.constant(0.6931471803691238, largest)  # p=32, abserr=1.1612227229362532e-26, same expm1 accuracy as p=32
    # fp64 = ctx.constant(0.6931471787393093, largest)  # p=28, abserr=4.00865610552017e-26, same expm1 accuracy as p=32
    # fp64 = ctx.constant(0.6931471675634384, largest)  # p=26, abserr=2.4688171419345863e-25, same as p=32
    # fp64 = ctx.constant(0.6931471805598903, largest)  # p=44, abserr=1.94704509238075e-31, same as p=32

    fp32 = ctx.constant(0.69314575, largest)  # p=16, abserr=5.497923e-14
    fp16 = ctx.constant(0.6875, largest)  # p=4, abserr=1.43e-06
    ln2hi = ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16))

    fp64 = ctx.constant(1.6465949582897082e-12, largest)  # p=36
    fp64 = ctx.constant(1.9082149292705877e-10, largest)  # p=32
    # fp64 = ctx.constant(1.8206359985041462e-09, largest)  # p=28
    # fp64 = ctx.constant(1.2996506893889889e-08, largest)  # p=26
    # fp64 = ctx.constant(5.497923018708371e-14, largest)  # p=44

    fp32 = ctx.constant(1.4286068e-06, largest)  # p=16
    fp16 = ctx.constant(0.005646, largest)  # p=4
    ln2lo = ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16))

    ln2inv = ctx.constant(1.4426950408889634074, largest)
    ln2half = ctx.constant(0.34657359027997265471, largest)
    if hasattr(ln2hi, "reference"):
        ln2hi = ln2hi.reference("ln2hi", force=True)
        ln2lo = ln2lo.reference("ln2lo", force=True)
        ln2inv = ln2inv.reference("ln2inv", force=True)
        ln2half = ln2half.reference("ln2half", force=True)

    return ln2hi, ln2lo, ln2inv, ln2half


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
    with precision p_dtype, log(2) and the addition are performed with
    precision p such that

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
    ln2hi, ln2lo, ln2inv, ln2half = get_log2_doubleword_and_inverse(ctx, get_largest(ctx, x))

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
