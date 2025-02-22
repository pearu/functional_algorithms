"""References:

- Emulation of 3Sum, 4Sum, the FMA and the FD2 instructions in
  rounded-to-nearest floating-point arithmetic. Stef Graillat,
  Jean-Michel Muller. https://hal.science/hal-04624238/
"""


def fma_native(ctx, x: float, y: float, z: float):
    """Evaluate x * y + z."""
    return x * y + z


def fma_upcast(ctx, x: float, y: float, z: float):
    """Evaluate x * y + z using upcast of operands."""
    x_ = ctx.upcast(x)
    y_ = ctx.upcast(y)
    z_ = ctx.upcast(z)
    return ctx.downcast(x_ * y_ + z_)


def fma_upcast2(ctx, x: float, y: float, z: float):
    """Evaluate x * y + z using double upcast of operands."""
    x_ = ctx.upcast2(x)
    y_ = ctx.upcast2(y)
    z_ = ctx.upcast2(z)
    return ctx.downcast2(x_ * y_ + z_)


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
    Q = ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16))

    fp64 = ctx.constant(1 << (53 - 1) + 1, largest)
    fp32 = ctx.constant(1 << (24 - 1) + 1, largest)
    fp16 = ctx.constant(1 << (11 - 1) + 1, largest)
    P = ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16))

    if hasattr(Q, "reference"):
        Q = Q.reference("Qispowof2", force=True)
        P = P.reference("Pispowof2", force=True)
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

    It is assumed that the arithmetical operations use rounding to
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


def split_veltkamp2(ctx, x):
    """2nd Veltkamp splitter:

      x = xh + xl + xh

    Different from split_veltkamp, 2nd Veltkamp splitter can be used
    for large inputs.

    Note that for large x, `xh + xh` may overflow but `xh + xl + xh`
    will never overflow.

    Domain of applicability:

      smallest_normal * (2 * C - 2) <= abs(x) <= largest

    where

      C = 1 + 2 ** ceil(p / 2)

    and p is the precision of the floating point system.
    """
    C, _, _ = get_veltkamp_splitter_constants(ctx, get_largest(ctx, x))
    S = C - ctx.constant(1, x)
    S2 = S + S
    # S is power of 2, so is S2, hence x / S2 is exact when it does
    # not underflow:
    xh, xl = split_veltkamp(ctx, x / S2, C=C, scale=False)
    return xh * S, xl * S2


def mul_dw(ctx, x, y, xh, xl, yh, yl):
    xyh = x * y
    t1 = (-xyh) + xh * yh
    t2 = t1 + xh * yl
    t3 = t2 + xl * yh
    xyl = t3 + xl * yl
    return xyh, xyl


def mul_dekker(ctx, x, y, C=None):
    """Dekker's product:

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


def _mul_series_scalar(ctx, x, y):
    assert type(y) is not tuple
    terms = tuple(x_ * y for x_ in x[1:])
    return ctx._series(terms, dict(unit_index=x[0][0], scaling_exp=x[0][1]))


def _mul_scalar_series(ctx, x, y):
    assert type(x) is not tuple
    terms = tuple(x * y_ for y_ in y[1:])
    return ctx._series(terms, dict(unit_index=y[0][0], scaling_exp=y[0][1]))


def _div_series_scalar(ctx, x, y):
    assert type(y) is not tuple
    terms = tuple(x_ / y for x_ in x[1:])
    return ctx._series(terms, dict(unit_index=x[0][0], scaling_exp=x[0][1]))


def _mul_series_series(ctx, x, y):
    (index1, sexp1), terms1 = x[0], x[1:]
    (index2, sexp2), terms2 = y[0], y[1:]
    assert sexp1 == sexp2, (sexp1, sexp2)

    terms = []
    for n in range(len(terms1) + len(terms2) - 1):
        xy = None
        for i, x in enumerate(terms1):
            for j, y in enumerate(terms2):
                if i + j == n:
                    if xy is None:
                        xy = x * y
                    else:
                        xy += x * y
        assert xy is not None
        terms.append(xy)

    return ctx._series(tuple(terms), dict(unit_index=index1 + index2, scaling_exp=sexp1))


def mul_series(ctx, x, y):
    """Multiply x and y which may be series."""
    x = ctx._get_series_operands(x)
    y = ctx._get_series_operands(y)
    if type(x) is tuple:
        if type(y) is tuple:
            return _mul_series_series(ctx, x, y)
        return _mul_series_scalar(ctx, x, y)
    elif type(y) is tuple:
        return _mul_scalar_series(ctx, x, y)
    return x * y


def div_series(ctx, x, y):
    """Divide x by y where x may be series."""
    x = ctx._get_series_operands(x)
    y = ctx._get_series_operands(y)
    if type(x) is tuple:
        if type(y) is tuple:
            assert 0  # not impl
        return _div_series_scalar(ctx, x, y)
    elif type(y) is tuple:
        assert 0  # not impl
    return x / y


def _add_series_series(ctx, x, y):

    def op(x, y):
        if x is None:
            return y
        if y is None:
            return x
        return x + y

    return _binaryop_series_series(ctx, x, y, op)


def _subtract_series_series(ctx, x, y):

    def op(x, y):
        if x is None:
            return -y
        if y is None:
            return x
        return x - y

    return _binaryop_series_series(ctx, x, y, op)


def _binaryop_series_series(ctx, x, y, op):
    swapped = False
    if x[0][0] < y[0][0]:
        x, y = y, x
        swapped = True

    (index1, sexp1), terms1 = x[0], x[1:]
    (index2, sexp2), terms2 = y[0], y[1:]
    assert sexp1 == sexp2, (sexp1, sexp2)

    terms = []

    for n in range(max(len(terms1), index1 - index2 + len(terms2))):
        k = n - (index1 - index2)
        if n < len(terms1):
            if k >= 0 and k < len(terms2):
                if swapped:
                    terms.append(op(terms2[k], terms1[n]))
                else:
                    terms.append(op(terms1[n], terms2[k]))
            else:
                if swapped:
                    terms.append(op(None, terms1[n]))
                else:
                    terms.append(op(terms1[n], None))
        elif k >= 0 and k < len(terms2):
            if swapped:
                terms.append(op(terms2[k], None))
            else:
                terms.append(op(None, terms2[k]))
        else:
            terms.append(ctx.constant(0, terms1[0]))

    return ctx._series(tuple(terms), dict(unit_index=index1, scaling_exp=sexp1))


def add_series(ctx, x, y):
    """Add x and y which may be series."""

    x = ctx._get_series_operands(x)
    y = ctx._get_series_operands(y)

    if type(x) is tuple:
        if type(y) is tuple:
            return _add_series_series(ctx, x, y)
        return _add_series_series(ctx, x, ((0, 0), y))
    elif type(y) is tuple:
        return _add_series_series(ctx, ((0, 0), x), y)
    return x + y


def subtract_series(ctx, x, y):
    """Subtract y from x which may be series."""

    x = ctx._get_series_operands(x)
    y = ctx._get_series_operands(y)

    if type(x) is tuple:
        if type(y) is tuple:
            return _subtract_series_series(ctx, x, y)
        return _subtract_series_series(ctx, x, ((0, 0), y))
    elif type(y) is tuple:
        return _subtract_series_series(ctx, ((0, 0), x), y)
    return x - y


def mul_series_dekker(ctx, x, y, C=None):
    """Dekker's product on series:

        sum(x) * sum(y) = sum(xy)

    Series are represented by a tuple

      ((index, sexp), term1, term2, ..., termN)
    """
    x = ctx._get_series_operands(x)
    y = ctx._get_series_operands(y)

    def terms_add(terms, index, *operands):
        for i, v in enumerate(operands):
            if index + i < len(terms):
                if 1:
                    terms[index + i] += v
                else:
                    h, l = add_2sum(ctx, terms[index + i], v)
                    terms[index + i] = h
                    if index + i <= len(terms):
                        terms.append(l)
                    else:
                        terms[index + i + 1] += l
            else:
                assert index + i == len(terms)
                terms.append(v)

    offset = 10000

    if type(x) is tuple:
        if type(y) is tuple:
            assert x[0][1] == y[0][1]
            # (x1, x2, ...) * (y1, y2, ...)
            #  x11h, x11l = mul_dekker(x1, y1)
            #  x12h, x12l = mul_dekker(x1, y2)
            #  ...
            #  xijh == xjih, xijl == xjil
            # = (x11h, x11h * x12l + x11l * x12h + x21h * x12h + ..., ...)
            # = (..., ... + x[i,j]h, ... + x[i,j]l + x[i+1,j]h + x[i, j+1]h, ... + x[i+1,j]l + x[i,j+1]l + x[i+1,j+1]h, ... + x[i+1,j+1]l, ...)
            terms = []
            for i, x_ in enumerate(x[1:]):
                for j, y_ in enumerate(y[1:]):
                    if i + j >= offset:
                        terms_add(terms, i + j, x_ * y_)
                    else:
                        terms_add(terms, i + j, *mul_dekker(ctx, x_, y_, C=C))
            return ctx._series(tuple(terms), dict(unit_index=x[0][0] + y[0][0], scaling_exp=x[0][1]))
        else:
            # (x1, x2, ...) * y
            #   x1h, x1l = mul_dekker(x1, y)
            #   x2h, x2l = mul_dekker(x2, y)
            #   ...
            # = (x1h, x1l + x2h, x2l + x3h, ..., xNl)
            terms = []
            for i, x_ in enumerate(x[1:]):
                if i >= offset:
                    terms_add(terms, i, x_ * y)
                else:
                    terms_add(terms, i, *mul_dekker(ctx, x_, y, C=C))
            return ctx._series(tuple(terms), dict(unit_index=x[0][0], scaling_exp=x[0][1]))
    elif type(y) is tuple:
        terms = []
        for i, y_ in enumerate(y[1:]):
            if i >= offset:
                terms_add(terms, i, x * y_)
            else:
                terms_add(terms, i, *mul_dekker(ctx, x, y_, C=C))
        return ctx._series(tuple(terms), dict(unit_index=y[0][0], scaling_exp=y[0][1]))
    return ctx._series(mul_dekker(ctx, x, y, C=C), dict(unit_index=0, scaling_exp=0))


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


def add_dw(ctx, xh, xl, yh, yl, Q=None, P=None, three_over_two=None):
    """Add two double-word numbers:

    xh + xl + yh + yl = s
    """
    if Q is None:
        largest = get_largest(ctx, xh)
        Q, P = get_is_power_of_two_constants(ctx, largest)
        three_over_two = ctx.constant(1.5, largest)
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


def add_4sum(ctx, x, y, z, w, Q=None, P=None, three_over_two=None):
    """Add four numbers:

    x + y + z + w = s

    Domain of applicability:
      abs(x), abs(y), abs(z), abs(w) < largest / 4

    Accuracy: maximal allowed ULP difference is 1.

    Note:
      The accuracy of `s` is higher than that of `x + y + z + w`.
    """
    if Q is None:
        largest = get_largest(ctx, x)
        Q, P = get_is_power_of_two_constants(ctx, largest)
        three_over_two = ctx.constant(1.5, largest)
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


def mul_add(ctx, x, y, z, C=None, Q=None, P=None, three_over_two=None):
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
    if C is None:
        largest = get_largest(ctx, x)
        C, _, _ = get_veltkamp_splitter_constants(ctx, largest)
        Q, P = get_is_power_of_two_constants(ctx, largest)
        three_over_two = ctx.constant(1.5, largest)
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
    zero = ctx.constant(0, x)
    if n == 0:
        return ctx.constant(1, x)
    elif n == 1:
        return x
    elif n == 2:
        return ctx.fma(x, x, zero)
    assert n > 0
    r = fast_exponent_by_squaring(ctx, x, n // 2)
    if n % 2 == 0:
        return ctx.fma(r, r, zero)
    return ctx.fma(ctx.fma(r, r, zero), x, zero)


def fast_exponent_by_squaring_dekker(ctx, x, n: int, depth=0):
    """Evaluate x ** n by squaring using Dekker's product."""
    if n == 0:
        return ctx.constant(1, x)
    elif n == 1:
        return x
    elif n == 2:
        if depth == 0:
            # `sum(mul_dekker(x, x))` is less accurate than `x * x`.
            # However, `sum(mul_dekker(x, mul_dekker(x, x)))` is more
            # accurate that `x * x * x`.
            return x * x
        return mul_series_dekker(ctx, x, x)
    else:
        # TODO: introduce dekker_inverse and use
        #   return inverse_dekker(ctx, fast_exponent_by_squaring_dekker(ctx, x, -n))
        # or
        #   return fast_exponent_by_squaring_dekker(ctx, inverse_dekker(ctx, x), -n)
        # which ever has better accuracy properties
        assert n > 0, n  # unreachable
    r = fast_exponent_by_squaring_dekker(ctx, x, n // 2, depth=depth + 1)
    y = fast_exponent_by_squaring_dekker(ctx, r, 2, depth=depth + 1)
    if n % 2 == 0:
        return y
    return mul_series_dekker(ctx, x, y)


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

    when reverse is True, otherwise, evaluate a polynomial

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

    def w(c):
        if isinstance(c, type(x)):
            return c
        return ctx.constant(c, x)

    if N == 0:
        return w(coeffs[0])

    if N == 1:
        return ctx.fma(w(coeffs[1]), x, w(coeffs[0]))

    d = scheme(N, _N)

    if d == 0:
        # evaluate reduced polynomial as it is
        s = w(coeffs[0])
        for i in range(1, N):
            s = ctx.fma(w(coeffs[i]), fast_exponent_by_squaring(ctx, x, i), s)
        return s

    # P(x) = coeffs[0] + coeffs[1] * x + ... + coeffs[N] * x ** N
    #      = A(x) * x ** d + B(x)
    # where
    #   A(x) = coeffs[d] + coeffs[d + 1] * x + ... + coeffs[N] * x ** (N - d)
    #   B(x) = coeffs[0] + coeffs[1] * x + ... + coeffs[d - 1] * x ** (d - 1)

    a = fast_polynomial(ctx, x, coeffs[d:], reverse=reverse, scheme=scheme, _N=_N)
    b = fast_polynomial(ctx, x, coeffs[:d], reverse=reverse, scheme=scheme, _N=_N)
    xd = fast_exponent_by_squaring(ctx, x, d)
    return ctx.fma(a, xd, b)


def fast_polynomial2(ctx, x, coeffs, reverse=True, scheme=None, _N=None):
    """Evaluate a polynomial

     P(x) = coeffs[N] + coeffs[N - 1] * (x**2) + ... + coeffs[0] * (x**2) ** N

    when reverse is True, otherwise, evaluate a polynomial

      P(x) = coeffs[0] + coeffs[1] * (x**2) + ... + coeffs[N] * (x**2) ** N

    where `N = len(coeffs) - 1`.

    See also fast_polynomial2.
    """

    if reverse:
        return fast_polynomial2(ctx, x, list(reversed(coeffs)), reverse=False, scheme=scheme)

    if scheme is None:
        scheme = balanced_dac_scheme

    N = len(coeffs) - 1
    if _N is None:
        _N = N

    def w(c):
        if isinstance(c, type(x)):
            return c
        return ctx.constant(c, x)

    if N == 0:
        return w(coeffs[0])

    if N == 1:
        return w(coeffs[0]) + x * w(coeffs[1]) * x

    d = scheme(N, _N)

    if d == 0:
        # evaluate reduced polynomial as it is
        s = w(coeffs[0])
        for i in range(1, N):
            c = fast_exponent_by_squaring(ctx, x, i)
            s += c * w(coeffs[i]) * c
        return s

    # P(x) = coeffs[0] + coeffs[1] * x + ... + coeffs[N] * x ** N
    #      = A(x) * x ** d + B(x)
    # where
    #   A(x) = coeffs[d] + coeffs[d + 1] * x + ... + coeffs[N] * x ** (N - d)
    #   B(x) = coeffs[0] + coeffs[1] * x + ... + coeffs[d - 1] * x ** (d - 1)

    a = fast_polynomial2(ctx, x, coeffs[d:], reverse=reverse, scheme=scheme, _N=_N)
    b = fast_polynomial2(ctx, x, coeffs[:d], reverse=reverse, scheme=scheme, _N=_N)
    xd = fast_exponent_by_squaring(ctx, x, d)
    return xd * a * xd + b


def fast_polynomial_dekker(ctx, x, coeffs, reverse=True, scheme=None, _N=None):
    """Evaluate a polynomial using Dekker's product.

    See also fast_polynomial.
    """
    if reverse:
        return fast_polynomial_dekker(ctx, x, list(reversed(coeffs)), reverse=False, scheme=scheme)

    if scheme is None:
        scheme = balanced_dac_scheme

    N = len(coeffs) - 1
    if _N is None:
        _N = N

    def w(c):
        if type(c) is tuple:  # c is series
            return c
        if type(x) is tuple:  # x is series
            x_ = x[1]
        else:
            x_ = x
        if isinstance(c, type(x_)):
            return c
        return ctx.constant(c, x_)

    if N == 0:
        return w(coeffs[0])

    if N == 1:
        return add_series(ctx, w(coeffs[0]), mul_series_dekker(ctx, w(coeffs[1]), x))

    d = scheme(N, _N)

    if d == 0:
        # evaluate reduced polynomial as it is
        s = w(coeffs[0])
        for i in range(1, N):
            s = add_series(ctx, s, mul_series_dekker(ctx, w(coeffs[i]), fast_exponent_by_squaring_dekker(ctx, x, i)))
        return s

    # P(x) = coeffs[0] + coeffs[1] * x + ... + coeffs[N] * x ** N
    #      = A(x) * x ** d + B(x)
    # where
    #   A(x) = coeffs[d] + coeffs[d + 1] * x + ... + coeffs[N] * x ** (N - d)
    #   B(x) = coeffs[0] + coeffs[1] * x + ... + coeffs[d - 1] * x ** (d - 1)

    a = fast_polynomial_dekker(ctx, x, coeffs[d:], reverse=reverse, scheme=scheme, _N=_N)
    b = fast_polynomial_dekker(ctx, x, coeffs[:d], reverse=reverse, scheme=scheme, _N=_N)
    xd = fast_exponent_by_squaring_dekker(ctx, x, d)
    return add_series(ctx, b, mul_series_dekker(ctx, a, xd))


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


def sin_kernel(ctx, k, r, t):
    sn = sine_dw(ctx, r, t)
    cs = cosine_dw(ctx, r, t)
    if k == 0:
        return sn
    if k == 1:
        return cs
    if k == 2:
        return -sn
    if k == 3:
        return -cs
    assert 0, k  # unreachable


def sine_taylor(ctx, x, order=7, split=False):
    """Return sine of x using Taylor series approximation:

    S(x) = x - x ** 3 / 6 + ... + O(x ** (order + 2))

    If split is True, return `sh, sl` such that

      S(x) = sh + sl

    where the high and low values are obtained from Dekker's split of
    `x ** 2`.

    For best accuracy, use the order argument according to the
    following table [assuming abs(x) <= p / 4]:

    dtype   | order | mean ULP error   | numpy comparsion
    --------+-------+------------------+-----------------
    float16 |  7    |  91 / 14921      |  72 / 14921
    float32 |  9    | 356 / 1_000_000  | 276 / 1_000_000
    float64 |  15   |  56 / 1_000_000  |   5 / 1_000_000

    For the corresponding order choices, we have

      sine_taylor(x, order, split=False) == sum(*sine_taylor(x, order, split=True))
    """

    """
    For float16, consider
      S(x) = x - x ** 3 / 6 + x ** 5 / 120
    We write
      S(x) = P(x * x, C=[x, -x/6, x/120])
    where
      P(y, C) = C[0] + C[1] * y + C[2] * y ** 2
    """
    if not split:
        zero = ctx.constant(0, x)
        C, f = [x], 1
        for i in range(3, order + 1, 2):
            f *= -i * (i - 1)
            C.append(ctx.fma(x, ctx.constant(1 / f, x), zero))
        # Horner's scheme is most accurate
        xx = ctx.fma(x, x, zero)
        return fast_polynomial(ctx, xx, C, reverse=False, scheme=[None, horner_scheme, estrin_dac_scheme, canonical_scheme][1])
    """
    For float16, consider
      x * x = xxh + xxl
    which is an exact representation of the square x ** 2.
    Let's find
      P(xxh + xxl, C) ~ P(xxh, C) + P'(xxh, C) * xxl
        = C[0] + C[1] * xxh + C[2] * xxh ** 2 +
                 (C[1] + 2 * C[2] * xxh) * xxl
        = P(xxh, C=C0) + P(xxh, C=C1) * xxl
    where
      C1 = [i * c for i, c in enumerate(C0) if i > 0]
    """
    xxh, xxl = mul_dekker(ctx, x, x)
    C0 = [x]
    C1 = []
    f = 1
    for i in range(3, order + 1, 2):
        C1.append(x / ctx.constant(f * (i + 1), x))
        f *= -i * (i - 1)
        C0.append(x / ctx.constant(f, x))
    # Horner's scheme is most accurate
    p0 = fast_polynomial(ctx, xxh, C0, reverse=False, scheme=[None, horner_scheme, estrin_dac_scheme, canonical_scheme][1])
    p1 = fast_polynomial(ctx, xxh, C1, reverse=False, scheme=[None, horner_scheme, estrin_dac_scheme, canonical_scheme][1])
    return p0, p1 * xxl


def sine_taylor_dekker(ctx, x, order=7):
    """Return sine of x using Taylor series approximation and Dekker's product.

    See also sine_taylor.
    """
    C, f = [x], 1
    for i in range(3, order + 1, 2):
        f *= -i * (i - 1)
        f1 = ctx.constant(1 / f, x)
        C.append(mul_series(ctx, x, f1))
    xx = mul_series(ctx, x, x)
    # Horner's scheme is most accurate
    return fast_polynomial_dekker(
        ctx, xx, C, reverse=False, scheme=[None, horner_scheme, estrin_dac_scheme, canonical_scheme][1]
    )


def cosine_taylor(ctx, x, order=6, split=False, drop_leading_term=False):
    """Return sine of x using Taylor series approximation:

    C(x) = 1 - x ** 2 / 2 + x ** 4 / 24 - ... + O(x ** (order + 2))

    If split is True, return `ch, cl` such that

      C(x) = ch + cl

    where the high and low values are obtained from Dekker's split of
    `x ** 2`.

    For best accuracy, use the order argument according to the
    following table [assuming abs(x) <= p / 4]:

    dtype   | order | mean ULP error    | numpy comparsion
    --------+-------+-------------------+-----------------
    float16 |  7    |  174 / 14921      |  80 / 14921
    float32 |  9    | 1382 / 1_000_000  | 725 / 1_000_000
    float64 |  19   |  146 / 1_000_000  |   0 / 1_000_000

    For the corresponding order choices, we have

      cosine_taylor(x, order, split=False) == sum(*cosine_taylor(x, order, split=True))

    To compute cosm1(x), use drop_leading_term=True.
    """
    """
    An alternative evaluation scheme:

    C(x) = 1 - x ** 2 / 2 + x ** 4 / 24 - ... + O(x ** (order + 2))
         = 1 + x ** 4 / 4! + ... + - (x ** 2 / 2 + x ** 6 / 6! + x ** 10 / 10!)
         = P1(x ** 4, C=C1) - x**2 * P(x**4, C=C2)

    where

      C1 = [1, 1/4!, 1/8!, ...]
      C2 = [1/2!, 1/6!, 1/10!, ...]
    """
    if not split:
        zero = ctx.constant(0, x)
        one = ctx.constant(1, x)
        if 0:
            f = 1
            iC1 = [1]
            iC2 = []
            for i in range(2, order, 2):
                f *= i * (i - 1)
                if i % 4 == 0:
                    iC1.append(f)
                else:
                    iC2.append(f)
            s1 = iC1[-1]
            s2 = iC2[-1]
            C1 = [s1 // c for c in iC1]
            C2 = [s2 // c for c in iC2]
            assert len(C1) == len(C2), (C1, C2)
            xx = x * x
            xxxx = xx * xx
            C = [c1 / s1 - x * (c2 / s2) * x for c1, c2 in zip(C1, C2)]
            return fast_polynomial(
                ctx, xxxx, C, reverse=False, scheme=[None, horner_scheme, estrin_dac_scheme, canonical_scheme][1]
            )

        f = 1
        xx = ctx.fma(x, x, zero)
        C = []
        for i in range(2, order, 2):
            f *= -i * (i - 1)
            C.append(ctx.fma(xx, ctx.constant(1 / f, x), zero))
        # Horner's scheme is most accurate
        p = fast_polynomial(ctx, xx, C, reverse=False, scheme=[None, horner_scheme, estrin_dac_scheme, canonical_scheme][1])
        if drop_leading_term:
            return p
        return ctx.fma(one, one, p)

    xxh, xxl = mul_dekker(ctx, x, x)
    C0 = []
    C1 = []
    f = 1
    xx = x * x
    for i in range(2, order, 2):
        C1.append(xx * ctx.constant(1 / (f * (i + 1)), x))
        f *= -i * (i - 1)
        C0.append(xx / ctx.constant(f, x))
    # Horner's scheme is most accurate
    p0 = fast_polynomial(ctx, xxh, C0, reverse=False, scheme=[None, horner_scheme, estrin_dac_scheme, canonical_scheme][1])
    p1 = fast_polynomial(ctx, xxh, C1, reverse=False, scheme=[None, horner_scheme, estrin_dac_scheme, canonical_scheme][1])
    if drop_leading_term:
        return p0, p1 * xxl
    return ctx.constant(1, x) + p0, p1 * xxl


def sine_pade(ctx, x, variant=None):
    # See https://math.stackexchange.com/questions/2196371/how-to-approximate-sinx-using-pad%C3%A9-approximation

    if variant is None:
        variant = 13, 4

    P, Q = {
        # k=2 : p/q = (-1/12096*x^7 + 13/2160*x^5 - 11/72*x^3 + x)/(1/72*x^2 + 1)
        (11, 2): ([1, -11 / 72, 13 / 2160, -1 / 12096], [1, 1 / 72]),
        # k=3 : p/q = (551/166320*x^5 - 53/396*x^3 + x)/(5/11088*x^4 + 13/396*x^2 + 1)
        (11, 3): ([1, -53 / 396, 551 / 166320], [1, 13 / 396, 5 / 11088]),
        # k=4 : p/q = (-127/1240*x^3 + x)/(551/9374400*x^6 + 53/22320*x^4 + 239/3720*x^2 + 1)
        (11, 4): ([1, -127 / 1240], [1, 239 / 3720, 53 / 22320, 551 / 9374400]),
        # k=5 : p/q = (x)/(127/604800*x^8 + 31/15120*x^6 + 7/360*x^4 + 1/6*x^2 + 1)
        (11, 5): ([1], [1, 1 / 6, 7 / 360, 31 / 15120, 127 / 604800]),
        # k=2 : p/q = (19/19958400*x^9 - 17/138600*x^7 + 3/440*x^5 - 26/165*x^3 + x)/(1/110*x^2 + 1)
        (13, 2): ([1, -26 / 165, 3 / 440, -17 / 138600, 19 / 19958400], [1, 1 / 110]),
        # k=3 : p/q = (-121/2268000*x^7 + 601/118800*x^5 - 241/1650*x^3 + x)/(19/118800*x^4 + 17/825*x^2 + 1)
        (13, 3): ([1, -241 / 1650, 601 / 118800, -121 / 2268000], [1, 17 / 825, 19 / 118800]),
        # k=4 : p/q = (12671/4363920*x^5 - 2363/18183*x^3 + x)/(121/16662240*x^6 + 601/872784*x^4 + 445/12122*x^2 + 1)
        (13, 4): ([1, -2363 / 18183, 12671 / 4363920], [1, 445 / 12122, 601 / 872784, 121 / 16662240]),
        # k=5 : p/q = (-2555/25146*x^3 + x)/(12671/7604150400*x^8 + 2363/31683960*x^6 + 3787/1508760*x^4 + 818/12573*x^2 + 1)
        (13, 5): ([1, -2555 / 25146], [1, 818 / 12573, 3787 / 1508760, 2363 / 31683960, 12671 / 7604150400]),
        # k=6 : p/q = (x)/(73/3421440*x^10 + 127/604800*x^8 + 31/15120*x^6 + 7/360*x^4 + 1/6*x^2 + 1)
        (13, 6): ([1], [1, 1 / 6, 7 / 360, 31 / 15120, 127 / 604800, 73 / 3421440]),
        # k=2 : p/q = (1/24216192000*x^13 - 1/83825280*x^11 + 23/12700800*x^9 - 1/6300*x^7 + 19/2520*x^5 - 17/105*x^3 + x)/(1/210*x^2 + 1)
        (17, 2): ([1, -17 / 105, 19 / 2520, -1 / 6300, 23 / 12700800, -1 / 83825280, 1 / 24216192000], [1, 1 / 210]),
        # k=3 : p/q = (-911/250637587200*x^11 + 3799/3797539200*x^9 - 89/753480*x^7 + 2503/376740*x^5 - 151/966*x^3 + x)/(3/83720*x^4 + 5/483*x^2 + 1)
        (17, 3): (
            [1, -151 / 966, 2503 / 376740, -89 / 753480, 3799 / 3797539200, -911 / 250637587200],
            [1, 5 / 483, 3 / 83720],
        ),
        # k=4 : p/q = (1768969/4763930371200*x^9 - 36317/472612140*x^7 + 80231/14321580*x^5 - 8234/55083*x^3 + x)/(911/1890448560*x^6 + 3799/28643160*x^4 + 631/36722*x^2 + 1)
        (17, 4): (
            [1, -8234 / 55083, 80231 / 14321580, -36317 / 472612140, 1768969 / 4763930371200],
            [1, 631 / 36722, 3799 / 28643160, 911 / 1890448560],
        ),
        # k=5 : p/q = (-62077121/1727021696400*x^7 + 9713777/2242885320*x^5 - 2020961/14377470*x^3 + x)/(1768969/124345562140800*x^8 + 36317/12335869260*x^6 + 26015/74762844*x^4 + 187642/7188735*x^2 + 1)
        (17, 5): (
            [1, -2020961 / 14377470, 9713777 / 2242885320, -62077121 / 1727021696400],
            [1, 187642 / 7188735, 26015 / 74762844, 36317 / 12335869260, 1768969 / 124345562140800],
        ),
        # k=6 : p/q = (1074305779/407195104680*x^5 - 4749115/37288929*x^3 + x)/(62077121/45149793206918400*x^10 + 9713777/58636095073920*x^8 + 5513233/407195104680*x^6 + 3873323/4524390052*x^4 + 2931413/74577858*x^2 + 1)
        # k=7 : p/q = (-286685/2828954*x^3 + x)/(1074305779/924837658512768000*x^12 + 135689/2419774093440*x^10 + 210431/95052854400*x^8 + 426523/5346723060*x^6 + 1300789/509211720*x^4 + 277211/4243431*x^2 + 1)
    }[variant]

    p = fast_polynomial(ctx, x * x, P, reverse=False, scheme=[None, horner_scheme, estrin_dac_scheme][1]) * x
    q = fast_polynomial(ctx, x * x, Q, reverse=False, scheme=[None, horner_scheme, estrin_dac_scheme][1])
    return p / q


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
