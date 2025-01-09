"""References:

- Emulation of 3Sum, 4Sum, the FMA and the FD2 instructions in
  rounded-to-nearest floating-point arithmetic. Stef Graillat,
  Jean-Michel Muller. https://hal.science/hal-04624238/
"""


def split_veltkamp(x, C):
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


def mul_dekker(x, y, C):
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
    xh, xl = split_veltkamp(x, C)
    yh, yl = split_veltkamp(y, C)
    xyh = x * y
    t1 = (-xyh) + xh * yh
    t2 = t1 + xh * yl
    t3 = t2 + xl * yh
    xyl = t3 + xl * yl
    return xyh, xyl


def add_2sum(x, y, fast=False):
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


def is_power_of_two(x, Q, P, invert=False):
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


def _is_ternary(inp):
    return type(inp) is tuple and len(inp) == 3


def _select_eval(inp):
    if _is_ternary(inp):
        if isinstance(inp[0], bool):
            if inp[0]:
                return _select_eval(inp[1])
            return _select_eval(inp[2])
        t = type(inp[0])
        if t.__module__ == "numpy" and t.__name__.startswith("bool"):
            if inp[0]:
                return _select_eval(inp[1])
            return _select_eval(inp[2])
    return inp


def _select_apply(op, *inputs):
    """Apply operator to the values in a nested 3-tuple of ternary
    conditional operator arguments.
    """
    if len(inputs) == 1:
        (inp,) = inputs
        if _is_ternary(inp):
            return (inp[0], _select_apply(op, inp[1]), _select_apply(op, inp[2]))
        return op(inp)
    elif len(inputs) == 2:
        inp1, inp2 = inputs
        if _is_ternary(inp1):
            assert not _is_ternary(inp2)
            r1, r2 = _select_apply(op, inp1[1], inp2), _select_apply(op, inp1[2], inp2)
            return (inp1[0], r1[0], r2[0]), (inp1[0], r1[1], r2[1])
        elif _is_ternary(inp2):
            r1, r2 = _select_apply(op, inp1, inp2[1]), _select_apply(op, inp1, inp2[2])
            return (inp2[0], r1, r2[0]), (inp2[0], r1, r2[1])
        else:
            return op(inp1, inp2)
    else:
        raise NotImplementedError(len(inputs))


def add_3sum(x, y, z, Q, P, three_over_two):
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
    xh, xl = add_2sum(x, y)
    sh, sl = add_2sum(xh, z)
    vh, vl = add_2sum(xl, sl)
    zh, zl = add_2sum(sh, vh, fast=True)
    w = vl + zl
    s1 = zh + w
    d = w - zl
    t = vl - d
    wp = three_over_two * w
    s2 = zh + wp
    g = t * w
    select = _select_eval((is_power_of_two(w, Q, P, invert=True), s1, (s2 == zh, zh, (t == 0, s1, (g < 0, zh, s2)))))
    a = _select_apply(lambda s: s - zh, select)
    e = _select_apply(lambda a: w - a, a)
    e, t = _select_apply(lambda e, t: add_2sum(e, t, fast=True), e, t)
    return select, e, t


def add_dw(xh, xl, yh, yl, Q, P, three_over_two):
    """Add two double-word numbers:

    xh + xl + yh + yl = s
    """
    sh, sl = add_2sum(xh, yh)
    th, tl = add_2sum(xl, yl)
    gh, gl = add_2sum(sl, th)
    vh, vl = add_2sum(sh, gh, fast=True)
    wh, wl = add_2sum(vl, tl, fast=True)
    zh, zl = add_2sum(vh, wh, fast=True)
    r, e, t = add_3sum(zl, wl, gl, Q, P, three_over_two)
    t = e + t
    rp = three_over_two * r
    s1 = zh + r
    s2 = zh + rp
    g = t * r
    return _select_eval((is_power_of_two(t, Q, P, invert=True), zh, (s2 == zh, zh, (t == 0, s1, (g <= 0, zh, s2)))))


def add_4sum(x, y, z, w, Q, P, three_over_two):
    """Add four numbers:

    x + y + z + w = s

    Domain of applicability:
      abs(x), abs(y), abs(z), abs(w) < largest / 4

    Accuracy: maximal allowed ULP difference is 1.

    Note:
      The accuracy of `s` is higher than that of `x + y + z + w`.
    """
    xh, xl = add_2sum(x, y)
    yh, yl = add_2sum(z, w)
    return add_dw(xh, xl, yh, yl, Q, P, three_over_two)


def dot2(x, y, z, w, C, Q, P, three_over_two):
    """Dot product:

    x * y + z * w = s

    Emulates fused dot-product.

    Domain of applicability:
      abs(x), abs(y), abs(w), abs(z) < sqrt(largest) / 2

    Accuracy: maximal allowed ULP difference is 3.

    Note:
      The accuracy of `s` is higher than that of `x * y + z * w`.
    """
    xh, xl = mul_dekker(x, y, C)
    yh, yl = mul_dekker(z, w, C)
    return add_dw(xh, xl, yh, yl, Q, P, three_over_two)


def mul_add(x, y, z, C, Q, P, three_over_two):
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
    xh, xl = mul_dekker(x, y, C)

    # Inlined code of add_dw(xh, xl, c, 0):
    sh, sl = add_2sum(xh, z)
    gh, gl = add_2sum(sl, xl)
    zh, zl = add_2sum(sh, gh, fast=True)
    r, t = add_2sum(zl, gl)
    rp = three_over_two * r
    g = t * r
    s1 = zh + r
    s2 = zh + rp
    return _select_eval((is_power_of_two(t, Q, P, invert=True), zh, (s2 == zh, zh, (t == 0, s1, (g <= 0, zh, s2)))))
