"""Definitions of functional algorithms for math functions with
complex and float inputs. The aim is to provide algorithms that are
accurate on the whole complex plane or real line.
"""

# This module provides only the definitions of algorithms. As such, it
# should not import any other Python module or package except the ones
# that provide definitions of algorithms to be used here.
#
# Each math function should provide wrapper functions starting with
# `complex_` and `real_` prefix so that input-domain specific
# doc-strings could be provided.


def real_square(ctx, x: float):
    """Square on real input: x * x"""
    return square(ctx, x)


def complex_square(ctx, z: complex):
    """Square on complex input:

    If abs(z.real) == abs(z.imag) then
        square(z).real = 0
    else
        square(z).real = (z.real - z.imag) * (z.real + z.imag)
    square(z).imag = 2 * (z.real * z.imag)
    """
    return square(ctx, z)


def square(ctx, z: complex | float):
    """Square on real and complex inputs.

    See complex_square and real_square for more information.
    """
    if z.is_complex:
        x = z.real
        y = z.imag
        # We'll use
        #   x * x - y * y = (x - y) * (x + y)
        # but we'll still treat `abs(x) == abs(y)` case separately to
        # avoid nan real part (e.g. as when taking x = inf and y =
        # -inf) while it should be 0.
        #
        # Notice that 2 * (x * y) is not the same as 2 * x * y when `2
        # * x` overflows (e.g. take x = finfo(..).max) while `x * y`
        # doesn't (e.g. take y such that `abs(y) < 1`).
        z_sq = ctx.complex(ctx.select(abs(x) == abs(y), 0, (x - y) * (x + y)), 2 * (x * y))
    else:
        z_sq = z * z
    return ctx(z_sq)


def hypot(ctx, x: float, y: float):
    """Square root of the sum of the squares of x and y

    Define:
      mn = min(abs(x), abs(y))
      mn = max(abs(x), abs(y))

    If mn == mx then
      hypot(x, y) = mx * sqrt(2)
    else
      r = square(mn / mx)
      sq = sqrt(1 + r)
      if sq == 1 and r > 0 then
        hypot(x, y) = mx + mx * r / 2
      else
        hypot(x, y) = mx * sq
    """
    assert not (x.is_complex or y.is_complex), (x, y)
    mx = ctx.maximum(abs(x), abs(y))
    mn = ctx.minimum(abs(x), abs(y))
    mn_over_mx = mn / mx
    r = ctx.square(mn_over_mx)
    sqa = ctx.sqrt(1 + r)
    # detect underflow for small r:
    if ctx.alt is None:
        sqrt_two = ctx.sqrt(ctx.constant(2, mx))
    else:
        sqrt_two_ = ctx.alt.sqrt(2)
        sqrt_two = ctx.constant(sqrt_two_, mx)
    h1 = sqrt_two * mx
    h2 = ctx.select(ctx.And(sqa == 1, r > 0), mx + mx * r / 2, mx * sqa)
    return ctx(ctx.select(mx == mn, h1, h2))


def absolute(ctx, z: float | complex):
    """Absolute value of float and complex inputs."""
    if z.is_complex:
        return ctx(hypot(ctx, z.real, z.imag))
    return ctx(abs(z))


def real_asin_acos_kernel(ctx, x: float):
    """A kernel for evaluating asin and acos functions on real inputs.

    This is a stripped-down version of asin_acos_kernel(complex(x, 0)).
    """
    ax = ctx.abs(x)
    one = ctx.constant(1, x)
    two = ctx.constant(2, x)
    largest = ctx.constant("largest", x)
    safe_max = ctx.sqrt(largest) / 8
    xm1 = ax - one
    sq = ctx.sqrt((one + ax) * abs(xm1))
    im = ctx.select(ax >= safe_max, ctx.log(two) + ctx.log(ax), ctx.log1p(xm1 + sq))
    return ctx.select(ax <= one, sq, im)


def asin_acos_kernel(ctx, z: complex):
    """A kernel for evaluating asin and acos functions on complex inputs.

    If
        w = asin_acos_kernel(z)
    then
        asin(z) = complex(atan2(z.real, w.real), sign(z.imag) * w.imag)
    and
        acos(z) = complex(atan2(w.real, z.real), -sign(z.imag) * w.imag)

    See asin and acos for the description of asin_acos_kernel algorithm.
    """
    signed_x = z.real
    signed_y = z.imag
    x = ctx.abs(signed_x)
    y = ctx.abs(signed_y)

    zero = ctx.constant(0, signed_x)
    half = ctx.constant(0.5, signed_x)
    one = ctx.constant(1, signed_x)
    one_and_half = ctx.constant(1.5, signed_x)
    two = ctx.constant(2, signed_x)
    log2 = ctx.log(two)
    smallest = ctx.constant("smallest", signed_x)
    largest = ctx.constant("largest", signed_x)

    safe_min = ctx.sqrt(smallest) * 4
    safe_max = ctx.sqrt(largest) / 8
    safe_max_m6 = safe_max * 1e-6
    safe_max_p12 = safe_max * 1e12
    safe_max_p2 = safe_max * 1e2
    safe_max_opt = ctx.select(x < safe_max_p12, safe_max_m6, safe_max_p2)

    xp1 = x + one
    xm1 = x - one
    r = ctx.hypot(xp1, y)
    s = ctx.hypot(xm1, y)
    a = half * (r + s)
    ap1 = a + one
    rpxp1 = r + xp1
    spxm1 = s + xm1
    smxm1 = s - xm1
    yy = y * y
    apx = a + x
    half_yy = half * yy
    half_apx = half * apx
    y1 = ctx.select(
        ctx.max(x, y) >= safe_max,
        y,  # C5
        ctx.select(
            x <= one, ctx.sqrt(half_apx * (yy / rpxp1 + smxm1)), y * ctx.sqrt(half_apx / rpxp1 + half_apx / spxm1)  # R2  # R3
        ),
    )

    x_ge_1_or_not = ctx.select(
        x >= one,
        half_yy / rpxp1 + half * spxm1,  # I3
        ctx.select(a <= one_and_half, half_yy / rpxp1 + half_yy / smxm1, a - one),  # I2  # I1
    ).reference()

    am1 = ctx.select(ctx.And(y < safe_min, x < one), -((xp1 * xm1) / ap1), x_ge_1_or_not)  # C123_LT1
    mx = ctx.select((y >= safe_max_opt).reference("y_gt_safe_max_opt"), y, x)
    sq = ctx.sqrt(am1 * ap1)
    xoy = ctx.select(ctx.And(y >= safe_max_opt, ctx.Not(y.is_posinf)), x / y, zero)
    imag = ctx.select(
        mx >= ctx.select(y >= safe_max_opt, safe_max_opt, safe_max),
        log2 + ctx.log(mx) + half * ctx.log1p(xoy * xoy),  # C5 & C123_INF
        ctx.select(ctx.And(y < safe_min, x < one), y / sq, ctx.log1p(am1 + sq)),  # C123_LT1  # I1 & I2 & I3
    )
    return ctx(ctx.complex(y1, imag))


def complex_asin(ctx, z: complex):
    # fmt: off
    """Arcus sine on complex input.

    arcsin(z) = 2 * arctan2(z, (1 + sqrt(1 - z * z)))

    Here we well use a modified version of the [Hull et
    al]((https://dl.acm.org/doi/10.1145/275323.275324) algorithm with
    a reduced number of approximation regions.

    Hull et al define complex arcus sine as

      arcsin(x + I*y) = arcsin(x/a) + sign(y; x) * I * log(a + sqrt(a*a-1))

    where

      x and y are real and imaginary parts of the input to arcsin, and
      I is imaginary unit,
      a = (hypot(x+1, y) + hypot(x-1, y))/2,
      sign(y; x) = 1 when y >= 0 and abs(x) <= 1, otherwise -1.

    x and y are assumed to be non-negative as the arcus sine on other
    quadrants of the complex plane are defined by

      arcsin(-z) == -arcsin(z)
      arcsin(conj(z)) == conj(arcsin(z))

    where z = x + I*y.

    Hull et al split the first quadrant into 11 regions in total, each
    region using a different approximation of the arcus sine
    function. It turns out that when considering the evaluation of
    arcus sine real and imaginary parts separately, the 11 regions can
    be reduced to 3 regions for the real part, and to 4 regions for
    the imaginary part. This reduction of the appriximation regions
    constitutes the modification of the Hull et al algorithm that is
    implemented below and it is advantageous for functional
    implmentations as there will be less branches. The modified Hull
    et al algorithm is validated against the original Hull algorithm
    implemented in MPMath.

    Per Hull et al Sec. "Analyzing Errors", in the following we'll use
    symbol ~ (tilde) to denote "approximately equal" relation with the
    following meaning:

      A ~ B  iff  A = B * (1 + s * eps)

    where s * eps is a small multiple of eps that quantification
    depends on the particular case of error analysis.
    To put it simply, A ~ B means that the numerical values of A and B
    within the given floating point system are equal or very
    close. So, from the numerical evaluation point of view it does not
    matter which of the expressions, A or B, to use as the numerical
    results will be the same.

    We define:
      safe_min = sqrt(<smallest normal value>) * 4
      safe_max = sqrt(<largest finite value>) / 8

    Real part
    ---------
    In general, the real part of arcus sine input can be expressed as
    follows:

      arcsin(x / a) = arctan((x/a) / sqrt(1 - (x/a)**2))
                    = arctan(x / sqrt(a**2 - x**2))
                    = arctan2(x, sqrt(a**2 - x**2))              Eq. 1
                    = arctan2(x, sqrt((a + x) * (a - x)))        Eq. 2

    for which the following approximations will be used (in the
    missing derivation cases, see Hull et al paper for details):

    - Hull et al Case 5:
      For x > safe_max and any y, we have
        x + 1 ~ x - 1 ~ x
      so that
        a ~ hypot(x, y)
      For y > safe_max and x < safe_max, we have
        hypot(x + 1, y) ~ hypot(x - 1, y) ~ hypot(x, y) ~ a.
      Combining these together gives: if max(x, y) > safe_max then
        a**2 ~ hypot(x, y)**2 ~ x**2 + y**2
      and Eq. 1 becomes
        arcsin(x / a) ~ arctan2(x, y)

    - Hull et al Safe region: for max(x, y) < safe_max, we have (see
      `a - x` approximation in Hull et al Fig. 2):

      If x <= 1 then
        arcsin(x / a) ~ arctan2(x, sqrt(0.5 * (a + x) * (y * y / (hypot(x + 1, y) + x + 1) + hypot(x - 1, y) - x - 1)))
      else
        arcsin(x / a) ~ arctan2(x, y * sqrt(0.5 * (a + x) * (1 / (hypot(x + 1, y) + x + 1) + 1 / (hypot(x - 1, y) + x - 1))))
 
    Imaginary part
    --------------
    In general, the unsigned imaginary part of arcus sine input can be
    expressed as follows:

      log(a + sqrt(a*a-1)) = log(a + sqrt((a + 1) * (a - 1)))
                           = log1p(a - 1 + sqrt((a + 1) * (a - 1)))   # Eq.3

    for which the following approximations will be used (for the
    derivation, see Hull et al paper):

    - modified Hull et al Case 5: for y > safe_max_opt we have
        log(a + sqrt(a*a-1)) ~ log(2) + log(y) + 0.5 * log1p((x / y) * (x / y))
      where using
        safe_max_opt = safe_max * 1e-6 if x < safe_max * 1e12 else safe_max * 1e2
      will expand the approximation region to capture also the Hull et
      Case 4 (x is large but less that eps * y) that does not have
      log1p term but under the Case 4 conditions, log(y) +
      0.5*log1p(...) ~ log(y).

    - Hull et al Case 1 & 2: for 0 <= y < safe_min and x < 1, we have
        log(a + sqrt(a*a-1)) ~ y / sqrt((a - 1) * (a + 1))
      where
        a - 1 ~ -(x + 1) * (x - 1) / (a + 1)

    - Hull et al Safe region. See the approximation of `a -
      1` in Hull et al Fig. 2 for Eq. 3:
        log(a + sqrt(a*a-1)) ~ log1p(a - 1 + sqrt((a + 1) * (a - 1)))
      where
        a - 1 ~ 0.5 * y * y / (hypot(x + 1, y) + x + 1) + 0.5 * (hypot(x - 1, y) + x - 1)        if x >= 1
        a - 1 ~ 0.5 * y * y * (1 / (hypot(x + 1, y) + x + 1) + 1 / (hypot(x - 1, y) - x - 1))    if x < 1 and a < 1.5
        a - 1 ~ a - 1                                                                            otherwise

    Different from Hull et al, we don't handle Cases 3 and 6 because
    these only minimize the number of operations which may be
    advantageous for procedural implementations but for functional
    implementations these would just increase the number of branches
    with no gain in accuracy.

    The above algorithm is implemented in asin_acos_kernel function so
    that we'll have

      asin(z) = complex(atan2(z.real, w.real), sign(z.imag) * w.imag)

    where

      w = asin_acos_kernel(z).
    """
    # fmt: on
    signed_x = z.real.reference("signed_x", force=True)
    signed_y = z.imag.reference("signed_y", force=True)
    w = ctx.asin_acos_kernel(z)
    w_imag = w.imag
    zero = ctx.constant(0, w_imag)
    real = ctx.atan2(signed_x, w.real).reference("real")
    imag = ctx.select(signed_y < zero, -w_imag, w_imag)
    return ctx.complex(real, imag)


def real_asin(ctx, x: float):
    """Arcus sine on real input:

    arcsin(x) = 2 * arctan2(x, 1 + sqrt(1 - x * x))

    To avoid cancellation errors at abs(x) close to 1, we'll use

      1 - x * x == (1 - x) * (1 + x)
    """
    # Alternative formulas:
    #
    #   (i)  arcsin(x) = pi/2 - arccos(x)
    #   (ii) arcsin(x) = arctan2(x, sqrt(1 - x * x))
    #   (iii) arcsin(x) = arccos(1 - 2 * x * x) * sign(x) / 2
    #
    # (i) has cancellation errors for small abs(x)
    # (ii) is slightly less accurate than 2 * arctan2(x, (1 + sqrt(1 - x * x)))
    # (iii) is inaccurate for small abs(x)
    one = ctx.constant(1, x)
    sq = ctx.sqrt((one - x) * (one + x))
    ta = ctx.atan2(x, one + sq)
    return ctx(ta + ta)


def asin(ctx, z: complex | float):
    """Arcus sine on complex and real inputs.

    See complex_asin and real_asin for more information.
    """
    if not z.is_complex:
        return real_asin(ctx, z)
    return complex_asin(ctx, z)


def complex_asinh(ctx, z: complex):
    """Inverse hyperbolic sine on complex input:

      asinh(z) = -I * asin(I * z)

    where

      asin(z') = complex(atan2(z'.real, w.real), sign(z'.imag) * w.imag)
      w = asin_acos_kernel(z')
      z' = I * z

    Let's find

      asinh(z) = -I * asin(z')
               = -I * complex(atan2(z'.real, w.real), sign(z'.imag) * w.imag)
               = complex(sign(z'.imag) * w.imag, -atan2(z'.real, w.real))
               [z'.imag = z.real, z'.real = -z.imag]
               = complex(sign(z.real) * w.imag, atan2(z.imag, w.real))
    where

      w = asin_acos_kernel(complex(-z.imag, z.real))
    """
    signed_x = z.real.reference("signed_x", force=True)
    signed_y = z.imag.reference("signed_y", force=True)
    w = ctx.asin_acos_kernel(ctx.complex(-signed_y, signed_x))
    real = ctx.select(signed_x < 0, -w.imag, w.imag)
    imag = ctx.atan2(signed_y, w.real)
    return ctx(ctx.complex(real, imag))


def real_asinh(ctx, x: float):
    """Inverse hyperbolic sine on real input:

      asinh(x) = log(x + hypot(1, x))

    This algorithm is based on the StableHLO v1.1.4 function CHLO_AsinhOp.

    To avoid underflow in 1 + x * x, we'll define z = hypot(1, x) and
    write

      log(x + z)
      = log(((x + z) * (1 + z)) / (1 + z))
      [ z ** 2 = 1 + x ** 2]
      = log((x + x * z + z + 1 + x**2) / (1 + z))
      = log(1 + x + x ** 2 / (1 + z))
      = log1p(x + x ** 2 / (1 + hypot(1, x)))

    that is, for non-negative x, we have

      asinh(x) = log1p(x + x ** 2 / (1 + hypot(1, x)))

    It turns out, this is accurate for all abs(x) < sqrt(max).

    To avoid overflow in x ** 2, we'll use

      asinh(x) = log(2) + log(x)

    when abs(x) > sqrt(max),

    For x < 0, we'll use

      asinh(x) = -asinh(-x)

    """
    one = ctx.constant(1, x)
    two = ctx.constant(2, x)
    ax = abs(x)
    ax2 = ax * ax
    z = ctx.sqrt(one + ax2)
    a0 = ctx.log(two) + ctx.log(ax)
    a1 = ctx.log1p(ax + ax2 / (one + z))
    a2 = ctx.log(ax + z)

    safe_max_limit_coefficient = ctx.parameters.get("safe_max_limit_coefficient", None)
    if safe_max_limit_coefficient is None:
        safe_max_limit = ctx.sqrt(ctx.constant("largest", x))
    else:
        safe_max_limit = ctx.sqrt(ctx.constant("largest", x)) * safe_max_limit_coefficient

    # | Function                        | dtype   | dULP=0 | dULP=1 | dULP=2 | dULP=3 | dULP>3 | errors  |
    # | ------------------------------- | ------- | ------ | ------ | ------ | ------ | ------ | ------- |
    # | asinh                           | float32 | 895795 | 104084 | 122    | -      | -      | -       |
    # | real_asinh[safe_min_limit=1]    | float32 | 922339 | 77566  | 96     | -      | -      | -       |
    # | real_asinh[safe_min_limit=10]   | float32 | 922885 | 77050  | 66     | -      | -      | -       |
    # | real_asinh[safe_min_limit=100]  | float32 | 922811 | 77124  | 66     | -      | -      | -       |
    # | real_asinh[safe_min_limit=1000] | float32 | 922813 | 77122  | 66     | -      | -      | -       |
    # | real_asinh[safe_min_limit=None] | float32 | 922791 | 77144  | 66     | -      | -      | -       |
    # | real_asinh[safe_min_limit=0.1]  | float64 | 829011 | 170324 | 292    | 156    | 218    | -       |
    # | real_asinh[safe_min_limit=1]    | float64 | 829565 | 170340 | 96     | -      | -      | -       |
    # | real_asinh[safe_min_limit=10]   | float64 | 829593 | 170314 | 94     | -      | -      | -       |
    # | real_asinh[safe_min_limit=100]  | float64 | 829605 | 170302 | 94     | -      | -      | -       |
    # | real_asinh[safe_min_limit=1000] | float64 | 829599 | 170308 | 94     | -      | -      | -       |
    # | real_asinh[safe_min_limit=None] | float64 | 829637 | 170270 | 94     | -      | -      | -       |

    safe_min_limit = ctx.parameters.get("safe_min_limit", None)

    if safe_min_limit is None:
        r = ctx.select(ax >= safe_max_limit, a0, a1)
    else:
        r = ctx.select(
            ax >= safe_max_limit,
            a0,
            ctx.select(ax <= safe_min_limit, a1, a2),
        )

    return ctx(ctx.sign(x) * r)


def real_asinh_2(ctx, x: float):
    """Inverse hyperbolic sine on real input.

    This algorithm is based on the modified version of the [Hull et
    al]((https://dl.acm.org/doi/10.1145/275323.275324) algorithm used
    for asin and the relation
     asinh(x) = asin(complex(0, x)).imag
    """
    y = abs(x)
    one = ctx.constant(1, y)
    two = ctx.constant(2, y)
    s = ctx.hypot(one, y)
    am1 = ctx.select(s <= 1.5, y * y / (s + one), s - one)
    sq = ctx.sqrt(am1 * (s + one))

    a0 = ctx.log(two) + ctx.log(y)
    a1 = ctx.log1p(am1 + sq)

    safe_max = ctx.sqrt(ctx.constant("largest", y)) / 8
    large_y_cond = y > safe_max * 1e-6

    safe_min = ctx.sqrt(ctx.constant("smallest", y)) * 4
    small_y_cond = y < safe_min

    imag = ctx.select(large_y_cond, a0, ctx.select(small_y_cond, y, a1))

    return ctx(ctx.select(x < 0, -imag, imag))


def asinh(ctx, z: complex | float):
    """Inverse hyperbolic sine on complex and real inputs.

    See complex_asinh and real_asinh for more information.
    """
    if not z.is_complex:
        return real_asinh(ctx, z)
    return complex_asinh(ctx, z)


def real_acos(ctx, x: float):
    """Arcus cosine on real input:

    arccos(x) = 2 * arctan2(sqrt(1 - x * x), 1 + x)
              [to avoid undefined value at x == -1]
              = arctan2(sqrt(1 - x * x), x)

    To avoid cancellation errors at abs(x) close to 1, we'll use

      1 - x * x == (1 - x) * (1 + x)
    """
    one = ctx.constant(1, x)
    sq = ctx.sqrt((one - x) * (one + x))
    return ctx.atan2(sq, x)


def complex_acos(ctx, z: complex):
    """Arcus cosine on complex input

    Here we well use a modified version of the [Hull et
    al]((https://dl.acm.org/doi/10.1145/275323.275324) algorithm with
    a reduced number of approximation regions.

    Hull et al define complex arcus cosine as

      arccos(x + I*y) = arccos(x/a) - sign(y; x) * I * log(a + sqrt(a*a-1))

    where

      x and y are real and imaginary parts of the input to arccos, and
      I is imaginary unit,
      a = (hypot(x+1, y) + hypot(x-1, y))/2,
      sign(y; x) = 1 when y >= 0 and abs(x) <= 1, otherwise -1.

    The algorithm for arccos is identical to arcsin except that its
    real part uses real arccos and the imaginary part has opposite
    sign. Therefore, refer to arcsin documentation regarding the
    details of the algorithm and notice that from

      real(arcsin(z)) = arctan2(p, q)

    follows that

      real(arccos(z)) = argtan2(q, p),

    and we have the following identity

       imag(arccos(z)) = -imag(arcsin(z)).

    With the above notes, we'll have

      acos(z) = complex(atan2(w.real, z.real), -sign(z.imag) * w.imag)

    where

      w = asin_acos_kernel(z)
    """
    signed_x = z.real.reference("signed_x", force=True)
    signed_y = z.imag.reference("signed_y", force=True)
    w = ctx.asin_acos_kernel(z)
    real = ctx.atan2(w.real, signed_x)
    imag = ctx.select(signed_y < 0, w.imag, -w.imag)
    return ctx.complex(real, imag)


def acos(ctx, z: complex | float):
    """Arcus cosine on complex and real inputs.

    See complex_acos and real_acos for more information.
    """
    if z.is_complex:
        return complex_acos(ctx, z)
    return real_acos(ctx, z)


def complex_acosh(ctx, z: complex):
    """Inverse hyperbolic cosine on complex input:

    acosh(z) = sqrt(z - 1) / sqrt(1 - z) * acos(z)
             = I * acos(z)               # when z.imag >= 0
             = -I * acos(z)              # otherwise

    where

      w = asin_acos_kernel(z)
      acos(z) = complex(atan2(w.real, z.real), -sign(z.imag) * w.imag)

    For `z.imag >= 0`, we'll have `sign(z.imag) = 1` and

      acosh(z) = I * complex(atan2(w.real, z.real), -sign(z.imag) * w.imag)
               = complex(w.imag, atan2(w.real, z.real))

    For `z.imag < 0`, we'll have `sign(z.imag) = -1` and

      acosh(z) = -I * complex(atan2(w.real, z.real), -sign(z.imag) * w.imag)
               = -I * complex(atan2(w.real, z.real),  w.imag)
               = complex(w.imag, -atan2(w.real, z.real))

    So, for any `z.imag`, we'll have

      acosh(z) = complex(w.imag, sign(z.imag) * atan2(w.real, z.real))

    """
    signed_x = z.real.reference("signed_x", force=True)
    signed_y = z.imag.reference("signed_y", force=True)
    w = ctx.asin_acos_kernel(z)
    imag = ctx.atan2(w.real, signed_x)
    return ctx(ctx.complex(w.imag, ctx.select(signed_y < 0, -imag, imag)))


def real_acosh(ctx, x: float):
    """Inverse hyperbolic cosine on real input:

    acosh(x) = log(x + sqrt(x * x - 1))
             = log(x + sqrt(x+1)*sqrt(x-1)))
             = log(1 + x-1 + sqrt(x+1)*sqrt(x-1)))
             = log1p(sqrt(x-1) * (sqrt(x+1) + sqrt(x-1)))

    The last expression avoids errors from cancellations when x is
    close to one. This also ensures the nan result when x < 1 because
    sqrt(x') returns nan when x' < 0.

    To avoid overflow in multiplication for large x (x > max / 2),
    we'll use

      acosh(x) = log(2) + log(x)

    """
    one = ctx.constant(1, x)
    two = ctx.constant(2, x)
    sqxm1 = ctx.sqrt(x - one)
    sqxp1 = ctx.sqrt(x + one)
    a0 = ctx.log(two) + ctx.log(x)
    a1 = ctx.log1p(sqxm1 * (sqxp1 + sqxm1))

    safe_max_limit_coefficient = ctx.parameters.get("safe_max_limit_coefficient", None)
    if safe_max_limit_coefficient is None:
        safe_max_limit = ctx.constant("largest", x) / 2
    else:
        safe_max_limit = ctx.constant("largest", x) * safe_max_limit_coefficient
    return ctx.select(x >= safe_max_limit, a0, a1)


def acosh(ctx, z: complex | float):
    """Inverse hyperbolic cosine on complex and real inputs.

    See complex_acosh and real_acosh for more information.
    """
    if z.is_complex:
        return complex_acosh(ctx, z)
    return real_acosh(ctx, z)


def sqrt(ctx, z: complex | float):
    return ctx.sqrt(z)


def angle(ctx, z: complex):
    return ctx.atan2(z.imag, z.real)


def kahan3(ctx, x1: float, x2: float, x3: float):
    """Kahan sum of three floating-point numbers"""
    s = x1 + x2
    c = (s - x1) - x2
    y3 = x3 - c
    return s + y3


def kahan4(ctx, x1: float, x2: float, x3: float, x4: float):
    """Kahan sum of four floating-point numbers"""
    t2 = x1 + x2
    c2 = (t2 - x1) - x2
    y3 = x3 - c2
    t3 = t2 + y3
    c3 = (t3 - t2) - y3
    y4 = x4 - c3
    return t3 + y4


def fma(ctx, x, y, z):
    """Evaluate x * y + z"""
    return x * y + z


def complex_log1p(ctx, z: complex):
    """Logarithm of 1 + z on complex input:

      log1p(x + I * y) = 0.5 * log((x+1)**2 + y**2) + I * arctan2(y, x + 1)

    where

      x and y are real and imaginary parts of the input to log1p, and
      I is imaginary unit.

    Let's define

      mx = max(abs(x + 1), abs(y))
      mn = min(abs(x + 1), abs(y))

    then the real part of the complex log1p value reads

      real(log(x + I * y)) = log(hypot(x + 1, y))
        = log(mx * sqrt(1 + (mn / mx) ** 2))
        = log(mx) + 0.5 * log1p((mn / mx) ** 2)

    where

      log(mx) = log(max(abs(x + 1), abs(y)))
              = log1p(max(abs(x + 1) - 1, abs(y) - 1))
              = log1p(select(x + 1 >= abs(y), x, mx - 1))

    To handle mn == mx == inf case, we'll use

      log1p((mn / mx) ** 2) = log1p(select(mn == mx, one, (mn / mx) ** 2))

    Problematic regions
    -------------------

    Notice that when abs(y) < 1 and abs(x + 0.5 * y ** 2) is small,
    catastrophic cancellation errors occur in evaluating the real part
    of complex log1p:

      log(hypot(x + 1, y)) = 0.5 * log1p(2 * x + y * y + x * x)

    where the magnitude of the correct value `x * x` is smaller than
    the rounding errors occurring from addition `2 * x + y * y`,
    especially when using FP32. A similar phenomenon is expected when
    x is close to -2 and abs(y) is small so that rounding errors from
    `2 * x + x ** 2` dominate over `y ** 2`.
    """
    x = z.real
    y = z.imag
    one = ctx.constant(1, x)
    half = ctx.constant(0.5, x)
    xp1 = x + one
    axp1 = abs(xp1)
    ay = abs(y)
    mx = ctx.maximum(axp1, ay)
    mn = ctx.minimum(axp1, ay)
    r = mn / mx
    re = ctx.log1p(ctx.select(xp1 >= ay, x, mx - one)) + half * ctx.log1p(ctx.select(ctx.eq(mn, mx), one, r * r))
    im = ctx.atan2(y, xp1)
    return ctx(ctx.complex(re, im))


def log1p(ctx, z: complex | float):
    """log(1 + z)

    See complex_log1p for more information.
    """
    if z.is_complex:
        return complex_log1p(ctx, z)
    return ctx.log1p(z)
