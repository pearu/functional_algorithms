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


import functools


class definition:
    """Decorator for definitions of algorithms.

    Usage
    -----

    To define an algorithm foo, provide the following three functions
    decorated with `definition(...)` as follows:

      @definition("foo", domain='real')
      def real_foo(ctx, ...):
          '''
          <explain the real foo algorithm here>
          '''
          return ctx(<expression for real foo(...)>)

      @definition("foo", domain='complex')
      def complex_foo(ctx, ...):
          '''
          <explain the complex foo algorithm here>
          '''
          return ctx(<expression for complex foo(...)>)

      @definition("foo")
      def foo(ctx, ...):
          # nothing to implement here as `definition(...,
          # domain=None)` implements dispatch to real_foo and
          # complex_foo based on the arguments domain.
          assert 0  # unreachable
    """

    # dict(<domain>=<dict of <native function name>:<definition for domain>>)
    _registry = {}

    def __init__(self, native_func_name, domain=None):
        assert domain in {"real", "complex", None}, domain
        self.domain = domain
        self.native_func_name = native_func_name

        if domain is not None:
            if domain not in self._registry:
                self._registry[domain] = {}
            self.registry = self._registry[domain]

    def __call__(self, func):

        if self.domain is None:

            @functools.wraps(func)
            def wrapper(ctx, *args, **kwargs):
                domain = "complex" if args[0].is_complex else "real"
                defn = self._registry[domain].get(self.native_func_name)
                if defn is None:
                    raise NotImplementedError(f"definition for {domain} {self.native_func_name} is not provided in algorithms")

                return defn(ctx, *args, **kwargs)

            return wrapper

        @functools.wraps(func)
        def wrapper(ctx, *args, **kwargs):

            result = func(ctx, *args, **kwargs)
            if result is NotImplemented:
                raise NotImplementedError(f"{self.native_func_name} not implemented for {self.domain} domain: {func.__name__}")

            return result

        self.registry[self.native_func_name] = wrapper

        return wrapper


@definition("conj", domain="complex")
def complex_conj(ctx, z: complex):
    """Conjugate of complex inputs."""
    return ctx.complex(z.real, -z.imag)


@definition("conj", domain="real")
def real_conj(ctx, z: float):
    """Conjugate of real inputs."""
    return NotImplemented


@definition("conj")
def conj(ctx, z: float | complex):
    """Conjugate of real and complex inputs."""
    assert 0  # unreachable


@definition("square", domain="real")
def real_square(ctx, x: float):
    """Square on real input: x * x"""
    return ctx(x * x)


@definition("square", domain="complex")
def complex_square(ctx, z: complex):
    """Square on complex input:

    If abs(z.real) == abs(z.imag) then
        square(z).real = 0
    else
        square(z).real = (z.real - z.imag) * (z.real + z.imag)
    square(z).imag = 2 * (z.real * z.imag)
    """
    x = z.real
    y = z.imag
    # We'll use
    #   x * x - y * y = (x - y) * (x + y)

    # We treat `abs(x) == abs(y)` case separately to avoid nan real
    # part while it should be 0, for example, when taking x = largest
    # and y = largest (x-y is 0 but x+y is inf). However, for infinite
    # x and y, the correct real part is nan.
    #
    # Notice that 2 * (x * y) is not the same as 2 * x * y when `2
    # * x` overflows (e.g. take x = finfo(..).max) while `x * y`
    # doesn't (e.g. take y such that `abs(y) < 1`).
    z_sq = ctx.complex(ctx.select(ctx.And(ctx.is_finite(x), abs(x) == abs(y)), 0, (x - y) * (x + y)), 2 * (x * y))
    return ctx(z_sq)


@definition("square")
def square(ctx, z: complex | float):
    """Square on real and complex inputs.

    See complex_square and real_square for more information.
    """
    assert 0  # unreachable


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


@definition("absolute", domain="complex")
def complex_absolute(ctx, z: complex):
    """Absolute value of complex inputs."""
    return ctx(hypot(ctx, z.real, z.imag))


@definition("absolute", domain="real")
def real_absolute(ctx, z: float | complex):
    """Absolute value of real inputs."""
    return ctx(abs(z))


@definition("absolute")
def absolute(ctx, z: float | complex):
    """Absolute value of real and complex inputs."""
    assert 0  # unreachable


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

      asin_acos_kernel(z) = sqrt(a ** 2 - x ** 2) + I * log(a + sqrt(a ** 2 - 1))

    where

      x and y are real and imaginary parts of the input to asin_acos_kernel,
      I is imaginary unit, and
      a = (hypot(x + 1, y) + hypot(x - 1, y)) / 2

    See asin for the description of the asin_acos_kernel algorithm.

    We have
        asin(z) = complex(atan2(z.real, w.real), sign(z.imag) * w.imag)
        acos(z) = complex(atan2(w.real, z.real), -sign(z.imag) * w.imag)
        asinh(z) = complex(sign(z.real) * w'.imag, atan2(z.imag, w'.real))
        acosh(z) = complex(w.imag, sign(z.imag) * atan2(w.real, z.real))
    where
        w = asin_acos_kernel(z)
        w' = asin_acos_kernel(I * z)

    See asin, asinh, acos, acosh for the derivation of the above relations.
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


@definition("asin", domain="complex")
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
    zero = ctx.constant(0, signed_x)
    real = ctx.atan2(signed_x, w.real).reference("real")
    imag = ctx.select(signed_y < zero, -w_imag, w_imag)
    return ctx.complex(real, imag)


@definition("asin", domain="real")
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


@definition("asin")
def asin(ctx, z: complex | float):
    """Arcus sine on complex and real inputs.

    See complex_asin and real_asin for more information.
    """
    assert 0  # unreachable


@definition("asinh", domain="complex")
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


@definition("asinh", domain="real")
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


@definition("asinh")
def asinh(ctx, z: complex | float):
    """Inverse hyperbolic sine on complex and real inputs.

    See complex_asinh and real_asinh for more information.
    """
    assert 0  # unreachable


@definition("acos", domain="real")
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


@definition("acos", domain="complex")
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


@definition("acos")
def acos(ctx, z: complex | float):
    """Arcus cosine on complex and real inputs.

    See complex_acos and real_acos for more information.
    """
    assert 0  # unreachable


@definition("acosh", domain="complex")
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


@definition("acosh", domain="real")
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


@definition("acosh")
def acosh(ctx, z: complex | float):
    """Inverse hyperbolic cosine on complex and real inputs.

    See complex_acosh and real_acosh for more information.
    """
    assert 0  # unreachable


@definition("angle", domain="complex")
def angle(ctx, z: complex):
    return ctx.atan2(z.imag, z.real)


@definition("angle")
def angle_(ctx, z: complex):
    assert 0  # unreachable


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


def get_veltkamp_splitter_constant(ctx, largest: float):
    """Return 2 ** s + 1 where s = ceil(p / 2) and s is the precision of
    the floating point number system.

    Using `largest` to detect the floating point type: float16,
    float32, or float64.
    """
    fp64 = ctx.constant(2 ** (54 // 2) + 1, largest)
    fp32 = ctx.constant(2 ** (24 // 2) + 1, largest)
    fp16 = ctx.constant(2 ** (12 // 2) + 1, largest)
    return ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16)).reference(
        "veltkamp_splitter_constant", force=True
    )


def split_veltkamp(ctx, C, x):
    """Veltkamp splitter: x = xh + xl"""
    g = C * x
    d = x - g
    xh = g + d
    xl = x - xh
    return xh, xl


def square_dekker(ctx, x, xh, xl):
    """Square using Dekker's product:

    x ** 2 = xxh + xxl
    """
    xxh = x * x
    t1 = (-xxh) + xh * xh
    t2 = t1 + xh * xl
    t3 = t2 + xh * xl
    xxl = t3 + xl * xl
    return xxh.reference("square_dekker_high"), xxl.reference("square_dekker_low")


def add_2sum(x, y, fast=False):
    """Sum of x and y.

    Return s, t such that

      x + y = s + t

    When fast is True, abs(x) >= abs(y) is assumed.
    """
    s = x + y
    z = s - x
    if fast:
        t = y - z
    else:
        t = (x - (s - z)) + (y - z)
    prefix = "add_fast2sum" if fast else "add_2sum"
    return s.reference(prefix + "_high"), t.reference(prefix + "_low")


def sum_2sum(seq, fast=False):
    """Sum all items in a sequence using 2Sum algorithm."""
    if len(seq) == 1:
        s, t = seq[0], type(seq[0])(0)
    elif len(seq) == 2:
        s, t = add_2sum(seq[0], seq[1], fast=fast)
    elif len(seq) >= 3:
        s, t = add_2sum(seq[0], seq[1], fast=fast)
        for n in seq[2:]:
            s, t1 = add_2sum(s, n, fast=fast)
            t = t + t1
        s, t = add_2sum(s, t, fast=fast)
    else:
        assert 0  # unreachable
    prefix = "sum_fast2sum" if fast else "sum_2sum"
    return s.reference(prefix + "_high"), t.reference(prefix + "_low")


@definition("log1p", domain="complex")
def complex_log1p(ctx, z: complex):
    """Logarithm of 1 + z on complex input:

      log1p(x + I * y) = 0.5 * log((x + 1) ** 2 + y ** 2) + I * arctan2(y, x + 1)

    where

      x and y are real and imaginary parts of the input to log1p, and
      I is imaginary unit.

    For evaluating the real part of log1p accurately on the whole
    complex plane, the following cases must be handled separately:

    A) Avoid catastrophic cancellation errors when x is close `-0.5 * y * y`
       and `abs(y) < 1`.
    B) Avoid overflow from square when x or y are large in absolute value.
    C) Avoid cancellation errors when x is close to -1 and y is not large.
    D) Avoid cancellation errors when x is close to -2 and y is not large.

    Case A
    ------

    The real part of log1p reads:

      0.5 * log((x + 1) ** 2 + y ** 2) = 0.5 * log1p(x + x + x * x + y * y)

    When abs(y) < 1 and abs(x + 0.5 * y ** 2) is small, catastrophic
    cancellation errors occur when evaluating `x + x + x * x + y * y`
    using floating-point arithmetics. To avoid these errors, we'll use
    Dekker's product for computing `x * x` and `y * y` which
    effectively doubles the precision of the used floating-point
    system. In addition, the terms are summed together using 2Sum
    algorithm that minimizes cancellation errors. We'll have

      xxh, xxl = square_dekker(x)
      yyh, yyl = square_dekker(y)
      x + x + x * x + y * y = sum_2sum([x + x, yyh, xxh, yyl, xxl])

    which is accurate when the following inequalities hold:

      abs(x) < sqrt(largest) * 0.1
      abs(y) < sqrt(largest) * 0.99

    [verified numerically for float32 and float64], except when x is
    close to -1 (see Case C).

    Case B
    ------

    If abs(x) or abs(y) is larger than sqrt(largest), squareing
    these will overflow. To avoid such overflows, we'll apply
    rescaling of log1p arguments.

    First notice that if `abs(x) > sqrt(largest) > 4 / eps` holds then
    `x + 1 ~= x`. Also, if `abs(x) < 4 / eps` then `(x + 1) ** 2 + y
    ** 2 ~= y ** 2`. Proof:

      (x + 1) ** 2 + y ** 2 ~= y ** 2    iff y ** 2 > 4 * (x + 1) ** 2 / eps

      The lower limit to `y ** 2` is largest.  The upper limit to
      `4 * (x + 1) ** 2 / eps` is `64 / eps ** 3` which is smaller than
      largest. QED.

    In conclusion, we can write

      (x + 1) ** 2 + y ** 2 ~= x ** 2 + y ** 2

    whenever abs(x) or abs(y) is greater than sqrt(largest).

    Define

      mx = max(abs(x), abs(y))
      mn = min(abs(x), abs(y))

    then under the given restrictions we'll have

      real(log(x + I * y)) ~= 0.5 * log(x ** 2 + y ** 2)
        = 0.5 * log(mx ** 2 * (1 + (mn / mx) ** 2))
        = log(mx) + 0.5 * log1p((mn / mx) ** 2)

    If mn == inf and mx == inf, we'll define `mn / mx == 1` for the
    sake of reusing the above expression for complex infinities
    (recall, `real(log(+-inf +-inf * I)) == inf`).

    Case C
    ------

    If x is close to -1, then we'll use

      real(log1p(x + I * y)) = 0.5 * log((1 + x) ** 2 + y ** 2)

    which is accurate when the following inequalities hold:

      -1.5 < x < -0.5  or  abs(x + 1) < 0.5
      abs(y) < sqrt(largest)

    [verified numerically for float32 and float64]. For simplicity,
    we'll use the case C only when `abs(x) + abs(y) < 0.2`.

    Case D
    ------

    If x is close to -2, the cancellation errors are avoided by using
    the Case A method [verified numerically for float32 and float64].

    """
    # TODO: improve the accuracy of arctan2(y, x + 1) when x is small
    # (x + 1 will be inaccurate) or when x and y are large. The ULP
    # difference is 3 due to these errors for float32 inputs
    # 1.7378703e-07+0.119728915j and 3.5712988e+36-1.7536198e+36j, for
    # instance.
    fast = ctx.parameters.get("use_fast2sum", False)

    x = z.real
    y = z.imag
    one = ctx.constant(1, x)
    half = ctx.constant(0.5, x)
    xp1 = x + one
    axp1 = abs(xp1)

    largest = ctx.constant("largest", x).reference("largest")
    safe_max = ctx.sqrt(largest) * 0.01

    # Case A and D
    C = get_veltkamp_splitter_constant(ctx, largest)
    x2h = x + x
    xh, xl = split_veltkamp(ctx, C, x)
    yh, yl = split_veltkamp(ctx, C, y)
    xxh, xxl = square_dekker(ctx, x, xh, xl)
    yyh, yyl = square_dekker(ctx, y, yh, yl)
    s, _ = sum_2sum([x2h, yyh, xxh, yyl, xxl], fast=fast)
    re_A = half * ctx.log1p(s)

    # Case B
    ax = abs(x)
    ay = abs(y)
    mx = ctx.maximum(ax, ay)
    mn = ctx.minimum(ax, ay)
    r = mn / mx
    re_B = ctx.log(mx) + half * ctx.log1p(ctx.select(ctx.eq(mn, mx), one, r * r))

    # Case C
    re_C = half * ctx.log(xp1 * xp1 + y * y)

    re = ctx.select(mx > safe_max, re_B, ctx.select(axp1 + ay < 0.2, re_C, re_A))
    im = ctx.atan2(y, xp1)
    return ctx(ctx.complex(re, im))


@definition("log1p")
def log1p(ctx, z: complex | float):
    """log(1 + z)

    See complex_log1p for more information.
    """
    assert 0  # unreachable


def atanh_imag_is_half_pi(ctx, x: float):
    """Return smallest positive x such that imag(atanh(I*x)) == pi/2

    Using `largest` to detect the floating point type: float16,
    float32, or float64.
    """
    largest = ctx.constant("largest", x)
    # see tools/tune_atanh.py for computing the following constants:
    fp64 = ctx.constant(5805358775541310.0, x)
    fp32 = ctx.constant(62919776.0, x)
    fp16 = ctx.constant(1028.0, x)
    return ctx(ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16)))


def when_add_one_is_identity(ctx, largest: float):
    """Return smallest positive x such that x + 1.0 ~= x

    `~=` denotes floating-point equality.

    Using `largest` to detect the floating point type: float16,
    float32, or float64.
    """
    import numpy

    def get_value(dtype):
        return float(numpy.nextafter(dtype(1 / numpy.finfo(dtype).epsneg), dtype(numpy.inf)))

    fp64 = ctx.constant(get_value(numpy.float64), largest)
    fp32 = ctx.constant(get_value(numpy.float32), largest)
    fp16 = ctx.constant(get_value(numpy.float16), largest)
    return ctx(ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16)))


def when_exp_is_zero(ctx, largest: float):
    """Return smallest positive x such that 1 - exp(-x) ~= 1.

    `~=` denotes floating-point equality.

    Using `largest` to detect the floating point type: float16,
    float32, or float64.

    Algorithm
    ---------

    Within the given floating-point system, we have

      1 - y ~= 1

    whenever y <= epsneg where epsneg = 1 - nextafter(1, 0).

    Hence, the smallest `x` when `1 - exp(-x)` is 1, is

      -log(epsneg)
    """
    import numpy

    def get_value(dtype):
        return float(numpy.nextafter(dtype(-numpy.log(numpy.finfo(dtype).epsneg)), dtype(numpy.inf)))

    fp64 = ctx.constant(get_value(numpy.float64), largest)
    fp32 = ctx.constant(get_value(numpy.float32), largest)
    fp16 = ctx.constant(get_value(numpy.float16), largest)
    return ctx(ctx.select(largest > 1e308, fp64, ctx.select(largest > 1e38, fp32, fp16)))


@definition("atanh", domain="real")
def real_atanh(ctx, z: float):
    """Inverse hyperbolic tangent on real inputs:"""
    return NotImplemented


@definition("atanh", domain="complex")
def complex_atanh(ctx, z: complex):
    """Inverse hyperbolic tangent on complex inputs:

    atanh(z) = (log1p(z) - log1p(-z)) / 2

    Algorithm derivation
    --------------------

    We have

      log1p(x + I * y) = log((x + 1)**2 + y ** 2) / 2 + I * atan2(y, x + 1)

    where

      x and y are real and imaginary parts of the input to log1p, and
      I is imaginary unit.

    The real part
    -------------

    We have

      real(atanh(x + I * y)) = (log((x + 1)**2 + y ** 2) - log((x - 1)**2 + y ** 2)) / 4
        = log(((x + 1)**2 + y ** 2) / ((x - 1)**2 + y ** 2)) / 4
        = log1p(((x + 1)**2 + y ** 2) / ((x - 1)**2 + y ** 2) - 1) / 4
        = log1p(4 * x / ((x - 1)**2 + y ** 2)) / 4
        = sign(x) * log1p(4 * abs(x) / ((abs(x) - 1)**2 + y ** 2)) / 4      Eq 2.3.

    However, when abs(x) is large so that (~= denotes equality between
    floating point numbers)

      abs(x) + 1 ~= abs(x)

    or equivalently,

      abs(x) > 1 / epsneg ** 2

    where

      epsneg = 1 - nextafter(1, 0)

    we'll find

      real(atanh(x + I * y))
        = sign(x) * log1p(4 * x / (x * y * (x / y + y / x))) / 4
        = log1p(4 / y / (x / y + y / x)) / 4                                Eq 5.

    If abs(y) is not large, so that

      x / y + y / x ~= x / y

    or equivalently,

      (x / y) ** 2 + 1 ~= (x / y) ** 2
      (x / y) ** 2 > 1 / epsneg ** 2
      abs(y) < abs(x) * epsneg

    then we'll use

      real(atanh(x + I * y)) = log1p(4 / x) / 4                             Eq 4.

    The imaginary part
    ------------------

      imag(atanh(x + I * y)) = (atan2(y, x + 1) - atan2(y, x - 1)) / 2
        = atan2(y * (1/(x+1) - 1/(x-1)), 1 + y ** 2 / (x ** 2 - 1))
        = atan2(-2 * y, x ** 2 + y ** 2 - 1)
        = atan2(-2 * y, (x + 1) * (x - 1) + y * y)

    For large values of abs(x) or abs(y), that is, when

      x ** 2 + y ** 2 > 1 / epsneg ** 2

    we have

        imag(atanh(x + I * y)) = sign(y) * pi / 2
    """
    x = z.real
    y = z.imag

    zero = ctx.constant(0, x)
    one = ctx.constant(1, x)
    four = ctx.constant(4, x)
    half = ctx.constant(0.5, x)
    quarter = ctx.constant(0.25, x)
    pi = ctx.constant("pi", x)

    largest = ctx.constant("largest", x).reference("largest")
    inv_negeps = when_add_one_is_identity(ctx, largest)
    safe_max = inv_negeps * inv_negeps

    ax = abs(x)
    ay = abs(y)
    naxm1 = one - ax
    y2 = y * y
    sx = ctx.select(x >= 0, one, -one)
    sy = ctx.select(y >= 0, one, -one)
    in_safe_region = ctx.And(ax < safe_max, ay < safe_max)

    # real part
    arg23 = ax / (naxm1 * naxm1 + y2)  # Eq 2.3
    arg4 = one / ax  # Eq 4
    arg5 = one / (ax / y + y / ax) / y  # Eq 5.
    arg5 = ctx.select(ctx.Or(x.is_inf, y.is_inf), zero, arg5)
    arg = ctx.select(in_safe_region, arg23, ctx.select(ay * inv_negeps < ax, arg4, arg5))
    real = sx * ctx.log1p(four * arg) * quarter

    # imaginary part
    imag = ctx.select(in_safe_region, ctx.atan2(y + y, naxm1 * (one + ax) - y2), sy * pi) * half

    return ctx(ctx.complex(real, imag))


@definition("atanh")
def atanh(ctx, z: complex):
    """Inverse hyperbolic tangent on real and complex inputs."""
    # the implementation is provided by definition decorator
    assert 0  # unreachable


@definition("atan", domain="real")
def real_atan(ctx, z: float):
    """Arcus tangent on real inputs"""
    return NotImplemented


@definition("atan", domain="complex")
def complex_atan(ctx, z: complex):
    """Arcus tangent on complex inputs:

    atan(z) = -I * atanh(I * z)
    """
    w = ctx.atanh(ctx.complex(-z.imag, z.real))
    return ctx(ctx.complex(w.imag, -w.real))


@definition("atan")
def atan(ctx, z: complex | float):
    """Arcus tangent on complex and real inputs.

    See complex_atan for more information.
    """
    assert 0  # unreachable


@definition("sin")
def sin(ctx, z: float):
    assert 0  # unreachable


@definition("cos")
def cos(ctx, z: float):
    assert 0  # unreachable


@definition("tan", domain="real")
def real_naive_tan(ctx, z: float):
    return ctx.sin(z) / ctx.cos(z)


@definition("tan", domain="real")
def real_tan(ctx, z: float):
    return NotImplemented


@definition("tan")
def tan(ctx, z: complex | float):
    """Tangent on complex and real inputs.

    See complex_atan for more information.
    """
    assert 0  # unreachable


@definition("tanh")
def tanh(ctx, z: complex | float):
    """Hyperbolic tangent on complex and real inputs.

    See complex_atan for more information.
    """
    assert 0  # unreachable


@definition("sqrt", domain="real")
def real_sqrt(ctx, z: float):
    """Square root on real inputs"""
    return ctx.sqrt(z)


def complex_sqrt_polar(ctx, z: complex):
    """Square root on complex inputs:

      sqrt(z) = sqrt(r) * (cos(t) + I * sin(t))

    where

      z = x + I * y
      r = hypot(x, y)
      t = arctan2(y, x) / 2

    See pearu/functional_algorithms#53 for why not to use this algorithm
    """
    x = z.real
    y = z.imag
    r = ctx.hypot(x, y)
    r_sq = ctx.sqrt(r)
    t = ctx.atan2(y, x) / 2
    re = r_sq * ctx.cos(t)
    im = r_sq * ctx.sin(t)
    return ctx(ctx.complex(re, im))


@definition("sqrt", domain="complex")
def complex_sqrt(ctx, z: complex):
    """Square root on complex inputs:

      sqrt(z) = sqrt((hypot(x, y) + x)/2) + I * sgn(y) * sqrt((hypot(x, y) - x) / 2)

    where z = x + I * y, sgn(y) = 1 if y >= 0, and sgn(y) = -1 otherwise.

    Algorithm
    ---------

    In the above formula, catastrophic cancellation errors occur in
    the imaginary part when x is positive, and in the real part when x
    is negative. To avoid these, let us define

      u = sqrt((hypot(x, y) + abs(x))/2)
      v = sgn(y) * sqrt((hypot(x, y) - abs(x))/2)

    and find

      u * v = sgn(y) * sqrt(hypot(x, y) ** 2 - x ** 2) / 2 = y / 2

    That is, if x > 0, then we have

      sqrt(z) = u + I * y / u / 2

    and if x < 0,

      sqrt(z) = abs(y) / u / 2 + I * sgn(y) * u

    If abs(x) and abs(y) are smaller that smallest normal, then as a
    result of underflow, u will be zero and v will be undefined. On
    the other hand, if abs(x) and abs(y) are close to largest floating
    point number, then `hypot(x, y) + abs(x)` will overflow, and u
    will be `inf`. To address the issues from underflow and overflow,
    we'll use the following formula:

    1. abs(x) == abs(y), or abs(x) == inf and abs(y) == inf, then

      u_eq = sqrt(abs(x)) * sqrt((1 + sqrt(2))/2)
      abs(y) / u = sqrt(abs(x)) / sqrt((1 + sqrt(2))/2)

    2. If abs(x) > abs(y) and u == 0 (the underflow case) or u == inf
      (the overflow case), denote r = abs(y) / abs(x), then

      u_gt = sqrt(abs(x)) * sqrt((1 + hypot(1, r)) / 2)
      abs(y) / u = sqrt(abs(y)) * sqrt(r) / sqrt((1 + hypot(1, r)) / 2)

    3. If abs(x) < abs(y) and u == 0 (the underflow case) or u == inf
      (the overflow case), denote r = abs(x) / abs(y), then

      u_lt = sqrt(abs(y)) * sqrt((r + sqrt(1, r)) / 2)
      abs(y) / u = sqrt(abs(y)) / sqrt((r + sqrt(1, r)) / 2)
    """
    if 0:
        # Reproducer of pearu/functional_algorithms#53
        return complex_sqrt_polar(ctx, z)
    x = z.real
    y = z.imag
    ax = abs(x)
    ay = abs(y)
    sq_ax = ctx.sqrt(ax)
    sq_ay = ctx.sqrt(ay)

    one = ctx.constant(1, x)
    two = ctx.constant(2, x)

    if ctx.alt is None:
        sq_2 = ctx.sqrt(two)
        sq_12 = ctx.sqrt(one + sq_2)
    else:
        sq_2_ = ctx.alt.sqrt(2)
        sq_12_ = ctx.alt.sqrt(1 + sq_2_)
        sq_2 = ctx.constant(sq_2_, x)
        sq_12 = ctx.constant(sq_12_, x)

    u_general = ctx.sqrt((ctx.hypot(ax, ay) / two + ax / two))
    ay_div_u_general = ay / (u_general * two)

    r = ctx.select(ax == ay, one, ctx.select(ax < ay, ax / ay, ay / ax))
    sq_r = ctx.select(ax == ay, one, ctx.select(ax < ay, sq_ax / sq_ay, sq_ay / sq_ax))

    h = ctx.hypot(one, r)

    u_eq = sq_ax * sq_12 / sq_2
    ay_div_eq = sq_ay / (sq_12 * sq_2)

    sq_1h = ctx.sqrt(one + h)
    u_gt = sq_ax * (sq_1h / sq_2)
    ay_div_u_gt = sq_ay * sq_r / (sq_1h * sq_2)

    sq_rh = ctx.sqrt(r + h)
    u_lt = sq_ay * (sq_rh / sq_2)
    ay_div_u_lt = sq_ay / (sq_rh * sq_2)

    u = ctx.select(
        ax == ay, u_eq, ctx.select(ctx.Or(u_general == 0, u_general.is_posinf), ctx.select(ax > ay, u_gt, u_lt), u_general)
    )
    ay_div_u = ctx.select(
        ax == ay,
        ay_div_eq,
        ctx.select(
            ctx.Or(u_general == 0, u_general.is_posinf), ctx.select(ax > ay, ay_div_u_gt, ay_div_u_lt), ay_div_u_general
        ),
    )

    re = ctx.select(x >= 0, u, ay_div_u)
    im = ctx.select(x < 0, ctx.select(y < 0, -u, u), ctx.select(y < 0, -ay_div_u, ay_div_u))
    return ctx(ctx.complex(re, im))


@definition("sqrt")
def sqrt(ctx, z: complex | float):
    """Square root on complex and real inputs.

    See complex_sqrt for more information.
    """
    assert 0  # unreachable
