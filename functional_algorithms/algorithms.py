"""Definitions of functional algorithms for math functions with
complex and float inputs. The aim is to provide algorithms that are
stable on the whole complex plane or real line.
"""
# This module provides only the definitions of algorithms. As such, it
# should not import any other Python modules or packages except the
# ones that provide definitions of algorithms to be used here.


def square(ctx, z: complex | float):
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
        z_sq = ctx.complex(ctx.select(abs(x) == abs(y),
                                      0,
                                      (x - y) * (x + y)),
                           2 * (x * y))
    else:
        z_sq = z * z
    ctx.update_refs()
    return z_sq


def hypot(ctx, x: float, y: float):
    assert not (x.is_complex or y.is_complex), (x, y)
    mx = ctx.maximum(abs(x), abs(y))
    mn = ctx.minimum(abs(x), abs(y))
    result = ctx.select(mx == mn,
                        mx * ctx.sqrt(ctx.constant(2, mx)),
                        mx * ctx.sqrt(ctx.square(mn / mx) + 1))
    ctx.update_refs()
    return result


def asin(ctx, z: complex | float):
    """Arcsin on complex and real inputs.

    The function relies on stable sqrt, atan2, and
    multiplication/division on float values.
    """
    if not z.is_complex:
        # TODO: implement standalone asin on float inputs
        return ctx.asin(z)

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
    smallest = ctx.constant('smallest', signed_x)
    largest = ctx.constant('largest', signed_x)

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
    y1 = ctx.select(ctx.max(x, y) >= safe_max,
                    y,  # C5
                    ctx.select(x <= one,
                               ctx.sqrt(half_apx * (yy/rpxp1 + smxm1)),  # R2
                               y * ctx.sqrt(half_apx/rpxp1 + half_apx/spxm1)  # R3
                               ))
    real = ctx.atan2(signed_x, y1).props_(force_ref=True)

    am1 = ctx.select(ctx.And(y < safe_min, x < one),
                     -((xp1 * xm1) / ap1),  # C123_LT1
                     ctx.select(x >= one,
                                half_yy / rpxp1 + half * spxm1,  # I3
                                ctx.select(a <= one_and_half,
                                           half_yy / rpxp1 + half_yy / smxm1,  # I2
                                           a - one  # I1
                                           )
                                ).props_(ref='x_ge_1_or_not', force_ref=True),
                     ref='am1')
    mx = ctx.select((y >= safe_max_opt).props_(ref='y_gt_safe_max_opt'), y, x)
    sq = ctx.sqrt(am1 * ap1)
    xoy = ctx.select(ctx.And(y >= safe_max_opt, ctx.Not(y.is_posinf)), x / y, zero)
    imag = ctx.select(mx >= ctx.select(y >= safe_max_opt, safe_max_opt, safe_max),
                      log2 + ctx.log(mx) + half * ctx.log1p(xoy * xoy),  # C5 & C123_INF
                      ctx.select(ctx.And(y < safe_min, x < one),
                                 y / sq,  # C123_LT1
                                 ctx.log1p(am1 + sq)  # I1 & I2 & I3
                                 ))

    signed_imag = ctx.select(signed_y < zero, -imag, imag)

    result = ctx.complex(real, signed_imag)
    ctx.update_refs()
    return result
