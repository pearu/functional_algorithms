from . import algorithms as faa
from . import floating_point_algorithms as fpa
from . import apmath

FloatArrayOrScalar = fpa.FloatArrayOrScalar
expansion: type = list[float | complex, ...]
fexpansion: type = list[float, ...]
cexpansion: type = list[complex, ...]


class definition(faa.definition):
    _registry = {}


@definition("square", domain="real")
def real_square(ctx, x: fexpansion, functional: bool = True, size: int = None):
    """Square on real input: x * x"""
    return apmath.square(ctx, x, functional=functional, size=2)


@definition("square")
def square(ctx, z: expansion):
    """Square on floating-point expansion"""
    assert 0


@definition("fma", domain="real")
def fma_real(
    ctx,
    x: float,
    y: float,
    z: float,
    functional: bool = True,
    size: int = None,
    possibly_zero_z: bool = True,
    fix_overflow: bool = True,
    algorithm: str = "a7",
    assume_fma: bool = False,
):
    """Emulated fused add-multiply float expansion: x * y + z

    Using Algorithm 9 from https://hal.science/hal-04575249 with
    modifications to handle possible overflows and underflows.

    Applicability:
      x * y and x * y + z are both finite.

    Accuracy:

      The ULP distance between the result and correct value is <= 1.

      fma(x, y, z) is exact (ULP distance is zero) for most inputs.
      If not, either

      - overflow occured in fma arithmetics (the return value is nan).
        Use fix_overflow to workaround overflow in Dekker product or 2Sum;
      - underflow occured in fma arithmetics. Use possibly_zero_z to
        fix z == 0 cases;
      - or the target uses flush-to-zero (FTZ) mode. Disable FTZ in target.
    """
    ctx._assume_same_dtype(x, y, z)
    if algorithm == "a9":
        # Algorithm 9 from https://hal.science/hal-04575249
        #
        # If fix_overflow is True, avoid overflow in Dekker product.
        mh, ml = apmath.two_prod(ctx, x, y, fix_overflow=fix_overflow)
        sh, sl = apmath.two_sum(ctx, mh, z, fix_overflow=fix_overflow)
        vh, vl = apmath.two_sum(ctx, ml, sl)

        p2 = fpa.is_one_or_three_times_power_of_two(ctx, vh, invert=True)
        cond = ctx.logical_or(p2, vl == 0)

        # (vh > 0 and vl > 0) or (vh <= 0 and vl <= 0)
        # (vh > 0 and vl > 0) or (not (vh > 0) and not (vl > 0))
        # (vh > 0 and vl > 0) or not (vh > 0 or vl > 0)
        same_sign = ctx.logical_or(ctx.logical_and(vh > 0, vl > 0), ctx.logical_not(ctx.logical_or(vh > 0, vl > 0)))

        C = ctx.constant(1, x)
        if possibly_zero_z:
            C = ctx.select(z == 0, ctx.constant(0, x), C)
        C = ctx.select(cond, C, ctx.select(same_sign, ctx.constant(9 / 8, x), ctx.constant(7 / 8, x)))

        return C * vh + sh
    elif algorithm == "a7":
        # Algorithm 7 from https://hal.science/hal-04624238
        # It's faster than other algorithms.
        mh, ml = apmath.two_prod(ctx, x, y, fix_overflow=fix_overflow, assume_fma=assume_fma)
        sh, sl = apmath.two_sum(ctx, mh, z, fix_overflow=fix_overflow, assume_fma=assume_fma)
        v = ml + sl
        # zh, zl = apmath.quick_two_sum(ctx, sh, v, fix_overflow=fix_overflow)
        zh, zl = apmath.two_sum(ctx, sh, v, fix_overflow=fix_overflow, assume_fma=assume_fma)
        return zh + zl
    elif algorithm == "a8":
        # Algorithm 8 from https://hal.science/hal-04624238
        mh, ml = apmath.two_prod(ctx, x, y, fix_overflow=fix_overflow)
        xh, xl = apmath.two_sum(ctx, mh, ml, fix_overflow=fix_overflow)
        sh, sl = apmath.two_sum(ctx, xh, z, fix_overflow=fix_overflow)
        vh, vl = apmath.two_sum(ctx, xl, sl, fix_overflow=fix_overflow)
        zh, zl = apmath.quick_two_sum(ctx, sh, vh, fix_overflow=fix_overflow)
        w = vl + zl
        st1 = zh + w
        wpof2 = fpa.is_power_of_two(ctx, w)
        wp = ctx.constant(3 / 2, x) * w
        st2 = zh + wp

        delta = w - zl
        t = vl - delta
        g = t * w
        r = ctx.select(wpof2, ctx.select(st2 == zh, zh, ctx.select(t == 0, st1, ctx.select(g < 0, zh, st2))), st1)
        return r
    elif algorithm == "apmath":
        # apmath native algorithm
        mh, ml = apmath.two_prod(ctx, x, y, fix_overflow=fix_overflow)
        hi, lo = apmath.renormalize(ctx, [z, mh, ml], functional=functional, size=2, fast=False, fix_overflow=fix_overflow)
        return hi + lo
    else:
        raise ValueError(f"unsupported algorithm value (got '{algorithm}'), expected 'a9|a7|a8|apmath'")


@definition("fma")
def fma(ctx, x: float | complex, y: float | complex, z: float | complex):
    """Fused add-multiply expansion"""
    assert 0
