# This file is generated using functional_algorithms tool (0.15.1.dev10+g8c5ea60.d20250120), see
#   https://github.com/pearu/functional_algorithms
# for more information.

import numpy
from jax._src import lax
from jax._src import numpy as jnp
from jax._src.api import jit
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import promote_args_inexact

_np_dtypes = [numpy.float64, numpy.float32, numpy.float16]


@jit
def two_sum_unsafe(x: ArrayLike, y: ArrayLike) -> list[Array, Array]:
    """Add floating-point numbers x and y using 2sum algorithm.

        Returns (s, t) such that

          s = RN(x + y)
          x + y == s + t

        Applicability:
          x + y is finite and no overflow occurs within the 2sum algorithm [fix_overflow == False],
          x + y is finite [fix_overflow == True].

        Accuracy:
          The pair (s, t) represents the floating-point sum `x + y` exactly.

    FA tracing parameters: fix_overflow=False assume_fma=False"""

    (
        x,
        y,
    ) = promote_args_inexact("two_sum_unsafe", x, y)
    add_x_y = lax.add(x, y)
    subtract_4 = lax.sub(add_x_y, x)
    return [add_x_y, lax.add(lax.sub(x, lax.sub(add_x_y, subtract_4)), lax.sub(y, subtract_4))]


@jit
def two_sum_general(x: ArrayLike, y: ArrayLike) -> list[Array, Array]:
    """Add floating-point numbers x and y using 2sum algorithm.

        Returns (s, t) such that

          s = RN(x + y)
          x + y == s + t

        Applicability:
          x + y is finite and no overflow occurs within the 2sum algorithm [fix_overflow == False],
          x + y is finite [fix_overflow == True].

        Accuracy:
          The pair (s, t) represents the floating-point sum `x + y` exactly.

    FA tracing parameters: fix_overflow=True assume_fma=False"""

    (
        x,
        y,
    ) = promote_args_inexact("two_sum_general", x, y)
    add_x_y = lax.add(x, y)
    subtract_4 = lax.sub(add_x_y, x)
    largest_lst = [numpy.float64(1.7976931348623157e308), numpy.float32(3.4028235e38), numpy.float16(65500.0)]
    largest = largest_lst[_np_dtypes.index(x.dtype.type)]
    return [
        add_x_y,
        jnp.where(
            lax.le(lax.abs(subtract_4), largest),
            lax.add(lax.sub(x, lax.sub(add_x_y, subtract_4)), lax.sub(y, subtract_4)),
            numpy.array(0, dtype=x.dtype.type),
        ),
    ]


@jit
def two_prod_unsafe(x: ArrayLike, y: ArrayLike) -> list[Array, Array]:
    """Multiply floating-point numbers x and y using Dekker's product.

        Returns (xyh, xyl) such that

          xyh = RN(x * y)
          x * y == xyh + xyl

        Applicability:
          x * y is finite and no overflow occurs within the Dekker's product [scale==True, fix_overflow == False],
          x * y is finite [scale==True, fix_overflow == True]
          If scale == False,
            -986     <= x, y <= 1007, abs(x * y) < 62940 for float16
            -7.5e33  <= x, y <= 8.3e34                   for float32
            -4.3e299 <= x, y <= 1.3e300                  for float64

        Accuracy:
          The pair (xyh, xyl) represents the floating-point product `x * y` exactly.

    FA tracing parameters: scale=False fix_overflow=False assume_fma=False"""

    (
        x,
        y,
    ) = promote_args_inexact("two_prod_unsafe", x, y)
    multiply_x_y = lax.mul(x, y)
    N_lst = [numpy.float64(134217728.0), numpy.float32(4096.0), numpy.float16(64.0)]
    N = N_lst[_np_dtypes.index(x.dtype.type)]
    multiply_N_x = lax.mul(N, x)
    subtract_56 = lax.sub(multiply_N_x, lax.sub(multiply_N_x, x))
    multiply_N_y = lax.mul(N, y)
    subtract_60 = lax.sub(multiply_N_y, lax.sub(multiply_N_y, y))
    subtract_y_subtract_60 = lax.sub(y, subtract_60)
    subtract_x_subtract_56 = lax.sub(x, subtract_56)
    return [
        multiply_x_y,
        lax.add(
            lax.add(
                lax.add(
                    lax.add(lax.neg(multiply_x_y), lax.mul(subtract_56, subtract_60)),
                    lax.mul(subtract_56, subtract_y_subtract_60),
                ),
                lax.mul(subtract_x_subtract_56, subtract_60),
            ),
            lax.mul(subtract_x_subtract_56, subtract_y_subtract_60),
        ),
    ]


@jit
def two_prod_general(x: ArrayLike, y: ArrayLike) -> list[Array, Array]:
    """Multiply floating-point numbers x and y using Dekker's product.

        Returns (xyh, xyl) such that

          xyh = RN(x * y)
          x * y == xyh + xyl

        Applicability:
          x * y is finite and no overflow occurs within the Dekker's product [scale==True, fix_overflow == False],
          x * y is finite [scale==True, fix_overflow == True]
          If scale == False,
            -986     <= x, y <= 1007, abs(x * y) < 62940 for float16
            -7.5e33  <= x, y <= 8.3e34                   for float32
            -4.3e299 <= x, y <= 1.3e300                  for float64

        Accuracy:
          The pair (xyh, xyl) represents the floating-point product `x * y` exactly.

    FA tracing parameters: scale=True fix_overflow=True assume_fma=False"""

    (
        x,
        y,
    ) = promote_args_inexact("two_prod_general", x, y)
    multiply_x_y = lax.mul(x, y)
    abs_x = lax.abs(x)
    x_max_lst = [numpy.float64(1.7976931080746007e308), numpy.float32(3.401993e38), numpy.float16(63500.0)]
    dtype_index_x = _np_dtypes.index(x.dtype.type)
    x_max = x_max_lst[dtype_index_x]
    constant_1 = numpy.array(1, dtype=x.dtype.type)
    lt_77 = lax.lt(abs_x, constant_1)
    N_lst = [numpy.float64(134217728.0), numpy.float32(4096.0), numpy.float16(64.0)]
    N = N_lst[dtype_index_x]
    invN_lst = [numpy.float64(7.450580596923828e-09), numpy.float32(0.00024414062), numpy.float16(0.01563)]
    invN = invN_lst[dtype_index_x]
    select_79 = jnp.where(lt_77, x, lax.mul(x, invN))
    multiply_N_select_79 = lax.mul(N, select_79)
    subtract_82 = lax.sub(multiply_N_select_79, lax.sub(multiply_N_select_79, select_79))
    constant_0 = numpy.array(0, dtype=x.dtype.type)
    negative_x_max = lax.neg(x_max)
    select_122 = jnp.where(
        lax.le(abs_x, x_max),
        jnp.where(lt_77, subtract_82, lax.mul(subtract_82, N)),
        jnp.where(lax.lt(x, constant_0), negative_x_max, x_max),
    )
    abs_y = lax.abs(y)
    lt_93 = lax.lt(abs_y, constant_1)
    select_95 = jnp.where(lt_93, y, lax.mul(y, invN))
    multiply_N_select_95 = lax.mul(N, select_95)
    subtract_98 = lax.sub(multiply_N_select_95, lax.sub(multiply_N_select_95, select_95))
    select_126 = jnp.where(
        lax.le(abs_y, x_max),
        jnp.where(lt_93, subtract_98, lax.mul(subtract_98, N)),
        jnp.where(lax.lt(y, constant_0), negative_x_max, x_max),
    )
    multiply_129 = lax.mul(select_122, select_126)
    largest_lst = [numpy.float64(1.7976931348623157e308), numpy.float32(3.4028235e38), numpy.float16(65500.0)]
    largest = largest_lst[dtype_index_x]
    subtract_y_select_126 = lax.sub(y, select_126)
    subtract_x_select_122 = lax.sub(x, select_122)
    return [
        multiply_x_y,
        jnp.where(
            lax.le(lax.abs(multiply_129), largest),
            lax.add(
                lax.add(
                    lax.add(lax.add(lax.neg(multiply_x_y), multiply_129), lax.mul(select_122, subtract_y_select_126)),
                    lax.mul(subtract_x_select_122, select_126),
                ),
                lax.mul(subtract_x_select_122, subtract_y_select_126),
            ),
            constant_0,
        ),
    ]


@jit
def fma_unsafe(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> Array:
    """Emulate fused add-multiply: x * y + z

        Applicability:
          x * y and x * y + z are finite.

        Accuracy:

          The ULP distance between the result and correct value is <= 1.

          fma(x, y, z) is exact (ULP distance is zero) for most inputs.
          If not, either

          - overflow occured in fma arithmetics (the return value is nan).
            Use fix_overflow to workaround overflow in Dekker product or 2Sum;
          - underflow occured in fma arithmetics. Use possibly_zero_z to
            fix z == 0 cases;
          - or the target uses flush-to-zero (FTZ) mode. Disable FTZ in target.

        Algorithms:
          a9: Algorithm 9 from https://hal.science/hal-04575249
          a7: Algorithm 7 from https://hal.science/hal-04624238
          a8: Algorithm 8 from https://hal.science/hal-04624238
          apmath: Using floating-point expansions.

    FA tracing parameters: fix_overflow=False assume_fma=False algorithm=apmath functional=True scale=False size=None possibly_zero_z=False
    """

    (
        x,
        y,
        z,
    ) = promote_args_inexact("fma_unsafe", x, y, z)
    multiply_x_y = lax.mul(x, y)
    N_lst = [numpy.float64(134217728.0), numpy.float32(4096.0), numpy.float16(64.0)]
    N = N_lst[_np_dtypes.index(x.dtype.type)]
    multiply_N_x = lax.mul(N, x)
    subtract_56 = lax.sub(multiply_N_x, lax.sub(multiply_N_x, x))
    multiply_N_y = lax.mul(N, y)
    subtract_60 = lax.sub(multiply_N_y, lax.sub(multiply_N_y, y))
    subtract_y_subtract_60 = lax.sub(y, subtract_60)
    subtract_x_subtract_56 = lax.sub(x, subtract_56)
    add_71 = lax.add(
        lax.add(
            lax.add(
                lax.add(lax.neg(multiply_x_y), lax.mul(subtract_56, subtract_60)), lax.mul(subtract_56, subtract_y_subtract_60)
            ),
            lax.mul(subtract_x_subtract_56, subtract_60),
        ),
        lax.mul(subtract_x_subtract_56, subtract_y_subtract_60),
    )
    add_149 = lax.add(multiply_x_y, add_71)
    add_z_add_149 = lax.add(z, add_149)
    subtract_156 = lax.sub(add_z_add_149, z)
    add_160 = lax.add(lax.sub(z, lax.sub(add_z_add_149, subtract_156)), lax.sub(add_149, subtract_156))
    add_162 = lax.add(add_z_add_149, add_160)
    constant_0 = numpy.array(0, dtype=z.dtype.type)
    subtract_163 = lax.sub(add_162, add_z_add_149)
    add_167 = lax.add(lax.sub(add_z_add_149, lax.sub(add_162, subtract_163)), lax.sub(add_160, subtract_163))
    logical_and_242 = jnp.logical_and(lax.ne(add_162, constant_0), lax.ne(add_167, constant_0))
    constant_1 = numpy.int64(1)
    select_210 = jnp.where(lax.eq(add_167, constant_0), add_162, add_167)
    subtract_150 = lax.sub(add_149, multiply_x_y)
    add_154 = lax.add(lax.sub(multiply_x_y, lax.sub(add_149, subtract_150)), lax.sub(add_71, subtract_150))
    add_212 = lax.add(select_210, add_154)
    ne_226 = lax.ne(add_212, constant_0)
    subtract_213 = lax.sub(add_212, select_210)
    add_217 = lax.add(lax.sub(select_210, lax.sub(add_212, subtract_213)), lax.sub(add_154, subtract_213))
    ne_218 = lax.ne(add_217, constant_0)
    logical_and_257 = jnp.logical_and(ne_226, ne_218)
    add_325 = lax.add(jnp.where(logical_and_242, constant_1, constant_0), jnp.where(logical_and_257, constant_1, constant_0))
    logical_or_231 = jnp.logical_or(ne_226, ne_218)
    select_222 = jnp.where(lax.eq(add_217, constant_0), add_212, add_217)
    return lax.add(
        lax.add(
            lax.add(
                jnp.where(jnp.logical_and(lax.eq(add_325, constant_0), logical_or_231), select_222, constant_0),
                jnp.where(logical_and_242, add_162, constant_0),
            ),
            jnp.where(
                jnp.logical_and(jnp.logical_and(logical_and_257, jnp.logical_not(logical_and_242)), ne_218),
                add_212,
                constant_0,
            ),
        ),
        lax.add(
            jnp.where(jnp.logical_and(lax.eq(add_325, constant_1), logical_or_231), select_222, constant_0),
            jnp.where(jnp.logical_and(jnp.logical_and(logical_and_242, logical_and_257), ne_218), add_212, constant_0),
        ),
    )


@jit
def fma_general(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> Array:
    """Emulate fused add-multiply: x * y + z

        Applicability:
          x * y and x * y + z are finite.

        Accuracy:

          The ULP distance between the result and correct value is <= 1.

          fma(x, y, z) is exact (ULP distance is zero) for most inputs.
          If not, either

          - overflow occured in fma arithmetics (the return value is nan).
            Use fix_overflow to workaround overflow in Dekker product or 2Sum;
          - underflow occured in fma arithmetics. Use possibly_zero_z to
            fix z == 0 cases;
          - or the target uses flush-to-zero (FTZ) mode. Disable FTZ in target.

        Algorithms:
          a9: Algorithm 9 from https://hal.science/hal-04575249
          a7: Algorithm 7 from https://hal.science/hal-04624238
          a8: Algorithm 8 from https://hal.science/hal-04624238
          apmath: Using floating-point expansions.

    FA tracing parameters: fix_overflow=True assume_fma=False algorithm=a7 functional=True scale=True size=None possibly_zero_z=True
    """

    (
        x,
        y,
        z,
    ) = promote_args_inexact("fma_general", x, y, z)
    multiply_x_y = lax.mul(x, y)
    add_372 = lax.add(multiply_x_y, z)
    abs_x = lax.abs(x)
    x_max_lst = [numpy.float64(1.7976931080746007e308), numpy.float32(3.401993e38), numpy.float16(63500.0)]
    dtype_index_x = _np_dtypes.index(x.dtype.type)
    x_max = x_max_lst[dtype_index_x]
    constant_1 = numpy.array(1, dtype=x.dtype.type)
    lt_77 = lax.lt(abs_x, constant_1)
    N_lst = [numpy.float64(134217728.0), numpy.float32(4096.0), numpy.float16(64.0)]
    N = N_lst[dtype_index_x]
    invN_lst = [numpy.float64(7.450580596923828e-09), numpy.float32(0.00024414062), numpy.float16(0.01563)]
    invN = invN_lst[dtype_index_x]
    select_79 = jnp.where(lt_77, x, lax.mul(x, invN))
    multiply_N_select_79 = lax.mul(N, select_79)
    subtract_82 = lax.sub(multiply_N_select_79, lax.sub(multiply_N_select_79, select_79))
    constant_0 = numpy.array(0, dtype=x.dtype.type)
    negative_x_max = lax.neg(x_max)
    select_122 = jnp.where(
        lax.le(abs_x, x_max),
        jnp.where(lt_77, subtract_82, lax.mul(subtract_82, N)),
        jnp.where(lax.lt(x, constant_0), negative_x_max, x_max),
    )
    abs_y = lax.abs(y)
    lt_93 = lax.lt(abs_y, constant_1)
    select_95 = jnp.where(lt_93, y, lax.mul(y, invN))
    multiply_N_select_95 = lax.mul(N, select_95)
    subtract_98 = lax.sub(multiply_N_select_95, lax.sub(multiply_N_select_95, select_95))
    select_126 = jnp.where(
        lax.le(abs_y, x_max),
        jnp.where(lt_93, subtract_98, lax.mul(subtract_98, N)),
        jnp.where(lax.lt(y, constant_0), negative_x_max, x_max),
    )
    multiply_129 = lax.mul(select_122, select_126)
    largest_lst = [numpy.float64(1.7976931348623157e308), numpy.float32(3.4028235e38), numpy.float16(65500.0)]
    largest = largest_lst[dtype_index_x]
    subtract_y_select_126 = lax.sub(y, select_126)
    subtract_x_select_122 = lax.sub(x, select_122)
    subtract_373 = lax.sub(add_372, multiply_x_y)
    add_383 = lax.add(
        jnp.where(
            lax.le(lax.abs(multiply_129), largest),
            lax.add(
                lax.add(
                    lax.add(lax.add(lax.neg(multiply_x_y), multiply_129), lax.mul(select_122, subtract_y_select_126)),
                    lax.mul(subtract_x_select_122, select_126),
                ),
                lax.mul(subtract_x_select_122, subtract_y_select_126),
            ),
            constant_0,
        ),
        jnp.where(
            lax.le(lax.abs(subtract_373), largest),
            lax.add(lax.sub(multiply_x_y, lax.sub(add_372, subtract_373)), lax.sub(z, subtract_373)),
            constant_0,
        ),
    )
    add_384 = lax.add(add_372, add_383)
    subtract_385 = lax.sub(add_384, add_372)
    return lax.add(add_384, jnp.where(lax.le(lax.abs(subtract_385), largest), lax.sub(add_383, subtract_385), constant_0))
