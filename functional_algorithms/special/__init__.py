__all__ = [
    "hyp0f1",
]

from . import utils


def hyp0f1(b, z, zero_indices=None, enable_largest_correction=False):
    r"""The 0F1 hypergeometric function.

    Functional implementation of :obj:`scipy.special.hyp0f1`.

    .. math::

         \mathrm{hyp0f1}(b, x) = {}_0F_1(x; , b) = \sum_{k=0}^\infty \frac{1}{(b)_kk!}x^k

    where :math:`(\cdot)_k` is the Pochammer symbol (refer to :func:`~jax.scipy.special.poch`).

    Restrictions:
      b is positive integer
      z is floating point value or array of floating point values

    zero_indices is a list of integers or range start/stop arguments
    of hyp0f1 zeros that will be corrected for accuracy. By default, a
    selected zeros will be corrected. To disable zeros corrections,
    set zero_indices to empty list.
    """
    import numpy
    import functional_algorithms as fa

    assert isinstance(b, int)
    assert isinstance(z, (numpy.ndarray, numpy.floating, float))

    if isinstance(z, numpy.ndarray):
        dtype = z.dtype.type
    else:
        dtype = type(z)

    import mpmath
    import warnings

    max_prec = {numpy.float16: 27, numpy.float32: 149, numpy.float64: 1074}[dtype]
    with warnings.catch_warnings(action="ignore"):
        mp_ctx = mpmath.mp
        with mp_ctx.workprec(max_prec):
            ctx = fa.utils.NumpyContext(dtype, mpmath_context=mp_ctx)
            seq = fa.apmath.hypergeometric0f1(
                ctx,
                b,
                [z],
                functional=True,
                size=2,
                dtype=dtype,
                zero_indices=zero_indices,
                enable_largest_correction=enable_largest_correction,
            )
    return sum(seq[:-1], seq[-1])
