import numpy
import itertools
import functional_algorithms as fa
from functional_algorithms import apmath_algorithms

import pytest


@pytest.fixture(scope="function", params=["float16", "float32", "float64"])
def dtype_name(request):
    return request.param


def test_square(dtype_name):
    dtype = getattr(numpy, dtype_name)
    ctx = fa.Context(paths=[apmath_algorithms])

    graph = ctx.trace(getattr(apmath_algorithms, "square"), list[dtype])
    graph = graph.rewrite(fa.targets.numpy, fa.rewrite, fa.rewrite)

    func = fa.targets.numpy.as_function(graph, debug=0, force_cast_arguments=False)

    size = 100000
    samples = fa.utils.real_samples(size, dtype=dtype)

    result = func([samples])

    np_ctx = fa.utils.NumpyContext(dtype)
    expected = fa.apmath.square(np_ctx, [samples], functional=True, size=2)

    numpy.testing.assert_equal(result, expected)


_ranges = [
    "0..eps",
    "eps..sqrt_eps",
    "sqrt_eps..1",
    "1..1/sqrt_eps",
    "1/sqrt_eps..sqrt_max",
    "sqrt_max..max",
]


@pytest.mark.parametrize("xrange,yrange,zrange", list(itertools.product(_ranges, _ranges, _ranges)))
def test_fma(dtype_name, xrange, yrange, zrange):
    import mpmath

    dtype = getattr(numpy, dtype_name)
    fi = numpy.finfo(dtype)

    def fix_range(r):
        if isinstance(r, str):
            mn, mx = r.split("..")
            ns = fi.__dict__.copy()
            ns.update(numpy=numpy, sqrt_max=numpy.sqrt(fi.max), sqrt_eps=numpy.sqrt(fi.eps), sqrt=numpy.sqrt)
            mn = eval(mn, {}, ns)
            mx = eval(mx, {}, ns)
        else:
            mn, mx = r
        mn, mx = dtype(mn), dtype(mx)
        return mn, mx

    xrange = fix_range(xrange)
    yrange = fix_range(yrange)
    zrange = fix_range(zrange)

    npctx = fa.utils.NumpyContext(dtype)
    ctx = fa.Context(paths=[apmath_algorithms])

    graph = ctx.trace(getattr(apmath_algorithms, "fma"), dtype, dtype, dtype)
    graph = graph.rewrite(fa.targets.numpy, fa.rewrite, fa.rewrite)

    func = fa.targets.numpy.as_function(graph, debug=0, force_cast_arguments=False)

    size = 5 if dtype == numpy.float16 else 10
    x, y, z = fa.utils.real_triple_samples(
        (size, size, size),
        dtype=dtype,
        min_value=(xrange[0], yrange[0], zrange[0]),
        max_value=(xrange[1], yrange[1], zrange[1]),
        target_func="fma",
        include_infinity=False,
    )

    x = numpy.concatenate((x, x))
    y = numpy.concatenate((y, y))
    z = numpy.concatenate((-z, z))

    mp_ctx = mpmath.mp

    def fma_mpmath(x, y, z):
        # Reference implementation of FMA using mpmath.

        # Warning: since mpmath does not support subnormals, when the
        # FMA arithmetics involves subnormals, the results may
        # slightly diverge from hardware FMA.
        x_mp = fa.utils.float2mpf(mp_ctx, x)
        y_mp = fa.utils.float2mpf(mp_ctx, y)
        z_mp = fa.utils.float2mpf(mp_ctx, z)
        return numpy.array(
            [
                fa.utils.mpf2float(dtype, mp_ctx.fadd(mp_ctx.fmul(x_, y_, exact=True), z_, exact=True))
                for x_, y_, z_ in zip(x_mp, y_mp, z_mp)
            ],
            dtype=dtype,
        )

    reference_fma = None
    if dtype in {numpy.float32, numpy.float64}:
        # Try using pyfma.fma when available as it is much faster than
        # fmp_mpmath and it will support subnormals.
        try:
            import pyfma

            reference_fma = pyfma.fma
        except ImportError:
            pass

    if reference_fma is None:
        reference_fma = fma_mpmath

    def issubnormal(x):
        # assumes isfinite
        ax = abs(x)
        return numpy.logical_and(numpy.isfinite(x), numpy.logical_and(ax < fi.smallest_normal, ax > 0))

    def hassubnormal(*args):
        assert len(args) > 0
        if len(args) == 1:
            return issubnormal(args[0])
        return numpy.logical_or(issubnormal(args[0]), hassubnormal(*args[1:]))

    expected = reference_fma(x, y, z)
    subnormals_mask = hassubnormal(expected, x, y, z, x * y)
    normal_indices = numpy.where(numpy.logical_not(subnormals_mask))
    subnormal_indices = numpy.where(subnormals_mask)

    result = func(x, y, z)

    ulps = fa.utils.diff_ulp(result, expected, flush_subnormals=True)

    difference = result - expected

    ind = numpy.where(ulps != 0)

    if ind[0].size > 0:
        fa.utils.show_ulp(ulps[normal_indices], title="No subnormals")
        fa.utils.show_ulp(ulps[subnormal_indices], title="FMA aithmetics involves subnormals")

    for i in ind[0]:
        if numpy.isfinite(x[i] * y[i]):
            assert ulps[i] <= 1
        else:
            assert not numpy.isfinite(result[i])
            assert not numpy.isfinite(expected[i])
