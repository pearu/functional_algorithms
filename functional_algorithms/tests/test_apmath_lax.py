# If jax and functional_algorithms live in different environments, you
# run this test script in an environment containing jax as follows:
#
# export PYTHONPATH=<path-to-functional_algorithms-folder>
# pytest <path-to-functional_algorithms-folder>/functional_algorithms/tests/test_apmath_lax.py


import pytest

jax = pytest.importorskip("jax")

import timeit
import numpy
import mpmath

import jax.numpy as jnp
import functional_algorithms as fa
import functional_algorithms.apmath_lax as ap


@pytest.fixture(scope="function", params=[jnp.float16, jnp.float32, jnp.float64])
def dtype(request):
    if request.param == jnp.float64 and not jax.config.jax_enable_x64:
        pytest.skip("Use JAX_ENABLE_X64=1 to enable 64-bit floating-point tests")
    return request.param


@pytest.fixture(scope="function", params=["general", "unsafe"])
def suffix(request):
    return request.param


def _samples(sizes, dtype, target_func=None):
    npdtype = dtype.dtype.type
    if len(sizes) == 1:
        samples = (map(jnp.array, fa.utils.real_samples(sizes, dtype=npdtype, include_infinity=False, include_huge=False)),)
    elif len(sizes) == 2:
        samples = map(
            jnp.array,
            fa.utils.real_pair_samples(
                sizes, dtype=npdtype, include_infinity=False, include_huge=False, target_func=target_func
            ),
        )
    elif len(sizes) == 3:
        samples = map(
            jnp.array,
            fa.utils.real_triple_samples(
                sizes, dtype=npdtype, include_infinity=False, include_huge=False, target_func=target_func
            ),
        )
    else:
        raise NotImplementedError(sizes)
    return tuple(s.block_until_ready() for s in samples)


def _benchmark(func, simple_func, args):
    timeit_number = 100
    timeit_repeat = 100
    warmup_repeat = 5

    for _ in range(warmup_repeat):
        result = func(*args)
        result[-1].block_until_ready() if isinstance(result, (tuple, list)) else result.block_until_ready()

    timing = timeit.repeat(
        "result = func(*args); (result[-1].block_until_ready() if isinstance(result, (tuple, list)) else result.block_until_ready())",
        number=timeit_number,
        repeat=timeit_repeat,
        globals=dict(func=func, args=args),
    )

    for _ in range(warmup_repeat):
        result = simple_func(*args)
        result[-1].block_until_ready() if isinstance(result, (tuple, list)) else result.block_until_ready()

    simple_timing = timeit.repeat(
        "result = func(*args); (result[-1].block_until_ready() if isinstance(result, (tuple, list)) else result.block_until_ready())",
        number=timeit_number,
        repeat=timeit_repeat,
        globals=dict(func=simple_func, args=args),
    )
    timing = numpy.min(timing)
    simple_timing = numpy.min(simple_timing)

    print(f"overhead factor: {timing / simple_timing:.2f}", end=" ")


def _accuracy(func, reference_func, args):
    result = numpy.asarray(func(*args))
    expected = numpy.asarray(reference_func(*args))
    ulps = fa.utils.diff_ulp(result, expected, flush_subnormals=True)

    if 0:
        i = numpy.where(ulps > 1)
        i = (i[0][:1],)
        print(f"{result[i]=} {expected[i]=} {tuple(a[i] for a in args)=}")
    fa.utils.show_ulp(ulps, title=f"\n{func.__name__}")


def test_two_sum(dtype, suffix):
    size = 100

    args = _samples((size, size), dtype, target_func="add")

    func = getattr(ap, "two_sum_" + suffix)

    def simple_func(x, y):
        return x + y

    s, t = func(*args)
    assert s.dtype == dtype
    assert jnp.array_equal(s, simple_func(*args))

    _benchmark(func, jax.jit(simple_func), args)
    return

    if 0:
        f = func
        print(dir(f))
        l = f.lower(x, y)
        print(dir(l))
        c = l.compile()
        print(c.as_text())
        print(dir(c))


def test_two_prod(dtype, suffix):
    size = 100

    args = _samples((size, size), dtype, target_func="mul")

    func = getattr(ap, "two_prod_" + suffix)

    def simple_func(x, y):
        return x * y

    s, t = func(*args)
    assert s.dtype == dtype
    assert jnp.array_equal(s, simple_func(*args))

    _benchmark(func, jax.jit(simple_func), args)


def test_fma(dtype, suffix):

    func = getattr(ap, "fma_" + suffix)

    def simple_func(x, y, z):
        return x * y + z

    args = _samples((10,) * 3, dtype, target_func="fma")

    r = func(*args)
    assert r.dtype == dtype

    def reference_func(x, y, z):
        x = numpy.asarray(x)
        y = numpy.asarray(y)
        z = numpy.asarray(z)
        if dtype in {jnp.float32, jnp.float64}:
            try:
                import pyfma

                if not hasattr(numpy, "find_common_type"):

                    def find_common_type(array_types, scalar_types):
                        assert not array_types
                        if len(scalar_types) == 1:
                            return scalar_types[0]
                        elif len(scalar_types) == 2:
                            return numpy.promote_types(*scalar_types)
                        else:
                            return numpy.promote_types(find_common_type(array_types, scalar_types[:-1]), scalar_types[-1])

                    numpy.find_common_type = find_common_type

                return jnp.array(pyfma.fma(x, y, z))
            except ImportError:
                pass

        mp_ctx = mpmath.mp
        x_mp = fa.utils.float2mpf(mp_ctx, x)
        y_mp = fa.utils.float2mpf(mp_ctx, y)
        z_mp = fa.utils.float2mpf(mp_ctx, z)
        return jnp.array(
            [
                fa.utils.mpf2float(dtype, mp_ctx.fadd(mp_ctx.fmul(x_, y_, exact=True), z_, exact=True))
                for x_, y_, z_ in zip(x_mp, y_mp, z_mp)
            ],
            dtype=dtype,
        )

    args = _samples((100,) * 3, dtype, target_func="fma")
    _accuracy(func, reference_func, args)
    _accuracy(jax.jit(simple_func), reference_func, args)

    _benchmark(func, jax.jit(simple_func), args)
