import numpy
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
