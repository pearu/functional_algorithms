import inspect
import numpy
import math
import cmath
from functional_algorithms import Context, targets, algorithms, utils
import pytest


@pytest.fixture(scope="function", params=["python", "numpy", "stablehlo"])
def target_name(request):
    return request.param


@pytest.fixture(scope="function", params=["complex64", "complex128", "float32", "float64"])
def dtype_name(request):
    return request.param


@pytest.fixture(scope="function", params=["asin", "square", "hypot"])
def func_name(request):
    return request.param


@pytest.fixture(scope="function", params=["asin", "square"])
def unary_func_name(request):
    return request.param


@pytest.fixture(scope="function", params=["hypot"])
def binary_func_name(request):
    return request.param


def test_unary(dtype_name, unary_func_name):
    dtype = getattr(numpy, dtype_name)
    ctx = Context(paths=[algorithms])

    graph = ctx.trace(getattr(algorithms, unary_func_name), dtype)

    graph2 = graph.implement_missing(targets.numpy).simplify()
    func = targets.numpy.as_function(graph2, debug=0)
    reference = getattr(utils.numpy_with_mpmath(extra_prec_multiplier=10), unary_func_name)

    if dtype in {numpy.complex64, numpy.complex128}:
        samples = utils.complex_samples((26, 26), dtype=dtype, include_huge=True).flatten()
    else:
        samples = utils.real_samples(500, dtype=dtype).flatten()
    matches_with_reference = utils.validate_function(func, reference, samples, dtype)
    assert matches_with_reference  # warning: also reference may be wrong


def test_binary(dtype_name, binary_func_name):
    if dtype_name.startswith("complex") and binary_func_name in {"hypot"}:
        pytest.skip(reason=f"{binary_func_name} does not support {dtype_name} inputs")

    dtype = getattr(numpy, dtype_name)
    ctx = Context(paths=[algorithms])

    graph = ctx.trace(getattr(algorithms, binary_func_name), dtype, dtype)

    graph2 = graph.implement_missing(targets.numpy).simplify()
    func = targets.numpy.as_function(graph2, debug=1)
    reference = getattr(utils.numpy_with_mpmath(extra_prec_multiplier=10), binary_func_name)

    if dtype in {numpy.complex64, numpy.complex128}:
        samples = utils.complex_samples((26, 26), dtype=dtype, include_huge=True).flatten()
        samples = [(sample, sample) for sample in samples]  # TODO: make better samples
    else:
        samples = utils.complex_samples((26, 26), dtype=dtype, include_huge=True).flatten()
        samples = [(x, y) for x, y in zip(samples.real, samples.imag)]
    matches_with_reference = utils.validate_function(func, reference, samples, dtype)
    assert matches_with_reference  # warning: also reference may be wrong


def test_target(func_name, target_name):
    # tests that all functions in algorithms have implementations to
    # all targets
    target = getattr(targets, target_name)
    ctx = Context(paths=[algorithms])
    graph = ctx.trace(getattr(algorithms, func_name)).implement_missing(target).simplify()
    src = graph.tostring(target)
    assert isinstance(src, str)
