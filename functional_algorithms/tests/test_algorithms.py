import inspect
import numpy
import math
import cmath
from functional_algorithms import Context, targets, algorithms, utils
import pytest


@pytest.fixture(scope="function", params=["python", "numpy", "stablehlo", "xla_client", "cpp"])
def target_name(request):
    return request.param


@pytest.fixture(scope="function", params=["complex64", "complex128", "float32", "float64"])
def dtype_name(request):
    return request.param


@pytest.fixture(scope="function", params=["absolute", "acos", "asin", "asinh", "hypot", "square"])
def func_name(request):
    return request.param


@pytest.fixture(scope="function", params=["absolute", "acos", "asin", "asinh", "square"])
def unary_func_name(request):
    return request.param


@pytest.fixture(scope="function", params=["hypot"])
def binary_func_name(request):
    return request.param


@pytest.fixture(scope="function", params=[False, True])
def flush_subnormals(request):
    return request.param


def test_unary(dtype_name, unary_func_name, flush_subnormals):
    dtype = getattr(numpy, dtype_name)
    ctx = Context(paths=[algorithms])

    graph = ctx.trace(getattr(algorithms, unary_func_name), dtype)
    graph2 = graph.implement_missing(targets.numpy).simplify()

    func = targets.numpy.as_function(graph2, debug=0)

    if unary_func_name in {"acos", "asin", "asinh"}:
        extra_prec_multiplier = 20
    else:
        extra_prec_multiplier = 1
    reference = getattr(
        utils.numpy_with_mpmath(extra_prec_multiplier=extra_prec_multiplier, flush_subnormals=flush_subnormals),
        unary_func_name,
    )

    size = 31
    if dtype in {numpy.complex64, numpy.complex128}:
        samples = utils.complex_samples(
            (size, size), dtype=dtype, include_huge=True, include_subnormal=not flush_subnormals
        ).flatten()
    else:
        samples = utils.real_samples(size * size, dtype=dtype, include_subnormal=not flush_subnormals).flatten()

    matches_with_reference, ulp_stats = utils.validate_function(
        func, reference, samples, dtype, flush_subnormals=flush_subnormals
    )
    if not matches_with_reference:
        print("Samples:")
        gt3_ulp_total = 0
        for ulp in sorted(ulp_stats):
            if ulp in {0, 1, 2, 3}:
                print(f"  dULP={ulp}: {ulp_stats[ulp]}")
            elif ulp > 0:
                gt3_ulp_total += ulp_stats[ulp]
        else:
            print(f"  dULP>3: {gt3_ulp_total}")

    assert matches_with_reference  # warning: also reference may be wrong

    extra_samples = []
    if unary_func_name == "absolute" and dtype_name.startswith("complex"):
        extra_samples.extend([1.0011048e35 + 3.4028235e38j])

    if extra_samples:
        samples = numpy.array(extra_samples, dtype=dtype)
        matches_with_reference, ulp_stats = utils.validate_function(
            func, reference, samples, dtype, flush_subnormals=flush_subnormals
        )
        if not matches_with_reference:
            print("Extra samples:")
            gt3_ulp_total = 0
            for ulp in sorted(ulp_stats):
                if ulp in {0, 1, 2, 3}:
                    print(f"  dULP={ulp}: {ulp_stats[ulp]}")
                elif ulp > 0:
                    gt3_ulp_total += ulp_stats[ulp]
            else:
                print(f"  dULP>3: {gt3_ulp_total}")
        assert matches_with_reference  # warning: also reference may be wrong


def test_binary(dtype_name, binary_func_name, flush_subnormals):
    if dtype_name.startswith("complex") and binary_func_name in {"hypot"}:
        pytest.skip(reason=f"{binary_func_name} does not support {dtype_name} inputs")

    dtype = getattr(numpy, dtype_name)
    ctx = Context(paths=[algorithms])

    graph = ctx.trace(getattr(algorithms, binary_func_name), dtype, dtype)

    graph2 = graph.implement_missing(targets.numpy).simplify()
    func = targets.numpy.as_function(graph2, debug=1)
    reference = getattr(utils.numpy_with_mpmath(extra_prec_multiplier=10, flush_subnormals=flush_subnormals), binary_func_name)

    if dtype in {numpy.complex64, numpy.complex128}:
        samples1, samples2 = utils.complex_pair_samples(
            ((26, 26), (13, 13)), dtype=dtype, include_huge=True, include_subnormal=not flush_subnormals
        )
        samples = [(x, y) for x, y in zip(samples1, samples2)]
    else:
        samples1, samples2 = utils.real_pair_samples(
            (26, 26), dtype=dtype, include_huge=True, include_subnormal=not flush_subnormals
        )
        samples = [(x, y) for x, y in zip(samples1, samples2)]

    matches_with_reference, _ = utils.validate_function(func, reference, samples, dtype, flush_subnormals=flush_subnormals)
    assert matches_with_reference  # warning: also reference may be wrong


def test_target(func_name, target_name):
    # tests that all functions in algorithms have implementations to
    # all targets
    target = getattr(targets, target_name)
    ctx = Context(paths=[algorithms], enable_alt=dict(xla_client=True).get(target_name, False), default_constant_type="DType2")
    graph = ctx.trace(getattr(algorithms, func_name)).implement_missing(target).simplify()
    src = graph.tostring(target)
    assert isinstance(src, str)
    assert "symbol__tmp" not in src
