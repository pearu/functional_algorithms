import numpy
from functional_algorithms import Context, targets, algorithms, utils, rewrite
import pytest


@pytest.fixture(scope="function", params=["python", "numpy", "stablehlo", "xla_client", "cpp"])
def target_name(request):
    return request.param


@pytest.fixture(scope="function", params=["complex64", "complex128", "float32", "float64"])
def dtype_name(request):
    return request.param


@pytest.fixture(
    scope="function",
    params=[
        "absolute",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "hypot",
        "square",
        "sqrt",
        "angle",
        "atan",
        "atanh",
        "tan",
        "tanh",
        "log",
        "log10",
        "log2",
        "exp",
    ],
)
def func_name(request):
    return request.param


@pytest.fixture(
    scope="function",
    params=[
        "absolute",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "square",
        "sqrt",
        "angle",
        "log1p",
        "atan",
        "atanh",
        "tan",
        "tanh",
        "log",
        "log10",
        "log2",
        "exp",
    ],
)
def unary_func_name(request):
    return request.param


@pytest.fixture(scope="function", params=["hypot"])
def binary_func_name(request):
    return request.param


@pytest.fixture(scope="function", params=["no-ftz", "ftz"])
def flush_subnormals(request):
    return {"no-ftz": False, "ftz": True}[request.param]


@pytest.mark.parametrize("dtype", [numpy.complex64, numpy.complex128])
def test_complex_sqrt_preserves_signed_zero_on_branch_cut(dtype):
    ctx = Context(paths=[algorithms])
    graph = ctx.trace(algorithms.sqrt, dtype).rewrite(targets.numpy, rewrite)
    func = targets.numpy.as_function(graph, debug=0)

    values = numpy.array([complex(-4.0, 0.0), complex(-4.0, -0.0)], dtype=dtype)
    expected = numpy.array([complex(0.0, 2.0), complex(0.0, -2.0)], dtype=dtype)

    result = numpy.array([func(value) for value in values], dtype=dtype)

    numpy.testing.assert_equal(result, expected)
    numpy.testing.assert_array_equal(numpy.signbit(result.imag), [False, True])
    numpy.testing.assert_equal(result[1], numpy.conjugate(result[0]))


def test_complex_sqrt_preserves_signed_zero_with_python_target():
    ctx = Context(paths=[algorithms])
    graph = ctx.trace(algorithms.sqrt, complex).rewrite(targets.python, rewrite)
    func = targets.python.as_function(graph)

    positive = func(complex(-4.0, 0.0))
    negative = func(complex(-4.0, -0.0))

    assert positive == complex(0.0, 2.0)
    assert negative == complex(0.0, -2.0)
    assert negative == positive.conjugate()


@pytest.mark.parametrize(("target_name", "dtype"), [("python", float), ("numpy", numpy.float64)])
def test_is_negzero_predicate(target_name, dtype):
    def is_negzero(ctx, x):
        return ctx(ctx.is_negzero(x))

    target = getattr(targets, target_name)
    ctx = Context(paths=[algorithms])
    graph = ctx.trace(is_negzero, dtype).rewrite(target, rewrite)
    func = target.as_function(graph)

    values = [0.0, -0.0, 1.0, -1.0, numpy.inf, -numpy.inf, numpy.nan]

    assert [bool(func(dtype(value))) for value in values] == [False, True, False, False, False, False, False]


@pytest.mark.parametrize(("target_name", "dtype"), [("lax", numpy.float64), ("symbolic", float)])
def test_is_negzero_generates_for_target(target_name, dtype):
    def is_negzero(ctx, x):
        return ctx(ctx.is_negzero(x))

    target = getattr(targets, target_name)
    graph = Context().trace(is_negzero, dtype).rewrite(target, rewrite)

    assert isinstance(graph.tostring(target), str)


@pytest.mark.parametrize(
    ("target_name", "func_name"),
    [
        ("python", "sqrt"),
        ("numpy", "sqrt"),
        ("stablehlo", "sqrt"),
        ("xla_client", "complex_sqrt"),
        ("cpp", "complex_sqrt"),
    ],
)
def test_complex_sqrt_generates_for_target(target_name, func_name):
    target = getattr(targets, target_name)
    enable_alt = target_name == "xla_client"
    default_constant_type = "FloatType" if enable_alt else None
    ctx = Context(paths=[algorithms], enable_alt=enable_alt, default_constant_type=default_constant_type)
    trace_arguments = target.trace_arguments[func_name][0]
    graph = ctx.trace(getattr(algorithms, func_name), *trace_arguments).rewrite(target, rewrite)

    source = graph.tostring(target)
    assert isinstance(source, str)
    if target_name == "xla_client":
        assert "IsNegZero" in source
        assert '#include "xla/client/lib/math.h"' in target.source_file_header


def test_unary(dtype_name, unary_func_name, flush_subnormals):
    if dtype_name.startswith("float") and unary_func_name in {"angle"}:
        pytest.skip(reason=f"{unary_func_name} does not support {dtype_name} inputs")

    dtype = getattr(numpy, dtype_name)
    ctx = Context(paths=[algorithms])

    try:
        graph = ctx.trace(getattr(algorithms, unary_func_name), dtype)
    except NotImplementedError as msg:
        pytest.skip(reason=str(msg))

    graph2 = graph.rewrite(targets.numpy, rewrite)

    func = targets.numpy.as_function(graph2, debug=0)

    params = utils.function_validation_parameters(unary_func_name, dtype_name)
    max_valid_ulp_count = params["max_valid_ulp_count"]
    max_bound_ulp_width = params["max_bound_ulp_width"]
    extra_prec_multiplier = params["extra_prec_multiplier"]

    reference = getattr(
        utils.numpy_with_mpmath(extra_prec_multiplier=extra_prec_multiplier, flush_subnormals=flush_subnormals),
        unary_func_name,
    )

    # samples consist of log-uniform grid of the complex plane plus
    # any extra samples that cover the special regions for the given
    # function.
    size = 51
    if dtype in {numpy.complex64, numpy.complex128}:
        samples = utils.complex_samples(
            (size, size),
            dtype=dtype,
            include_huge=True,
            include_subnormal=not flush_subnormals,
        ).flatten()
    else:
        samples = utils.real_samples(
            size * size,
            dtype=dtype,
            include_subnormal=not flush_subnormals,
        ).flatten()

    samples = numpy.concatenate((samples, utils.extra_samples(unary_func_name, dtype)))

    matches_with_reference, stats = utils.validate_function(
        func,
        reference,
        samples,
        dtype,
        flush_subnormals=flush_subnormals,
        max_valid_ulp_count=max_valid_ulp_count,
        max_bound_ulp_width=max_bound_ulp_width,
    )
    if not matches_with_reference:
        print("Samples:")
        gt3_ulp_total = 0
        gt3_outrange = 0
        ulp_stats = stats["ulp"]
        for ulp in sorted(ulp_stats):
            if ulp >= 0 and ulp <= max_valid_ulp_count:
                outrange = stats["outrange"][ulp]
                if outrange:
                    print(f"  dULP={ulp}: {ulp_stats[ulp]} ({outrange=})")
                else:
                    print(f"  dULP={ulp}: {ulp_stats[ulp]}")
            elif ulp > 0:
                gt3_ulp_total += ulp_stats[ulp]
                gt3_outrange += stats["outrange"][ulp]
            elif ulp == -1:
                c = ulp_stats[ulp]
                if c:
                    print(f"  total number of mismatches: {c}")
            else:
                assert 0, ulp  # unreachable
        else:
            if gt3_outrange:
                print(f"  dULP>{max_valid_ulp_count}: {gt3_ulp_total} (outrange={gt3_outrange})")
            else:
                print(f"  dULP>{max_valid_ulp_count}: {gt3_ulp_total}")

    assert matches_with_reference  # warning: also reference may be wrong


def test_binary(dtype_name, binary_func_name, flush_subnormals):
    if dtype_name.startswith("complex") and binary_func_name in {"hypot"}:
        pytest.skip(reason=f"{binary_func_name} does not support {dtype_name} inputs")

    dtype = getattr(numpy, dtype_name)
    ctx = Context(paths=[algorithms])

    graph = ctx.trace(getattr(algorithms, binary_func_name), dtype, dtype)

    graph2 = graph.rewrite(targets.numpy, rewrite)
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
    try:
        graph = ctx.trace(getattr(algorithms, func_name)).rewrite(target, rewrite)
    except NotImplementedError as msg:
        pytest.skip(reason=str(msg))
    src = graph.tostring(target)
    assert isinstance(src, str)
    assert "symbol__tmp" not in src
