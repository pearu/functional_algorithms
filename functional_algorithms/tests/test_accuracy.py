import numpy
import functional_algorithms as fa
import pytest
import warnings


@pytest.fixture(scope="function", params=["jax"])
def backend(request):
    return request.param


@pytest.fixture(scope="function", params=["cpu", "cuda"])
def device(request):
    return request.param


@pytest.fixture(scope="function", params=list(sig["name"] for sig in fa.filter_signatures("C", "C")))
def unary_func_name(request):
    return request.param


@pytest.fixture(scope="function", params=[numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
def dtype(request):
    return request.param


def test_unary(unary_func_name, backend, device, dtype):
    pytest.importorskip(backend)

    func = getattr(getattr(fa.utils, f"numpy_with_{backend}")(device=device), unary_func_name)

    if not func.backend_is_available(device):
        pytest.skip(f"{device} support is unavailable")
    params = fa.utils.function_validation_parameters(unary_func_name, dtype)

    max_valid_ulp_count = params["max_valid_ulp_count"]
    extra_prec_multiplier = params["extra_prec_multiplier"]
    samples_limits = params["samples_limits"]
    # JAX flushes subnormals to zero
    include_subnormal = backend != "jax"

    mpmath = fa.utils.numpy_with_mpmath(extra_prec_multiplier=extra_prec_multiplier)

    reference = getattr(mpmath, unary_func_name)
    npy_reference = getattr(fa.utils.numpy_with_numpy(), unary_func_name)

    re_blocks, im_blocks = 101, 51
    re_blocksize, im_blocksize = 20, 20
    re_size, im_size = re_blocks * re_blocksize, im_blocks * im_blocksize

    if dtype in {numpy.complex64, numpy.complex128}:
        samples = fa.utils.complex_samples(
            (re_size, im_size), dtype=dtype, include_subnormal=include_subnormal, **samples_limits
        )
    else:
        pytest.skip(f"{dtype} support not implemented")
    assert samples.shape == (im_size, re_size)

    expected = reference.call(samples)
    result = func(samples)

    ulp = fa.utils.diff_ulp(result, expected, flush_subnormals=not include_subnormal)

    if numpy.all(ulp == 0):
        return
    if numpy.all(ulp <= max_valid_ulp_count):
        print(f"maximal ULP difference: {ulp.max()}")
        return

    bsamples = numpy.zeros((im_blocks, re_blocks), dtype=samples.dtype)
    bulp = numpy.zeros((im_blocks, re_blocks), dtype=ulp.dtype)
    for j, blocks in enumerate(numpy.split(ulp, im_blocks, axis=0)):
        for i, block in enumerate(numpy.split(blocks, re_blocks, axis=1)):
            ind = numpy.unravel_index(numpy.argmax(block, axis=None), block.shape)
            j_, i_ = j * im_blocksize + ind[0], i * re_blocksize + ind[1]
            bsamples[j, i] = samples[j_, i_]
            bulp[j, i] = ulp[j_, i_]

    try:
        fa_reference = getattr(fa.utils.numpy_with_algorithms(dtype=dtype), unary_func_name)
    except Exception as msg:
        print(f"disabling functional_algorithms output: {msg}")
        fa_reference = None

    timage = fa.TextImage()

    timage.fill(0, 10, bulp == 0, symbol="=")
    for i in range(1, max_valid_ulp_count + 1):
        timage.fill(0, 10, bulp == i, symbol=str(i))

    timage.fill(0, 10, bulp > max_valid_ulp_count, symbol="!")
    timage.fill(0, 10, bulp > 100 * max_valid_ulp_count, symbol="E")

    real_axis = samples[0:1, ::re_blocksize].real
    imag_axis = samples[::im_blocksize, 0:1].imag
    timage.insert(0, 2, fa.TextImage.fromseq(imag_axis))
    timage.append(-1, 10, fa.TextImage.fromseq(real_axis[:, ::6], mintextwidth=5, maxtextwidth=5))

    print()
    print(timage)
    print()

    if fa_reference is not None:
        rows = [
            (
                "ULP-difference",
                "z",
                f"{backend}:{unary_func_name}(z)",
                f"mpmath:{unary_func_name}(z)",
                f"numpy:{unary_func_name}(z)",
                f"fa:{unary_func_name}(z)",
            )
        ]
    else:
        rows = [
            (
                "ULP-difference",
                "z",
                f"{backend}:{unary_func_name}(z)",
                f"mpmath:{unary_func_name}(z)",
                f"numpy:{unary_func_name}(z)",
            )
        ]

    samples = bsamples
    ulp = bulp

    max_rows = 20
    for value in reversed(sorted(set(ulp.flatten()))):
        if value <= max_valid_ulp_count:
            break
        clusters = fa.utils.Clusters()
        for re, im in zip(*numpy.where(ulp == value)):
            clusters.add((re, im))
        for cluster in clusters:
            re, im = cluster.center_point()
            with warnings.catch_warnings(action="ignore"):
                np_value = npy_reference(samples[re, im])
            r = func(samples[re, im])
            e = reference(samples[re, im])
            if fa_reference is not None:
                fa_value = fa_reference(samples[re, im])
                rows.append((value, samples[re, im], r, e, np_value, fa_value))
            else:
                rows.append((value, samples[re, im], r, e, np_value))
            if len(rows) > max_rows:
                break
        if len(rows) > max_rows:
            rows.append(("...",) * len(rows[0]))
            break
    col_widths = [max(len(str(row[col])) for row in rows) for col in range(len(rows[0]))]
    rows.insert(1, tuple("-" * w for w in col_widths))
    col_fmt = "| " + " | ".join([f"{{{i}:>{w}}}" for i, w in enumerate(col_widths)]) + " |"
    print("\n".join([col_fmt.format(*map(str, row)) for row in rows]))

    pytest.xfail("inaccurate or incorrect results")
