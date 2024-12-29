import numpy
import functional_algorithms as fa
import os
import pytest
import warnings
import contextlib


@pytest.fixture(scope="function", params=["jax", "functional_algorithms"])
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


if fa.fpu.MXCSRRegister.is_available():
    fpu_params = ["default", "enable-FZ", "enable-DAZ"]
else:
    fpu_params = ["default"]


@pytest.fixture(scope="function", params=fpu_params)
def fpu(request):
    return request.param


def test_unary(unary_func_name, backend, device, dtype, fpu):
    if backend == "functional_algorithms":
        backend = "algorithms"
    else:
        pytest.importorskip(backend)

    register_params = dict()
    register = lambda *args, **kwargs: contextlib.nullcontext()
    if fpu != "default":
        if backend not in {"algorithms"} or device != "cpu":
            pytest.skip(f"{unary_func_name}: fpu {fpu} mode N/A for {backend=} {device=}")
        register = fa.fpu.context
        if "enable-FZ" in fpu:
            register_params.update(FZ=True)
        if "disable-FZ" in fpu:
            register_params.update(FZ=False)
        if "enable-DAZ" in fpu:
            register_params.update(DAZ=True)
        if "disable-DAZ" in fpu:
            register_params.update(DAZ=False)
    else:
        if device == "cpu":
            register = fa.fpu.context
            register_params.update(RN="nearest")

    numpy_with_backend = getattr(fa.utils, f"numpy_with_{backend}")(device=device, dtype=dtype)
    try:
        func = getattr(numpy_with_backend, unary_func_name)
    except NotImplementedError as msg:
        pytest.skip(f"{unary_func_name}: {msg}")
    except AttributeError as msg:
        pytest.skip(f"{unary_func_name}: {msg}")

    if not func.backend_is_available(device):
        pytest.skip(f"{device} support is unavailable")
    params = fa.utils.function_validation_parameters(unary_func_name, dtype, device=device)

    max_valid_ulp_count = params["max_valid_ulp_count"]
    extra_prec_multiplier = params["extra_prec_multiplier"]
    samples_limits = params["samples_limits"]

    # detect the FTZ mode of array backend: if FTZ is enabled, don't
    # generate samples containing subnormals as well as exclude
    # subnormals in comparing results (read: ulp distance between 0
    # and smallest normal is defined as 1).
    fi = numpy.finfo(dtype)
    x = numpy.sqrt(fi.smallest_normal) * dtype(0.5)
    with register(**register_params):
        v1 = getattr(numpy_with_backend, "square")(x)
    v2 = numpy.square(x)
    d = fa.utils.diff_ulp(v1, v2)
    if d > 1000:
        include_subnormal = False
    else:
        include_subnormal = True

    mpmath = fa.utils.numpy_with_mpmath(extra_prec_multiplier=extra_prec_multiplier, flush_subnormals=not include_subnormal)

    reference = getattr(mpmath, unary_func_name)
    npy_reference = getattr(fa.utils.numpy_with_numpy(), unary_func_name)

    complex_plane = False
    real_line = False

    if dtype in {numpy.complex64, numpy.complex128}:
        complex_plane = True
        if int(os.environ.get("FA_HIGH_RESOLUTION", "0")):
            re_blocks, im_blocks = 101, 52
            re_blocksize, im_blocksize = 20, 20
        else:
            re_blocks, im_blocks = 51, 26
            re_blocksize, im_blocksize = 5, 5

        re_size, im_size = re_blocks * re_blocksize, im_blocks * im_blocksize

        samples = fa.utils.complex_samples(
            (re_size, im_size), dtype=dtype, include_subnormal=include_subnormal, **samples_limits
        )

        assert samples.shape == (im_size, re_size)
    elif dtype in {numpy.float32, numpy.float64}:
        real_line = True
        if int(os.environ.get("FA_HIGH_RESOLUTION", "0")):
            rows = 52
            blocks = 101
            blocksize = 400
        else:
            rows = 26
            blocks = 51
            blocksize = 5

        samples = fa.utils.real_samples(
            rows * blocks * blocksize,
            dtype=dtype,
            include_subnormal=include_subnormal,
            include_zero=False,
        )
        assert samples.size == rows * blocks * blocksize, (samples.size, rows * blocks * blocksize)
        samples = samples.reshape(rows, blocks * blocksize)
    else:
        pytest.skip(f"{dtype} support not implemented")

    expected = reference.call(samples)

    if backend == "jax" and device == "cpu" and include_subnormal:
        # XLA CPU client enables FTZ to follow TF convention. To
        # disable FTZ, replace all occurances of
        #   tsl::port::ScopedFlushDenormal flush;
        # with
        #   tsl::port::ScopedDontFlushDenormal flush;
        # in xla/pjrt/cpu/cpu_client.cc.
        #
        # However, disabling FTZ in XLA CPU client is effective only for the first
        # part of samples evaluations. To workaround this, we'll evaluate
        # JAX functions blockwise:
        if complex_plane:
            eval_blocksize = im_size
            eval_size = re_size
        elif real_line:
            eval_blocksize = rows
            eval_size = blocks * blocksize
        else:
            assert 0, (complex_plane, real_line)  # unreachable
        orig_eval_blocksize = eval_blocksize
        assert eval_size < 2**14
        while eval_blocksize * eval_size > 2**14:
            for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
                if eval_blocksize % p == 0:
                    eval_blocksize //= p
                    break
            else:
                assert 0  # adjust re/im_size/blocksize parameters to avoid this
        assert orig_eval_blocksize % eval_blocksize == 0, (orig_eval_blocksize, eval_blocksize)
        result = numpy.concatenate(
            tuple(
                func(samples[k * eval_blocksize : (k + 1) * eval_blocksize])
                for k in range(orig_eval_blocksize // eval_blocksize)
            )
        )
    else:
        with register(**register_params):
            result = func(samples)

    if 0 and complex_plane:
        # for sanity check
        for j in range(im_size):
            for i in range(re_size):
                r = func(samples[j, i])[()]
                assert numpy.array_equal(r, result[j, i], equal_nan=True), (
                    (j, i),
                    samples[j, i],
                    r,
                    result[j, i],
                )

    ulp = fa.utils.diff_ulp(result, expected, flush_subnormals=not include_subnormal, equal_nan=True)

    if numpy.all(ulp == 0):
        return
    if numpy.all(ulp <= max_valid_ulp_count):
        print(f"maximal ULP difference: {ulp.max()}")
        return

    if complex_plane:
        bsamples = numpy.zeros((im_blocks, re_blocks), dtype=samples.dtype)
        bulp = numpy.zeros((im_blocks, re_blocks), dtype=ulp.dtype)
        for j, blocks in enumerate(numpy.split(ulp, im_blocks, axis=0)):
            for i, block in enumerate(numpy.split(blocks, re_blocks, axis=1)):
                samples_block = samples[j * im_blocksize : (j + 1) * im_blocksize, i * re_blocksize : (i + 1) * re_blocksize]
                ind = numpy.unravel_index(numpy.argmax(block, axis=None), block.shape)
                assert block[ind[0], ind[1]] == numpy.max(block)
                bsamples[j, i] = samples_block[ind[0], ind[1]]
                bulp[j, i] = block[ind[0], ind[1]]
    elif real_line:
        bsamples = numpy.zeros((rows, blocks), dtype=samples.dtype)
        bulp = numpy.zeros((rows, blocks), dtype=ulp.dtype)
        for j, blocks_ in enumerate(numpy.split(ulp, rows, axis=0)):
            for i, block in enumerate(numpy.split(blocks_, blocks, axis=1)):
                samples_block = samples[j : j + 1, i * blocksize : (i + 1) * blocksize]
                ind = numpy.unravel_index(numpy.argmax(block, axis=None), block.shape)
                assert block[ind[0], ind[1]] == numpy.max(block)
                bsamples[j, i] = samples_block[ind[0], ind[1]]
                bulp[j, i] = block[ind[0], ind[1]]
    else:
        assert 0  # unreachable
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

    if complex_plane:
        real_axis = samples[0:1, ::re_blocksize].real
        imag_axis = samples[::im_blocksize, 0:1].imag
        timage.insert(0, 2, fa.TextImage.fromseq(imag_axis))
        timage.append(-1, 10, fa.TextImage.fromseq(real_axis[:, ::6], mintextwidth=5, maxtextwidth=5))
    elif real_line:
        laxis = samples[:, 0:1]
        raxis = samples[:, -1:]
        timage.insert(0, 2, fa.TextImage.fromseq(laxis))
        timage.insert(0, blocks + 10 + 5, fa.TextImage.fromseq(raxis), loc="ul")
    else:
        assert 0  # unreachable

    print()
    print(timage)
    print()

    z = "z" if complex_plane else "x"
    if fa_reference is not None and backend != "algorithms":
        table = [
            (
                "ULP-difference",
                z,
                f"{backend}:{unary_func_name}({z})",
                f"mpmath:{unary_func_name}({z})",
                f"numpy:{unary_func_name}({z})",
                f"fa:{unary_func_name}({z})",
            )
        ]
    else:
        table = [
            (
                "ULP-difference",
                z,
                f"{backend}:{unary_func_name}({z})",
                f"mpmath:{unary_func_name}({z})",
                f"numpy:{unary_func_name}({z})",
            )
        ]

    samples = bsamples
    ulp = bulp

    if complex_plane:
        max_rows = 20
        for value in reversed(sorted(set(ulp.flatten()))):
            if value <= max_valid_ulp_count:
                break
            clusters = fa.utils.Clusters()
            for re, im in zip(*numpy.where(ulp == value)):
                clusters.add((re, im))
            for cluster in clusters.clusters + clusters.split().clusters:
                re, im = cluster.center_point()
                with warnings.catch_warnings(action="ignore"):
                    np_value = npy_reference(samples[re, im])
                with register(**register_params):
                    r = func(samples[re, im])
                e = reference(samples[re, im])
                u = fa.utils.diff_ulp(r, e, flush_subnormals=not include_subnormal, equal_nan=True)
                assert u == value, (u, value, (re, im))
                if fa_reference is not None:
                    fa_value = fa_reference(samples[re, im])
                    table.append((value, samples[re, im], r, e, np_value, fa_value))
                else:
                    table.append((value, samples[re, im], r, e, np_value))

        if len(table) > max_rows:
            table = table[: max_rows // 2] + [(f"...",) * len(table[0])] + table[-max_rows // 2 :]
        col_widths = [max(len(str(row[col])) for row in table) for col in range(len(table[0]))]
        table.insert(1, tuple("-" * w for w in col_widths))
        col_fmt = "| " + " | ".join([f"{{{i}:>{w}}}" for i, w in enumerate(col_widths)]) + " |"
        print("\n".join([col_fmt.format(*map(str, row)) for row in table]))
    elif real_line:
        max_rows = 20
        for value in reversed(sorted(set(ulp.flatten()))):
            if value <= 0:  # max_valid_ulp_count:
                break
            clusters = fa.utils.Clusters()
            for r, s in zip(*numpy.where(ulp == value)):
                clusters.add((r * 10, s))
            for cluster in clusters.clusters:
                r, s = cluster.center_point()
                r //= 10
                with warnings.catch_warnings(action="ignore"):
                    np_value = npy_reference(samples[r, s])
                with register(**register_params):
                    res = func(samples[r, s])
                e = reference(samples[r, s])
                u = fa.utils.diff_ulp(res, e, flush_subnormals=not include_subnormal, equal_nan=True)
                assert u == value, (u, value, (r, s))
                if fa_reference is not None:
                    fa_value = fa_reference(samples[r, s])
                    table.append((value, samples[r, s], res, e, np_value, fa_value))
                else:
                    table.append((value, samples[r, s], res, e, np_value))

        if len(table) > max_rows:
            table = (
                table[: max_rows // 2]
                + [(f"<snip {len(table) - max_rows} rows>",) + ("...",) * (len(table[0]) - 1)]
                + table[-max_rows // 2 :]
            )
        col_widths = [max(len(str(row[col])) for row in table) for col in range(len(table[0]))]
        table.insert(1, tuple("-" * w for w in col_widths))
        col_fmt = "| " + " | ".join([f"{{{i}:>{w}}}" for i, w in enumerate(col_widths)]) + " |"
        print("\n".join([col_fmt.format(*map(str, row)) for row in table]))
    else:
        assert 0  # unreachable

    if fpu == "default":
        pytest.xfail("inaccurate or incorrect results")
    # otherwise the inaccurate results are expected due to FTZ or DAZ mode
