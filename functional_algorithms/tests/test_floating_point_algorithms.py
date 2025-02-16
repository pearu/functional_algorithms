import numpy
import os
import pytest
import itertools
from collections import defaultdict
import warnings

import functional_algorithms as fa
from functional_algorithms import utils
from functional_algorithms import floating_point_algorithms as fpa


@pytest.fixture(scope="function", params=[numpy.float16, numpy.float32, numpy.float64])
def dtype(request):
    return request.param


@pytest.fixture(scope="function", params=["add_2sum", "add_fast2sum", "mul_dekker"])
def binary_op(request):
    return request.param


NumpyContext = utils.NumpyContext


def test_split_veltkamp(dtype):
    ctx = NumpyContext()
    p = utils.get_precision(dtype)
    C = utils.get_veltkamp_splitter_constant(dtype)
    largest = fpa.get_largest(ctx, dtype(0))

    assert isinstance(C, dtype)
    assert C == dtype(2 ** ((p + 1) // 2) + 1)

    fi = numpy.finfo(dtype)
    for f in [
        -150,
        -0.3,
        -fi.eps,
        -fi.smallest_normal,
        -fi.smallest_subnormal * 15,
        -fi.smallest_subnormal,
        0,
        fi.smallest_subnormal,
        fi.smallest_subnormal * 10,
        fi.smallest_normal,
        fi.eps,
        0.3,
        150,
        numpy.pi,
    ]:
        x = dtype(f)
        xh, xl = fpa.split_veltkamp(ctx, x, C)
        assert x == xh + xl
        bh = utils.tobinary(xh).split("p")[0].lstrip("-")
        bl = utils.tobinary(xl).split("p")[0].lstrip("-")
        bh = bh[1 + bh.startswith("1.") :].lstrip("0")
        bl = bl[1 + bl.startswith("1.") :].lstrip("0")
        assert len(bh) < (p + 1) // 2
        assert len(bl) < (p + 1) // 2

    max_x = largest * dtype(1 - 1 / C)
    min_x = -max_x

    size = 1000
    for x in utils.real_samples(size, dtype=dtype, min_value=min_x, max_value=max_x):
        xh, xl = fpa.split_veltkamp(ctx, x, C, scale=True)
        assert x == xh + xl
        bh = utils.tobinary(xh).split("p")[0].lstrip("-")
        bl = utils.tobinary(xl).split("p")[0].lstrip("-")
        bh = bh[1 + bh.startswith("1.") :].lstrip("0")
        bl = bl[1 + bl.startswith("1.") :].lstrip("0")
        assert len(bh) < (p + 1) // 2
        assert len(bl) < (p + 1) // 2


def test_split_veltkamp2(dtype):
    fi = numpy.finfo(dtype)
    ctx = NumpyContext()
    p = utils.get_precision(dtype)
    C = utils.get_veltkamp_splitter_constant(dtype)
    largest = fpa.get_largest(ctx, dtype(0))

    max_x = largest
    min_x = fi.smallest_normal * (C + C - dtype(2))
    size = 1000
    for x in utils.real_samples(size, dtype=dtype, min_value=min_x, max_value=max_x):
        xh, xl = fpa.split_veltkamp2(ctx, x)
        assert x == xh + xl + xh

        x = -x
        xh, xl = fpa.split_veltkamp2(ctx, x)
        assert x == xh + xl + xh


def test_mul_dekker(dtype):
    import mpmath

    max_valid_ulp_count = 2

    if int(os.environ.get("FA_HIGH_RESOLUTION", "0")):
        size = 200
    else:
        size = 20

    p = utils.get_precision(dtype)
    min_x = {11: -986.0, 24: -7.51e33, 53: -4.33e299}[p]
    max_x = {11: 1007.0, 24: 8.3e34, 53: 1.33e300}[p]
    min_xy = {11: -62940.0, 24: -numpy.inf, 53: -numpy.inf}[p]
    max_xy = {11: 62940.0, 24: numpy.inf, 53: numpy.inf}[p]

    fi = numpy.finfo(dtype)
    if 1:
        max_value = numpy.sqrt(fi.max) / dtype(2)
        min_value = -max_value
        min_x = min_value
        max_x = max_value

    ulp_counts = defaultdict(int)
    ulp_counts_native = defaultdict(int)
    C = utils.get_veltkamp_splitter_constant(dtype)
    with mpmath.workprec(utils.vectorize_with_mpmath.float_prec[dtype.__name__] * 2):
        ctx = mpmath.mp
        for x in utils.real_samples(size, dtype=dtype, min_value=min_x, max_value=max_x):
            x_mp = utils.float2mpf(ctx, x)
            for y in utils.real_samples(size, dtype=dtype, min_value=min_x, max_value=max_x):
                xy = x * y
                xyh, xyl = fpa.mul_dekker(None, x, y, C)
                assert numpy.isfinite(xyl)
                assert xyh == xy
                y_mp = utils.float2mpf(ctx, y)
                xyh_mp = utils.float2mpf(ctx, xyh)
                xyl_mp = utils.float2mpf(ctx, xyl)
                expected = x_mp * y_mp
                if abs(xyl) >= fi.smallest_normal:
                    assert expected == xyh_mp + xyl_mp
                else:
                    dif = utils.mpf2float(dtype, x_mp * y_mp - xyh_mp)
                    assert utils.diff_ulp(xyl, dif) <= max_valid_ulp_count

                ex = utils.mpf2float(dtype, expected)
                u = utils.diff_ulp(xyh + xyl, ex)
                ulp_counts[u] += 1
                ulp_counts_native[utils.diff_ulp(xy, ex)] += 1
                assert u <= max_valid_ulp_count

    print(f"\nULP counts using mul_dekker: {dict(ulp_counts)}\nULP counts using x*y: {dict(ulp_counts_native)}")


def test_add_2sum(dtype):
    import mpmath

    max_valid_ulp_count = 1

    # Extra_prec_multiplier does not need to be larger that 5 for
    # float16, 23 for float32, and 84 for float64 (see comment in
    # test_accuracy).
    extra_prec_multiplier = 84
    working_prec = utils.vectorize_with_mpmath.float_prec[dtype.__name__] * extra_prec_multiplier

    if int(os.environ.get("FA_HIGH_RESOLUTION", "0")):
        size = 150
    else:
        size = 20

    fi = numpy.finfo(dtype)
    max_value = fi.max / dtype(2)
    min_value = -max_value

    ulp_counts = defaultdict(int)
    ulp_counts_native = defaultdict(int)
    with mpmath.workprec(working_prec):
        ctx = mpmath.mp
        for x in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
            x_mp = utils.float2mpf(ctx, x)
            for y in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
                xy = x + y
                xyh, xyl = fpa.add_2sum(None, x, y)
                assert xyh == xy

                y_mp = utils.float2mpf(ctx, y)
                expected = x_mp + y_mp
                xyh_mp = utils.float2mpf(ctx, xyh)
                xyl_mp = utils.float2mpf(ctx, xyl)
                assert expected == xyh_mp + xyl_mp

                ex = utils.mpf2float(dtype, expected)
                u = utils.diff_ulp(xyh + xyl, ex)
                ulp_counts[u] += 1
                ulp_counts_native[utils.diff_ulp(xy, ex)] += 1
                assert u <= max_valid_ulp_count

                if abs(x) < abs(y):
                    continue

                xyh, xyl = fpa.add_2sum(None, x, y, fast=True)
                assert xyh == xy

                x_mp = utils.float2mpf(ctx, x)
                y_mp = utils.float2mpf(ctx, y)
                xyh_mp = utils.float2mpf(ctx, xyh)
                xyl_mp = utils.float2mpf(ctx, xyl)
                assert x_mp + y_mp == xyh_mp + xyl_mp, (x_mp + y_mp, xyh_mp + xyl_mp)

    print(f"\nULP counts using add_2sum: {dict(ulp_counts)}\nULP counts using x+y: {dict(ulp_counts_native)}")


def test_accuracy(dtype, binary_op):
    if int(os.environ.get("FA_HIGH_RESOLUTION", "0")):
        blocksize = 10
        blocks = 51
    else:
        blocksize = 2
        blocks = 51

    size = blocks * blocksize
    bw = {numpy.float16: 16, numpy.float32: 32, numpy.float64: 64}[dtype]

    samples1, samples2 = utils.real_pair_samples((size, size), dtype=dtype, include_infinity=False, include_zero=False)

    import mpmath

    if binary_op in {"add_2sum", "add_fast2sum"}:
        # When adding values that exponents are far apart, the
        # addition result may be inexact (say, the largest
        # value). Increasing extra_prec_multiplier will compensate
        # this effect as follows:
        #
        # `(largest + tiny) - largest` is non-zero if the mpmath
        # precision is larger than 30 for float16, 255 for float32,
        # and 2046 for float64.
        #
        # `(largest + tiny) - largest` is exact when the precision is
        # larger that 55 for float16, 540 for float 32, and 4414 for
        # float64.
        #
        # That is, extra_prec_multiplier does not need to be larger
        # that 5 for float16, 23 for float32, and 84 for float64.
        extra_prec_multiplier = 84
    elif binary_op == "mul_dekker":
        extra_prec_multiplier = 2
    else:
        extra_prec_multiplier = 2

    with mpmath.workprec(utils.vectorize_with_mpmath.float_prec[dtype.__name__] * extra_prec_multiplier):
        ctx = mpmath.mp

        if binary_op in {"add_2sum", "add_fast2sum"}:
            expected = [utils.float2mpf(ctx, x) + utils.float2mpf(ctx, y) for x, y in zip(samples1, samples2)]
            with numpy.errstate(over="ignore", invalid="ignore"):
                result = [
                    utils.float2mpf(ctx, hi) + utils.float2mpf(ctx, lo)
                    for hi, lo in map(
                        lambda args: fpa.add_2sum(None, args[0], args[1], fast=binary_op == "add_fast2sum"),
                        zip(samples1, samples2),
                    )
                ]
            with numpy.errstate(over="ignore", invalid="ignore"):
                native = [utils.float2mpf(ctx, x + y) for x, y in zip(samples1, samples2)]
        elif binary_op == "mul_dekker":
            expected = [utils.float2mpf(ctx, x) * utils.float2mpf(ctx, y) for x, y in zip(samples1, samples2)]
            C = utils.get_veltkamp_splitter_constant(dtype)
            with numpy.errstate(over="ignore", invalid="ignore"):
                result = [
                    utils.float2mpf(ctx, hi) + utils.float2mpf(ctx, lo)
                    for hi, lo in map(lambda args: fpa.mul_dekker(None, args[0], args[1], C), zip(samples1, samples2))
                ]
            with numpy.errstate(over="ignore", invalid="ignore"):
                native = [utils.float2mpf(ctx, x * y) for x, y in zip(samples1, samples2)]
        else:
            raise NotImplementedError(binary_op)

        # log2ulp is normalized to range [0, 10]
        ulp_log2_result = numpy.array(
            [utils.diff_log2ulp(utils.mpf2float(dtype, e - r), dtype(0)) * 10 // bw for e, r in zip(expected, result)]
        )
        ulp_log2_native = numpy.array(
            [utils.diff_log2ulp(utils.mpf2float(dtype, e - r), dtype(0)) * 10 // bw for e, r in zip(expected, native)]
        )

    samples1 = samples1.reshape(size, size)
    samples2 = samples2.reshape(size, size)
    ulp_log2_result = ulp_log2_result.reshape(size, size)
    ulp_log2_native = ulp_log2_native.reshape(size, size)

    print()
    for i, u in enumerate(sorted(numpy.unique(ulp_log2_native.flatten()))):
        print(f"native: log2-ULP difference == {u}: {(ulp_log2_native == u).sum()}")
    for i, u in enumerate(sorted(numpy.unique(ulp_log2_result.flatten()))):
        print(f"{binary_op}: log2-ULP difference == {u}: {(ulp_log2_result == u).sum()}")

    bulp_native = numpy.zeros((blocks, blocks), dtype=ulp_log2_native.dtype)
    bulp_result = numpy.zeros((blocks, blocks), dtype=ulp_log2_result.dtype)
    bsamples1 = numpy.zeros((blocks, blocks), dtype=samples1.dtype)
    bsamples2 = numpy.zeros((blocks, blocks), dtype=samples2.dtype)
    for j, blocks_ in enumerate(numpy.split(ulp_log2_native, blocks, axis=0)):
        for i, block in enumerate(numpy.split(blocks_, blocks, axis=1)):
            samples_block1 = samples1[j * blocksize : (j + 1) * blocksize, i * blocksize : (i + 1) * blocksize]
            samples_block2 = samples2[j * blocksize : (j + 1) * blocksize, i * blocksize : (i + 1) * blocksize]
            ind = numpy.unravel_index(numpy.argmax(block, axis=None), block.shape)
            assert block[ind[0], ind[1]] == numpy.max(block)
            bsamples1[j, i] = samples_block1[ind[0], ind[1]]
            bsamples2[j, i] = samples_block2[ind[0], ind[1]]
            bulp_native[j, i] = block[ind[0], ind[1]]

    for j, blocks_ in enumerate(numpy.split(ulp_log2_result, blocks, axis=0)):
        for i, block in enumerate(numpy.split(blocks_, blocks, axis=1)):
            ind = numpy.unravel_index(numpy.argmax(block, axis=None), block.shape)
            assert block[ind[0], ind[1]] == numpy.max(block)
            bulp_result[j, i] = block[ind[0], ind[1]]

    timage = fa.TextImage()

    timage.fill(0, 10, bulp_native == 0, symbol="=")
    for i in range(1, 10):
        timage.fill(0, 10, bulp_native == i, symbol=hex(i)[2:].upper())
    timage.fill(0, 10, bulp_native >= 10, symbol="!")

    timage.fill(0, 16 + blocks, bulp_result == 0, symbol="=")
    for i in range(1, 10):
        timage.fill(0, 16 + blocks, bulp_result == i, symbol=hex(i)[2:].upper())
    timage.fill(0, 16 + blocks, bulp_result >= 10, symbol="!")

    haxis = samples1[0:1, ::blocksize]
    vaxis = samples2[::blocksize, -1:]

    timage.insert(0, 2, fa.TextImage.fromseq(vaxis))
    timage.append(-1, 10, fa.TextImage.fromseq(haxis[:, ::6], mintextwidth=5, maxtextwidth=5))
    timage.insert(-1, 16 + blocks, fa.TextImage.fromseq(haxis[:, ::6], mintextwidth=5, maxtextwidth=5))

    if binary_op in {"add_2sum", "add_fast2sum"}:
        timage.append(-1, 10, "\nnative x + y")
    elif binary_op == "mul":
        timage.append(-1, 10, "\nnative x * y")
    timage.insert(-1, 16 + blocks, f"{binary_op}(x, y)")

    print()
    print(timage)
    print()


def test_is_power_of_two(dtype):
    p = utils.get_precision(dtype)
    Q = dtype(1 << (p - 1))
    P = dtype((1 << (p - 1)) + 1)
    min_e = {11: -24, 24: -149, 53: -1074}[p]
    max_e = {11: 6, 24: 105, 53: 972}[p]

    for e in range(min_e, max_e):
        x = dtype(2**e)
        assert utils.tobinary(x).startswith("1p") or x == 0
        assert fpa.is_power_of_two(None, x, Q, P)
        assert not fpa.is_power_of_two(None, x, Q, P, invert=True)

        x1 = numpy.nextafter(x, dtype(numpy.inf))
        while utils.tobinary(abs(x1)).startswith("1p"):
            x1 = numpy.nextafter(x1, dtype(numpy.inf))
        assert not fpa.is_power_of_two(None, x1, Q, P)
        assert fpa.is_power_of_two(None, x1, Q, P, invert=True)

        x1 = numpy.nextafter(x, dtype(-numpy.inf))
        while utils.tobinary(abs(x1)).startswith("1p") or x1 == 0:
            x1 = numpy.nextafter(x1, dtype(-numpy.inf))
        assert not fpa.is_power_of_two(None, x1, Q, P)
        assert fpa.is_power_of_two(None, x1, Q, P, invert=True)


def test_add_3sum(dtype):
    np_ctx = NumpyContext()
    import mpmath

    max_valid_ulp_count = 1

    # Extra_prec_multiplier does not need to be larger that 5 for
    # float16, 23 for float32, and 84 for float64 (see comment in
    # test_accuracy).
    extra_prec_multiplier = 84
    working_prec = utils.vectorize_with_mpmath.float_prec[dtype.__name__] * extra_prec_multiplier

    if int(os.environ.get("FA_HIGH_RESOLUTION", "0")):
        size = 100
    else:
        size = 20

    fi = numpy.finfo(dtype)
    max_value = fi.max / dtype(4)
    min_value = -max_value

    p = utils.get_precision(dtype)
    Q = dtype(1 << (p - 1))
    P = dtype((1 << (p - 1)) + 1)
    three_over_two = dtype(1.5)

    ulp_counts = defaultdict(int)
    ulp_counts_native = defaultdict(int)
    with mpmath.workprec(working_prec):
        ctx = mpmath.mp
        for x in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
            x_mp = utils.float2mpf(ctx, x)
            for y in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
                xy = x + y
                y_mp = utils.float2mpf(ctx, y)
                xy_mp = x_mp + y_mp
                for z in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
                    xyz = xy + z
                    with numpy.errstate(over="ignore", invalid="ignore"):
                        xyzh, e, t = fpa.add_3sum(np_ctx, x, y, z, Q, P, three_over_two)
                    s = xyzh + (e + t)
                    z_mp = utils.float2mpf(ctx, z)
                    expected = xy_mp + z_mp
                    result = utils.float2mpf(ctx, xyzh) + utils.float2mpf(ctx, e) + utils.float2mpf(ctx, t)
                    assert result == expected
                    ex = utils.mpf2float(dtype, expected)
                    u = utils.diff_ulp(s, ex)
                    ulp_counts[u] += 1
                    ulp_counts_native[utils.diff_ulp(xyz, ex)] += 1
                    assert ex == s or u <= max_valid_ulp_count

    print(f"\nULP counts using add_3sum: {dict(ulp_counts)}\nULP counts using x+y+z: {dict(ulp_counts_native)}")


def test_add_4sum(dtype):
    np_ctx = NumpyContext()
    import mpmath

    max_valid_ulp_count = 1

    # Extra_prec_multiplier does not need to be larger that 5 for
    # float16, 23 for float32, and 84 for float64 (see comment in
    # test_accuracy).
    extra_prec_multiplier = 84
    working_prec = utils.vectorize_with_mpmath.float_prec[dtype.__name__] * extra_prec_multiplier

    if int(os.environ.get("FA_HIGH_RESOLUTION", "0")):
        size = 50
    else:
        size = 15

    fi = numpy.finfo(dtype)
    max_value = fi.max / dtype(4)
    min_value = -max_value

    p = utils.get_precision(dtype)
    Q = dtype(1 << (p - 1))
    P = dtype((1 << (p - 1)) + 1)
    three_over_two = dtype(1.5)

    ulp_counts = defaultdict(int)
    ulp_counts_native = defaultdict(int)
    with mpmath.workprec(working_prec):
        ctx = mpmath.mp
        for x in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
            x_mp = utils.float2mpf(ctx, x)
            for y in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
                xy = x + y
                y_mp = utils.float2mpf(ctx, y)
                xy_mp = x_mp + y_mp
                for z in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
                    xyz = xy + z
                    z_mp = utils.float2mpf(ctx, z)
                    xyz_mp = xy_mp + z_mp
                    for w in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
                        xyzw = xyz + w
                        with numpy.errstate(over="ignore", invalid="ignore"):
                            s = fpa.add_4sum(np_ctx, x, y, z, w, Q, P, three_over_two)

                        w_mp = utils.float2mpf(ctx, w)
                        e = utils.mpf2float(dtype, xyz_mp + w_mp)
                        # add_4sum does not provide error estimates,
                        # so we use ULP difference for testing
                        # closeness:
                        u = utils.diff_ulp(s, e)
                        ulp_counts[u] += 1
                        ulp_counts_native[utils.diff_ulp(xyzw, e)] += 1
                        assert e == s or u <= max_valid_ulp_count

    print(f"\nULP counts using add_4sum: {dict(ulp_counts)}\nULP counts using x+y+z+w: {dict(ulp_counts_native)}")


def test_dot2(dtype):
    np_ctx = NumpyContext()
    import mpmath

    max_valid_ulp_count = 3

    # Extra_prec_multiplier does not need to be larger that 5 for
    # float16, 23 for float32, and 84 for float64 (see comment in
    # test_accuracy).
    extra_prec_multiplier = 84
    working_prec = utils.vectorize_with_mpmath.float_prec[dtype.__name__] * extra_prec_multiplier

    if int(os.environ.get("FA_HIGH_RESOLUTION", "0")):
        size = 50
    else:
        size = 20

    fi = numpy.finfo(dtype)
    max_value = numpy.sqrt(fi.max) / dtype(2)
    min_value = -max_value

    p = utils.get_precision(dtype)
    C = utils.get_veltkamp_splitter_constant(dtype)
    Q = dtype(1 << (p - 1))
    P = dtype((1 << (p - 1)) + 1)
    three_over_two = dtype(1.5)

    ulp_counts = defaultdict(int)
    ulp_counts_native = defaultdict(int)
    with mpmath.workprec(working_prec):
        ctx = mpmath.mp
        for x in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
            x_mp = utils.float2mpf(ctx, x)
            for y in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
                y_mp = utils.float2mpf(ctx, y)
                xy_mp = x_mp * y_mp
                xy = x * y
                for z in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
                    z_mp = utils.float2mpf(ctx, z)
                    for w in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
                        zw = z * w
                        d2 = xy + zw
                        with numpy.errstate(over="ignore", invalid="ignore"):
                            s = fpa.dot2(np_ctx, x, y, z, w, C, Q, P, three_over_two)
                        w_mp = utils.float2mpf(ctx, w)
                        zw_mp = z_mp * w_mp
                        e = utils.mpf2float(dtype, xy_mp + zw_mp)
                        # dot2 does not provide error estimates, so we
                        # use ULP difference for testing closeness:
                        u = utils.diff_ulp(s, e)
                        ulp_counts[u] += 1
                        ulp_counts_native[utils.diff_ulp(d2, e)] += 1
                        assert e == s or u <= max_valid_ulp_count

    print(f"\nULP counts using dot2: {dict(ulp_counts)}\nULP counts using x*y+z*w: {dict(ulp_counts_native)}")


def test_mul_add(dtype):
    np_ctx = NumpyContext()
    import mpmath

    max_valid_ulp_count = 2
    # Extra_prec_multiplier does not need to be larger that 5 for
    # float16, 23 for float32, and 84 for float64 (see comment in
    # test_accuracy).
    extra_prec_multiplier = 84
    working_prec = utils.vectorize_with_mpmath.float_prec[dtype.__name__] * extra_prec_multiplier

    fi = numpy.finfo(dtype)
    max_value = numpy.sqrt(fi.max) / dtype(2)
    min_value = -max_value
    max_value_add = fi.max / dtype(2)
    min_value_add = -max_value_add

    if int(os.environ.get("FA_HIGH_RESOLUTION", "0")):
        size = 100
    else:
        size = 20

    p = utils.get_precision(dtype)
    C = utils.get_veltkamp_splitter_constant(dtype)
    Q = dtype(1 << (p - 1))
    P = dtype((1 << (p - 1)) + 1)
    three_over_two = dtype(1.5)

    ulp_counts = defaultdict(int)
    ulp_counts_native = defaultdict(int)
    with mpmath.workprec(working_prec):
        ctx = mpmath.mp
        for x in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
            x_mp = utils.float2mpf(ctx, x)
            for y in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
                y_mp = utils.float2mpf(ctx, y)
                with numpy.errstate(over="ignore", invalid="ignore"):
                    xy = x * y
                xy_mp = x_mp * y_mp
                for z in utils.real_samples(size, dtype=dtype, min_value=min_value_add, max_value=max_value_add):
                    with numpy.errstate(over="ignore", invalid="ignore"):
                        xyz = xy + z
                    with numpy.errstate(over="ignore", invalid="ignore"):
                        s = fpa.mul_add(np_ctx, x, y, z, C, Q, P, three_over_two)
                    z_mp = utils.float2mpf(ctx, z)
                    expected = xy_mp + z_mp
                    e = utils.mpf2float(dtype, expected)
                    # mul_add does not provide error estimates, so we
                    # use ULP difference for testing closeness:
                    u = utils.diff_ulp(s, e)
                    ulp_counts[u] += 1
                    ulp_counts_native[utils.diff_ulp(xyz, e)] += 1
                    assert e == s or u <= max_valid_ulp_count

    print(f"\nULP counts using mul_add: {dict(ulp_counts)}\nULP counts using x*y+z: {dict(ulp_counts_native)}")


def test_next(dtype):
    ctx = NumpyContext()
    size = 1000
    fi = numpy.finfo(dtype)
    min_value = fi.smallest_normal
    max_value = fi.max

    for x in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
        with warnings.catch_warnings(action="ignore"):
            result = fpa.next(ctx, x, up=True)
            expected = numpy.nextafter(x, dtype(numpy.inf))
        assert result == expected

        with warnings.catch_warnings(action="ignore"):
            result = fpa.next(ctx, -x, up=False)
            expected = numpy.nextafter(-x, dtype(-numpy.inf))
        assert result == expected

        if x > min_value:
            with warnings.catch_warnings(action="ignore"):
                result = fpa.next(ctx, x, up=False)
                expected = numpy.nextafter(x, dtype(-numpy.inf))
            assert result == expected

            with warnings.catch_warnings(action="ignore"):
                result = fpa.next(ctx, -x, up=True)
                expected = numpy.nextafter(-x, dtype(numpy.inf))
            assert result == expected


def test_argument_reduction_exponent(dtype):
    import mpmath

    ctx = NumpyContext()
    size = 10_000_000 // 1000
    fi = numpy.finfo(dtype)
    min_value = fi.smallest_normal
    max_value = numpy.log(fi.max)

    extra_prec_multiplier = 2
    working_prec = utils.vectorize_with_mpmath.float_prec[dtype.__name__] * extra_prec_multiplier

    ln2half = dtype(numpy.log(2) / 2)
    ln2 = dtype(numpy.log(2))

    with mpmath.workprec(working_prec):
        mpctx = mpmath.mp
        for x in utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_value):
            for s in [dtype(1), dtype(-1)]:
                x = s * x
                with warnings.catch_warnings(action="ignore"):
                    k, r, c = fpa.argument_reduction_exponent(ctx, x)

                    assert int(k) == k
                    assert utils.diff_ulp(x, k * ln2 + (r + c)) <= 1
                    # multiplication with 1.1 is required for float16
                    # dtype case due to rounding effects in float16
                    # arithmetics:
                    assert abs(r + c) <= ln2half * dtype(1.1)


def test_split_tripleword(dtype):
    ctx = NumpyContext()
    p = utils.get_precision(dtype)
    min_x = {numpy.float16: -986.0, numpy.float32: -7.51e33, numpy.float64: -4.33e299}[dtype]
    max_x = {numpy.float16: 1007.0, numpy.float32: 8.3e34, numpy.float64: 1.33e300}[dtype]

    largest = fpa.get_largest(ctx, dtype(0))
    C1 = fpa.get_tripleword_splitter_constants(ctx, largest)[0]
    max_x = largest * dtype(1 - 1 / C1)
    min_x = -max_x

    size = 1000
    for x in utils.real_samples(size, dtype=dtype, min_value=min_x, max_value=max_x):
        xh, xl, xr = fpa.split_tripleword(ctx, x, scale=True)
        assert x == xh + xl + xr

        bh = utils.tobinary(xh).split("p")[0].lstrip("-")
        bl = utils.tobinary(xl).split("p")[0].lstrip("-")
        br = utils.tobinary(xr).split("p")[0].lstrip("-")
        bh = bh[1 + bh.startswith("1.") :].lstrip("0")
        bl = bl[1 + bl.startswith("1.") :].lstrip("0")
        br = br[1 + br.startswith("1.") :].lstrip("0")
        ph = len(bh)
        pl = len(bl)
        pr = len(br)
        assert ph + pl + pr <= p


def test_argument_reduction_trigonometric(dtype):
    import mpmath

    print()
    ctx = NumpyContext()
    fi = numpy.finfo(dtype)
    min_value = fi.smallest_normal
    max_value = fi.max

    largest = fpa.get_largest(ctx, dtype(0))
    max_x = largest / dtype(2 ** {numpy.float64: 18, numpy.float32: 5, numpy.float16: 2}[dtype])

    max_prec = {numpy.float16: 24, numpy.float32: 149, numpy.float64: 1074}[dtype]

    if dtype == numpy.float64:
        x = numpy.ldexp(6381956970095103, 797)
        k, r, t = fpa.argument_reduction_trigonometric(ctx, x)  # 4.687165924254629e-19
        # ulp difference from dtype(4.687165924254624e-19) is 5. Is
        # the reference value from
        # https://userpages.cs.umbc.edu/phatak/645/supl/Ng-ArgReduction.pdf
        # wrong?
        u = utils.diff_ulp(r + t, dtype(4.687165924254624e-19))
        assert u <= 5, (r, t, r + t)

    size = 1000
    samples = utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_x)
    ulp_counts = defaultdict(int)
    for x in samples:
        assert isinstance(x, dtype)
        k, r, t = fpa.argument_reduction_trigonometric(ctx, x)

        assert k in [0, 1, 2, 3], x
        assert abs(r) <= dtype(numpy.pi / 4) * 1.1, (r, k, x)
        assert isinstance(r, dtype)

        mp_ctx = mpmath.mp
        with mp_ctx.workprec(max_prec * 10):
            x_mp = utils.float2mpf(mp_ctx, x)
            y_mp = x_mp * (2 / mp_ctx.pi)
            Nk = mp_ctx.floor(y_mp)

            expected_r1 = utils.mpf2float(dtype, (y_mp - Nk) * (mp_ctx.pi / 2))
            expected_k1 = utils.mpf2float(dtype, Nk % 4)

            expected_r2 = utils.mpf2float(dtype, (y_mp - (Nk + 1)) * (mp_ctx.pi / 2))
            expected_k2 = utils.mpf2float(dtype, (Nk + 1) % 4)

            u_r1 = utils.diff_ulp(r + t, expected_r1)
            u_r2 = utils.diff_ulp(r + t, expected_r2)
            u_r = min(u_r1, u_r2)
            if u_r == u_r1:
                expected_r = expected_r1
                expected_k = expected_k1
            else:
                expected_r = expected_r2
                expected_k = expected_k2

            ulp_counts[u_r] += 1

            assert k == expected_k
            if dtype == numpy.float16:
                assert r == expected_r or u_r <= 10
            else:
                assert r == expected_r or u_r <= 1

    for u in sorted(ulp_counts):
        print(f"ULP difference {u}: {ulp_counts[u]}")


def test_sine_pade(dtype):
    import mpmath
    from collections import defaultdict

    t_prec = utils.get_precision(dtype)
    working_prec = {11: 50 * 2, 24: 50 * 2, 53: 74 * 2}[t_prec]
    ctx = NumpyContext()
    size = 10_000
    samples = list(utils.real_samples(size, dtype=dtype, min_value=dtype(0), max_value=dtype(numpy.pi / 4)))
    size = len(samples)
    with mpmath.mp.workprec(working_prec):
        mpctx = mpmath.mp

        for variant in [
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
        ]:
            ulp = defaultdict(int)
            for x in samples:
                expected_sn = utils.mpf2float(dtype, mpctx.sin(utils.float2mpf(mpctx, x)))
                sn = fpa.sine_pade(ctx, x, variant=variant)
                u = utils.diff_ulp(sn, expected_sn)
                ulp[u] += 1

            print(f"{variant=}")
            rest = 0
            u5 = None
            for i, u in enumerate(sorted(ulp)):
                if i < 5:
                    print(f"  ULP difference {u}: {ulp[u]}")
                else:
                    if u5 is None:
                        u5 = u
                    rest += ulp[u]
            else:
                if rest:
                    print(f"  ULP difference >= {u5}: {rest}")


def test_sine_taylor(dtype):
    import mpmath
    from collections import defaultdict

    t_prec = utils.get_precision(dtype)
    working_prec = {11: 50 * 2, 24: 50 * 2, 53: 74 * 2}[t_prec]
    optimal_order = {11: 7, 24: 9, 53: 15}[t_prec]
    ctx = NumpyContext()
    size = 1000
    samples = list(utils.real_samples(size, dtype=dtype, min_value=dtype(0), max_value=dtype(numpy.pi / 4)))
    size = len(samples)
    with mpmath.mp.workprec(working_prec):
        mpctx = mpmath.mp
        for order in [optimal_order, 1, 3, 5, 7, 9, 11, 13, 17, 19][:1]:
            ulp = defaultdict(int)
            for x in samples:
                expected_sn = utils.mpf2float(dtype, mpctx.sin(utils.float2mpf(mpctx, x)))
                sn = fpa.sine_taylor(ctx, x, order=order, split=False)
                snh, snl = fpa.sine_taylor(ctx, x, order=order, split=True)
                sn2 = utils.mpf2float(dtype, utils.float2mpf(mpctx, snh) + utils.float2mpf(mpctx, snl))
                assert sn == sn2
                # sn = numpy.sin(x)
                u = utils.diff_ulp(sn, expected_sn)
                ulp[u] += 1

            rest = 0
            u5 = None
            for i, u in enumerate(sorted(ulp)):
                if i < 5:
                    print(f"  ULP difference {u}: {ulp[u]}")
                else:
                    if u5 is None:
                        u5 = u
                    rest += ulp[u]
            else:
                if rest:
                    print(f"  ULP difference >= {u5}: {rest}")
