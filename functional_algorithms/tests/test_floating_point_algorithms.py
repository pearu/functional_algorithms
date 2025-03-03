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


def show_ulp(ulp, title=None):
    rest = 0
    u5 = None
    if ulp and title is not None:
        print(f"{title}:")
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
                xyh, xyl = fpa.mul_dekker(None, x, y, C)
                assert numpy.isfinite(xyl)
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
                ulp_counts_native[utils.diff_ulp(x * y, ex)] += 1
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
    min_value = dtype(0.5)  # argument reduction is used only for value larger that pi / 4
    max_value = fi.max

    largest = fpa.get_largest(ctx, dtype(0))
    max_x = largest / dtype(2 ** {numpy.float64: 18, numpy.float32: 5, numpy.float16: 2}[dtype])

    max_prec = {numpy.float16: 24, numpy.float32: 149, numpy.float64: 1074}[dtype]

    size = 1000
    samples = utils.real_samples(size, dtype=dtype, min_value=min_value, max_value=max_x)
    ulp_counts = defaultdict(int)
    mp_ctx = mpmath.mp
    with mp_ctx.workprec(max_prec):
        # multiprecision value of 2 / pi and pi / 2
        two_over_pi_mp = 2 / mp_ctx.pi
        pi_over_two_mp = mp_ctx.pi / 2

        # multiword represention of 2 / pi
        two_over_pi_max_length = {numpy.float16: 4, numpy.float32: 11, numpy.float64: 39}.get(dtype, None)
        two_over_pi_mw = utils.get_two_over_pi_multiword(dtype, max_length=two_over_pi_max_length)
        print(f"{two_over_pi_mw=}")
        two_over_pi_mw_mp = None
        for v in reversed(two_over_pi_mw):
            if two_over_pi_max_length is None:
                assert abs(fpa.split_veltkamp(ctx, v)[1]) == 0
            elif two_over_pi_mw_mp is not None:
                assert abs(fpa.split_veltkamp(ctx, v)[1]) == 0
            v_mp = utils.float2mpf(mp_ctx, v)
            if two_over_pi_mw_mp is None:
                two_over_pi_mw_mp = v_mp
            else:
                two_over_pi_mw_mp += v_mp

        if two_over_pi_max_length is None:
            assert two_over_pi_mw_mp == two_over_pi_mp

        # pi / 2 = pi2h + pi2l
        pi2h = utils.mpf2float(dtype, pi_over_two_mp)
        pi2l = utils.mpf2float(dtype, pi_over_two_mp - utils.float2mpf(mp_ctx, pi2h))

        ulp_stage1 = defaultdict(int)
        ulp_stage2 = defaultdict(int)
        four = dtype(4)
        two = dtype(2)
        zero = dtype(0)
        cumerr = defaultdict(dtype)
        for x in samples:
            x_mp = utils.float2mpf(mp_ctx, x)

            # multiprecision value of x * (2 / pi)
            with mp_ctx.workprec(max_prec * 2):
                x2pi_mp = x_mp * two_over_pi_mp
                x2pi_mp_m4 = x2pi_mp - mp_ctx.floor((x2pi_mp + 2) / 4) * 4
                assert abs(x2pi_mp_m4) <= 2

            # split x into xh + xl + xh, valid for any x larger than 1/2
            xh, xl = fpa.split_veltkamp2(ctx, x)
            assert xh + xl + xh == x
            assert fpa.split_veltkamp(ctx, xh)[1] == 0
            assert fpa.split_veltkamp(ctx, xl)[1] == 0

            # multiword representation of x * (2 / pi)
            r_mw_h = [v * xh for v in two_over_pi_mw]
            r_mw_l = [v * xl for v in two_over_pi_mw]

            """Next. we'll reduce x * (2 / pi) by subtracting 4 * N, N is integer.

            There exists several ways to reduce a value by 4 * N that
            involve using functions like truncate, round, floor, ceil,
            remainder, fmod, rint, %, etc.

            As a result of the following tests, the cumulative error
            of these approches is as follows:

            float16:
            fmod      : 4.172325134277344e-06
            rint      : 4.172325134277344e-06
            round     : 4.172325134277344e-06
            truncate  : 4.172325134277344e-06
            ceil      : 0.4892578125
            %         : 0.53466796875
            floor     : 0.53466796875
            remainder : 0.53466796875

            float32:
            fmod      : 3.2229864679470793e-44
            rint      : 3.2229864679470793e-44
            round     : 3.2229864679470793e-44
            truncate  : 3.2229864679470793e-44
            ceil      : 6.776367808924988e-05
            %         : 7.783176988596097e-05
            floor     : 7.783176988596097e-05
            remainder : 7.783176988596097e-05

            float64:
            fmod      : 2.5e-323
            rint      : 2.5e-323
            round     : 2.5e-323
            truncate  : 2.5e-323
            ceil      : 1.20780455651524e-13
            %         : 1.3584433071495718e-13
            floor     : 1.3584433071495718e-13
            remainder : 1.3584433071495718e-13

            StableHLO provides the following functions: ceil, floor,
            remainder, round_nearest_afz, round_nearest_even.

            In the following, we'll use round-nearest-even approach for reducing by 4*N.
            """
            if 0:
                # (2 * xh) - truncate((2 * xh) / 4) * 4 = (xh - truncate(xh / 2) * 2) * 2
                r_mw_h_m4_t = [v - ctx.truncate(v / two) * two for v in r_mw_h]
                r_mw_l_m4_t = [v - ctx.truncate(v / four) * four for v in r_mw_l]

                r_mw_h_m4_r = [v - ctx.round(v / two) * two for v in r_mw_h]
                r_mw_l_m4_r = [v - ctx.round(v / four) * four for v in r_mw_l]

                r_mw_h_m4_f = [v - ctx.floor(v / two) * two for v in r_mw_h]
                r_mw_l_m4_f = [v - ctx.floor(v / four) * four for v in r_mw_l]

                r_mw_h_m4_c = [v - ctx.ceil(v / two) * two for v in r_mw_h]
                r_mw_l_m4_c = [v - ctx.ceil(v / four) * four for v in r_mw_l]

                r_mw_h_m4_rem = [ctx.remainder(v, two) for v in r_mw_h]
                r_mw_l_m4_rem = [ctx.remainder(v, four) for v in r_mw_l]

                r_mw_h_m4_mod = [v % two for v in r_mw_h]
                r_mw_l_m4_mod = [v % four for v in r_mw_l]

                r_mw_h_m4_fm = [ctx.fmod(v, two) for v in r_mw_h]
                r_mw_l_m4_fm = [ctx.fmod(v, four) for v in r_mw_l]

                r_mw_h_m4_rint = [v - ctx.rint(v / two) * two for v in r_mw_h]
                r_mw_l_m4_rint = [v - ctx.rint(v / four) * four for v in r_mw_l]

                def mp_m4(v):
                    if v >= 4 or v < 0:
                        r = v - mp_ctx.floor(v / 4) * 4
                    else:
                        r = v
                    assert r >= 0
                    assert r < 4
                    return r

                with mp_ctx.workprec(max_prec * 2):
                    r_mw_mp = utils.multiword2mpf(mp_ctx, r_mw_h) * 2 + utils.multiword2mpf(mp_ctx, r_mw_l)
                    r_mw_mp_m4_t = utils.multiword2mpf(mp_ctx, r_mw_h_m4_t) * 2 + utils.multiword2mpf(mp_ctx, r_mw_l_m4_t)
                    r_mw_mp_m4_r = utils.multiword2mpf(mp_ctx, r_mw_h_m4_r) * 2 + utils.multiword2mpf(mp_ctx, r_mw_l_m4_r)
                    r_mw_mp_m4_mod = utils.multiword2mpf(mp_ctx, r_mw_h_m4_mod) * 2 + utils.multiword2mpf(
                        mp_ctx, r_mw_l_m4_mod
                    )
                    r_mw_mp_m4_f = utils.multiword2mpf(mp_ctx, r_mw_h_m4_f) * 2 + utils.multiword2mpf(mp_ctx, r_mw_l_m4_f)
                    r_mw_mp_m4_c = utils.multiword2mpf(mp_ctx, r_mw_h_m4_c) * 2 + utils.multiword2mpf(mp_ctx, r_mw_l_m4_c)
                    r_mw_mp_m4_rem = utils.multiword2mpf(mp_ctx, r_mw_h_m4_rem) * 2 + utils.multiword2mpf(
                        mp_ctx, r_mw_l_m4_rem
                    )
                    r_mw_mp_m4_fm = utils.multiword2mpf(mp_ctx, r_mw_h_m4_fm) * 2 + utils.multiword2mpf(mp_ctx, r_mw_l_m4_fm)
                    r_mw_mp_m4_rint = utils.multiword2mpf(mp_ctx, r_mw_h_m4_rint) * 2 + utils.multiword2mpf(
                        mp_ctx, r_mw_l_m4_rint
                    )

                    r_mw_mp_m4_t = mp_m4(r_mw_mp_m4_t)
                    r_mw_mp_m4_r = mp_m4(r_mw_mp_m4_r)
                    r_mw_mp_m4_mod = mp_m4(r_mw_mp_m4_mod)
                    r_mw_mp_m4_f = mp_m4(r_mw_mp_m4_f)
                    r_mw_mp_m4_c = mp_m4(r_mw_mp_m4_c)
                    r_mw_mp_m4_rem = mp_m4(r_mw_mp_m4_rem)
                    r_mw_mp_m4_fm = mp_m4(r_mw_mp_m4_fm)
                    r_mw_mp_m4_rint = mp_m4(r_mw_mp_m4_rint)

                assert utils.mpf2float(dtype, x2pi_mp) == utils.mpf2float(dtype, r_mw_mp)

                cumerr["truncate"] += utils.mpf2float(dtype, abs(r_mw_mp_m4_t - x2pi_mp_m4))
                cumerr["round"] += utils.mpf2float(dtype, abs(r_mw_mp_m4_r - x2pi_mp_m4))
                cumerr["%"] += utils.mpf2float(dtype, abs(r_mw_mp_m4_mod - x2pi_mp_m4))
                cumerr["floor"] += utils.mpf2float(dtype, abs(r_mw_mp_m4_f - x2pi_mp_m4))
                cumerr["ceil"] += utils.mpf2float(dtype, abs(r_mw_mp_m4_c - x2pi_mp_m4))
                cumerr["remainder"] += utils.mpf2float(dtype, abs(r_mw_mp_m4_rem - x2pi_mp_m4))
                cumerr["fmod"] += utils.mpf2float(dtype, abs(r_mw_mp_m4_fm - x2pi_mp_m4))
                cumerr["rint"] += utils.mpf2float(dtype, abs(r_mw_mp_m4_rint - x2pi_mp_m4))

            r_mw_h_m4_r = [(v - ctx.round(v / two) * two) for v in r_mw_h]
            r_mw_l_m4_r = [v - ctx.round(v / four) * four for v in r_mw_l]
            with mp_ctx.workprec(max_prec * 2):
                r_mw_mp_m4_r = utils.multiword2mpf(mp_ctx, r_mw_h_m4_r) * 2 + utils.multiword2mpf(mp_ctx, r_mw_l_m4_r)
                r_mw_mp_m4_r = r_mw_mp_m4_r - mp_ctx.floor(r_mw_mp_m4_r / 4) * 4

            if two_over_pi_max_length is None:
                assert utils.mpf2float(dtype, r_mw_mp_m4_r) == utils.mpf2float(dtype, x2pi_mp_m4)

            y = None
            t = None
            for zh, zl in zip(reversed(r_mw_h_m4_r), reversed(r_mw_l_m4_r)):
                if y is None:
                    y, t = fpa.add_2sum(ctx, zh + zh, zl)
                else:
                    y, th = fpa.add_2sum(ctx, y, zh + zh)
                    y, tl = fpa.add_2sum(ctx, y, zl)
                    t += th + tl

            # compute `(y + t) - round((y + t) / 4) * 4 -> y' + t`
            # such that -2.0 <= y' + t <= 2.0
            n1 = ctx.round(y / four) * four
            y1 = y - n1
            y = ctx.select(y1 + two < -t, y - (n1 - four), ctx.select(y1 - two > -t, y - (n1 + four), y1))
            assert abs(y + t) <= two, (x, y, t)

            if 1:
                y2, t2 = fpa.argument_reduction_trigonometric_stage1_impl(ctx, dtype, x)
                assert y2 == y
                assert t2 == t

            with mp_ctx.workprec(max_prec * 2):
                y_mp = utils.float2mpf(mp_ctx, y) + utils.float2mpf(mp_ctx, t)

            u = utils.diff_ulp(utils.mpf2float(dtype, y_mp), utils.mpf2float(dtype, x2pi_mp_m4))
            ulp_stage1[u] += 1
            if u > 0:
                print(f"1: {x=} {y, t=}")
                print(
                    f"diff(result={str(utils.mpf2float(dtype, y_mp))}, expected={str(utils.mpf2float(dtype, x2pi_mp_m4))}) -> {u}"
                )

            k = ctx.round(y)
            yk = y - k

            if 1:
                assert k in [-2, -1, 0, 1, 2], (y, k, y - k, t, (y2, t2))

            """
            (yk + t) * (pi2h + pi2l) = yk * pi2h + yk * pi2l + t * pi2h + t * pi2l
            """
            r1h, r1l = fpa.mul_dekker(ctx, yk, pi2h)
            r2h, r2l = fpa.mul_dekker(ctx, yk, pi2l)
            r3h, r3l = fpa.mul_dekker(ctx, t, pi2h)
            r4 = t * pi2l

            r, rrl1 = fpa.add_2sum(ctx, r1h, r2h, fast=True)
            r, rrl2 = fpa.add_2sum(ctx, r, r3h, fast=True)
            r, rrl3 = fpa.add_2sum(ctx, r, r4, fast=True)
            s = rrl3 + r3l + rrl2 + r2l + rrl1 + r1l

            if 1:
                k2, r2, s2 = fpa.argument_reduction_trigonometric_stage2_impl(ctx, dtype, y, t)

                assert k == k2, ((k, r, s), (k2, r2, s2))
                assert r == r2
                assert s == s2

            with mp_ctx.workprec(max_prec * 2):
                if 0:
                    rx_mp = (
                        utils.float2mpf(mp_ctx, k) * pi_over_two_mp
                        + (utils.float2mpf(mp_ctx, y - k) + utils.float2mpf(mp_ctx, t)) * pi_over_two_mp
                    )
                else:
                    rx_mp = (
                        utils.float2mpf(mp_ctx, r) + utils.float2mpf(mp_ctx, s) + utils.float2mpf(mp_ctx, k) * pi_over_two_mp
                    )
                rx_mp_mp = x2pi_mp_m4 * pi_over_two_mp

                rrrt_mp = utils.float2mpf(mp_ctx, r) + utils.float2mpf(mp_ctx, s)
                ub = utils.mpf2float(dtype, pi_over_two_mp / 2)
                for i in range(5):
                    ub = numpy.nextafter(ub, dtype(1000))
                assert utils.mpf2float(dtype, abs(rrrt_mp)) <= ub

                u = utils.diff_ulp(utils.mpf2float(dtype, rx_mp), utils.mpf2float(dtype, rx_mp_mp))
            ulp_stage2[u] += 1
            if u > 0:
                print(f"2: {x=} {y, t=} {k, r, s=} {u=}")
                print(f"{rrl3, r3l, rrl2, r2l, rrl1, r1l=}")
        show_ulp(ulp_stage1, title="Stage1[x * (2 / pi) mod 4 -> y + t]")
        show_ulp(ulp_stage2, title="Stage2[(y + t) * (pi / 2) -> k * pi / 2 + r + s]")

        for e, m in sorted((e, m) for (m, e) in cumerr.items()):
            print(f"{m:10}: {e}")


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
            show_ulp(ulp)


@pytest.mark.parametrize(
    "func,fma", [("sin", "upcast"), ("sin", "mul_add"), ("sin_dekker", "native"), ("sin", "native"), ("numpy.sin", None)]
)
def test_sine_taylor(dtype, func, fma):
    import mpmath
    from collections import defaultdict

    t_prec = utils.get_precision(dtype)
    working_prec = {11: 50 * 4, 24: 50 * 4, 53: 74 * 16}[t_prec]
    optimal_order = {11: 7, 24: 9, 53: 15}[t_prec]
    size = 1000
    samples = list(utils.real_samples(size, dtype=dtype, min_value=dtype(0), max_value=dtype(numpy.pi / 4)))
    size = len(samples)
    with mpmath.mp.workprec(working_prec):
        mpctx = mpmath.mp
        for order in [optimal_order, 1, 3, 5, 7, 9, 11, 13, 17, 19][:1]:

            if func == "sin_dekker":

                @fa.targets.numpy.jit(
                    paths=[fpa],
                    dtype=dtype,
                    debug=(1.5 if size <= 10 else 0),
                )
                def sin_dekker_func(ctx, x):
                    with ctx.parameters(series_uses_dekker=True, series_uses_2sum=True):
                        return fpa.sine_taylor_dekker(ctx, x, order=order)

            elif func == "sin":

                @fa.targets.numpy.jit(
                    paths=[fpa],
                    dtype=dtype,
                    debug=(1.5 if size <= 10 else 0),
                    rewrite_parameters=dict(optimize_cast=False, fma_backend=fma),
                )
                def sin_func(ctx, x):
                    with ctx.parameters(series_uses_2sum=True):
                        return fpa.sine_taylor(ctx, x, order=order, split=False)

            ulp = defaultdict(int)
            for x in samples:
                expected_sn = utils.mpf2float(dtype, mpctx.sin(utils.float2mpf(mpctx, x)))
                if func == "numpy.sin":
                    sn = numpy.sin(x)
                elif func == "sin":
                    sn = sin_func(x)
                elif func == "sin_dekker":
                    sn = sin_dekker_func(x)
                else:
                    assert 0, func  # not implemented
                    """
                    sn = fpa.sine_taylor(ctx, x, order=order, split=False)
                    snh, snl = fpa.sine_taylor(ctx, x, order=order, split=True)
                    sn2 = utils.mpf2float(dtype, utils.float2mpf(mpctx, snh) + utils.float2mpf(mpctx, snl))
                    assert sn == sn2
                    """
                u = utils.diff_ulp(sn, expected_sn)
                ulp[u] += 1

            show_ulp(ulp)


def test_sine_taylor_dekker(dtype):
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
                sn = fpa.sine_taylor_dekker(ctx, x, order=order)
                if type(sn) is tuple:
                    sn = sum(reversed(sn[1:-1]), sn[-1])
                u = utils.diff_ulp(sn, expected_sn)
                ulp[u] += 1

                if 0:
                    c = "." if u == 0 else ("v" if sn < expected_sn else "^")
                    print(c, end="")

            show_ulp(ulp)


@pytest.mark.parametrize(
    "func,fma",
    [
        ("cos", "upcast"),
        ("cos", "mul_add"),
        ("cos", "native"),
        ("cos_dekker", "native"),
        ("cos_numpy", None),
        ("cosm1_dekker", "native"),
        ("cosm1", "upcast"),
        ("cosm1_sin", "upcast"),
        ("cosm1_sin_numpy", None),
        ("cosm1_numpy", None),
    ],
)
def test_cosine_taylor(dtype, fma, func):
    import mpmath
    from collections import defaultdict

    t_prec = utils.get_precision(dtype)
    working_prec = {11: 50 * 4, 24: 50 * 4, 53: 74 * 16}[t_prec]
    optimal_order = {11: 9, 24: 13, 53: 19}[t_prec]
    size = 1000
    samples = list(utils.real_samples(size, dtype=dtype, min_value=dtype(0), max_value=dtype(numpy.pi / 4)))
    size = len(samples)
    with mpmath.mp.workprec(working_prec):
        mpctx = mpmath.mp
        for order in [optimal_order, 1, 3, 5, 7, 9, 11, 13, 17, 19][:1]:

            if func.startswith("cosm1"):

                def f_expected(x):
                    return mpctx.cos(x) - 1

            else:

                def f_expected(x):
                    return mpctx.cos(x)

            if func in {"cos_dekker", "cosm1_dekker"}:

                @fa.targets.numpy.jit(
                    paths=[fpa],
                    dtype=dtype,
                    debug=(1.5 if size <= 10 else 0),
                    rewrite_parameters=dict(eliminate_zero_factors=True),
                )
                def f(ctx, x):
                    with ctx.parameters(series_uses_dekker=True, series_uses_2sum=True):
                        return fpa.cosine_taylor_dekker(ctx, x, order=order, drop_leading_term=func.startswith("cosm1"))

            elif func == "cos":

                @fa.targets.numpy.jit(
                    paths=[fpa],
                    dtype=dtype,
                    debug=(1.5 if size <= 10 else 0),
                    rewrite_parameters=dict(optimize_cast=False, fma_backend=fma),
                )
                def f(ctx, x):
                    return fpa.cosine_taylor(ctx, x, order=order, split=False)

            elif func == "cosm1":

                @fa.targets.numpy.jit(
                    paths=[fpa],
                    dtype=dtype,
                    debug=(1.5 if size <= 10 else 0),
                    rewrite_parameters=dict(optimize_cast=False, fma_backend=fma),
                )
                def f(ctx, x):
                    return fpa.cosine_taylor(ctx, x, order=order, split=False, drop_leading_term=True)

            elif func == "cosm1_sin":

                @fa.targets.numpy.jit(
                    paths=[fpa],
                    dtype=dtype,
                    debug=(1.5 if size <= 10 else 0),
                    rewrite_parameters=dict(optimize_cast=False, fma_backend=fma),
                )
                def f(ctx, x):
                    two = ctx.constant(2, x)
                    sn = fpa.sine_taylor(ctx, x / two, order=order, split=False)
                    return -two * sn * sn

            elif func == "cos_numpy":

                f = numpy.cos

            elif func == "cosm1_numpy":

                def f(x):
                    return numpy.cos(x) - dtype(1)

            elif func == "cosm1_sin_numpy":

                def f(x):
                    two = dtype(2)
                    sn = numpy.sin(x / two)
                    return -two * sn * sn

            else:
                assert 0, func  # not impl

            ulp = defaultdict(int)
            for x in samples:
                expected = utils.mpf2float(dtype, f_expected(utils.float2mpf(mpctx, x)))
                cs = f(x)
                u = utils.diff_ulp(cs, expected)
                ulp[u] += 1

            print()
            show_ulp(ulp)


@pytest.mark.parametrize("func,fma", [("fast", "upcast"), ("fast", "native"), ("dekker", None), ("numpy.power", None)])
@pytest.mark.parametrize("exponent", [2, 3, 4, 5])
def test_fast_exponent_by_squaring(dtype, exponent, func, fma):
    import mpmath

    t_prec = utils.get_precision(dtype)

    working_prec = {11: 50 * 2, 24: 50 * 2, 53: 74 * 2}[t_prec]
    npctx = fa.utils.NumpyContext()
    size = 1000
    samples = list(utils.real_samples(size, dtype=dtype, min_value=dtype(0), max_value=dtype(numpy.pi / 4)))

    with mpmath.mp.workprec(working_prec):
        mpctx = mpmath.mp

        @fa.targets.numpy.jit(
            paths=[fpa],
            dtype=dtype,
            debug=(1.5 if size <= 10 else 0),
            rewrite_parameters=dict(optimize_cast=False, fma_backend=fma),
        )
        def fast_func(ctx, x):
            return fpa.fast_exponent_by_squaring(ctx, x, exponent)

        @fa.targets.numpy.jit(
            paths=[fpa],
            dtype=dtype,
            debug=(1.5 if size <= 10 else 0),
            rewrite_parameters=dict(optimize_cast=False, fma_backend=fma),
        )
        def dekker_func(ctx, x):
            seq = fpa.fast_exponent_by_squaring_dekker(ctx, x, exponent)
            if type(seq) is tuple:
                return sum(reversed(seq[:-1]), seq[-1])
            return seq

        ulp = defaultdict(int)
        for x in samples[::-1]:
            x_mp = utils.float2mpf(mpctx, x)
            expected = utils.mpf2float(dtype, x_mp**exponent)

            if func == "fast":
                r = fast_func(x)
            elif func == "dekker":
                r = dekker_func(x)
            elif func == "numpy.power":
                r = numpy.power(x, exponent, dtype=dtype)
            else:
                assert 0, func  # not implemented

            u = utils.diff_ulp(r, expected, flush_subnormals=True)
            ulp[u] += 1

            if 0:
                c = "." if u == 0 else ("v" if r < expected else "^")
                print(c, end="")

        show_ulp(ulp)


@pytest.mark.parametrize("backend", ["native", "upcast", "upcast2"])
def test_fma(dtype, backend):

    if (dtype, backend) in {
        (numpy.float64, "upcast2"),
        (numpy.float128, "upcast"),
        (numpy.float128, "upcast2"),
    }:
        pytest.skip(f"support not implemented")

    import mpmath

    t_prec = utils.get_precision(dtype)
    working_prec = {11: 50 * 4, 24: 50 * 4, 53: 74 * 4}[t_prec]
    mth = dict(native=fpa.fma_native, upcast=fpa.fma_upcast, upcast2=fpa.fma_upcast2)[backend]
    npctx = NumpyContext()
    size = 30
    samples = list(utils.real_samples(size, dtype=dtype, min_value=dtype(-1), max_value=dtype(1)))

    with mpmath.mp.workprec(working_prec):
        mpctx = mpmath.mp
        samples_mp = [utils.float2mpf(mpctx, v) for v in samples]
        ulp = defaultdict(int)
        for x, x_mp in zip(samples, samples_mp):
            for y, y_mp in zip(samples, samples_mp):
                for z, z_mp in zip(samples, samples_mp):
                    expected = utils.mpf2float(dtype, x_mp * y_mp + z_mp)
                    r = mth(npctx, x, y, z)
                    u = utils.diff_ulp(r, expected, flush_subnormals=True)

                    ulp[u] += 1
                    if 0:
                        c = "." if u == 0 else ("v" if r < expected else "^")
                        print(c, end="")
        show_ulp(ulp)


@pytest.mark.parametrize("func", ["sine", "numpy.sin"])
def test_sine(dtype, func):
    import mpmath
    from collections import defaultdict

    npctx = NumpyContext()
    fi = numpy.finfo(dtype)
    t_prec = utils.get_precision(dtype)
    working_prec = {11: 24, 24: 50 * 4, 53: 74 * 16}[t_prec]
    size = 1000
    samples = list(utils.real_samples(size, dtype=dtype, min_value=dtype(0), max_value=dtype(numpy.pi / 4)))
    samples = list(utils.real_samples(size, dtype=dtype, min_value=dtype(0), max_value=fi.max))
    size = len(samples)
    with mpmath.mp.workprec(working_prec):
        mpctx = mpmath.mp

        def f_expected(x):
            return mpctx.sin(x)

        if func == "sine":

            @fa.targets.numpy.jit(
                paths=[fpa],
                dtype=dtype,
                debug=(1.5 if size <= 10 else 0),
            )
            def f(ctx, x):
                return fpa.sine(ctx, x)

        elif func == "numpy.sin":

            f = numpy.sin

        else:
            assert 0, func  # not implemented

        ulp = defaultdict(int)
        for x in samples:
            expected = utils.mpf2float(dtype, f_expected(utils.float2mpf(mpctx, x)))
            result = f(x)
            u = utils.diff_ulp(result, expected)
            if u > 5:
                print(f"{u=} {x, result, expected=}")
            ulp[u] += 1

        print()
        show_ulp(ulp)
