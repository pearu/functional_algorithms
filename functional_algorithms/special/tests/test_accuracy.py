import numpy
import pytest
import os
from collections import defaultdict
import functional_algorithms as fa


@pytest.fixture(scope="function", params=[numpy.float16, numpy.float32, numpy.float64])
def real_dtype(request):
    return request.param


@pytest.mark.parametrize(
    "backend,order",
    [
        ("scipy", 0),
        ("scipy", 1),
        ("scipy", 2),
        ("scipy", 3),
        ("small_x", 0),
        ("small_x", 1),
        ("small_x", 2),
        ("small_x", 3),
        ("large_x", 0),
        ("large_x", 1),
        ("large_x", 2),
        ("large_x", 3),
    ],
)
def test_bessel_validation(real_dtype, backend, order):
    import scipy.special as sc
    import mpmath

    dtype = real_dtype
    fi = numpy.finfo(dtype)

    def large_x_mp(ctx, x):
        h = ctx.cos(x - ctx.pi / 4 - order * ctx.pi / 2)
        return ctx.sqrt(2 / (ctx.pi * x)) * h

    def small_x_np(order, x):
        return dtype((x / 2) ** order)

    def large_x_np(order, x):
        p = numpy.pi / 4 + order * numpy.pi / 2
        h = numpy.cos(x) * numpy.cos(p) + numpy.sin(x) * numpy.sin(p)
        return numpy.sqrt(dtype(2 / (numpy.pi * x))) * dtype(h)

    ulp_ranges = [
        (0, 0, "dULP=0"),
        (1, 1, "dULP=1"),
        (2, 2, "dULP=2"),
        (3, 9, "dULP=3..9"),
        (10, 1000, "dULP=10..1000"),
        (1001, 1000000, "dULP=1001..1000000"),
        (1000001, numpy.inf, "dULP > 1000000"),
    ]

    rows = []
    col_width = defaultdict(int)
    for start, stop, label in [
        (0, fi.tiny, "0..tiny"),
        (fi.tiny, fi.eps**2, "tiny..eps**2"),
        (fi.eps**2, fi.eps, "eps**2..eps"),
        (fi.eps, fi.eps**0.5, "eps..sqrt(eps)"),
        (fi.eps**0.5, 1, "sqrt(eps)..1"),
        (1, 10, "1..10"),
        (10, 100, "10..100"),
        (1e2, 1 / fi.eps**0.5, "100..1/sqrt(eps)"),
        (1 / fi.eps**0.5, 1 / fi.eps, "1/sqrt(eps)..1/eps"),
        (1 / fi.eps, 1 / fi.eps**2, "1/eps..1/eps**2"),
        (1 / fi.eps**2, fi.max**0.5, "1/eps..sqrt(largest)"),
        (fi.max**0.5, fi.max, "sqrt(largest)..largest"),
    ]:
        if stop < start or numpy.isinf(dtype(start)) or numpy.isinf(dtype(stop)):
            continue
        size = 100
        samples = fa.utils.real_samples(size, dtype=dtype, min_value=start, max_value=stop)

        max_prec = {numpy.float16: 24, numpy.float32: 149, numpy.float64: 1074}[dtype]

        ulp_sc_mp = defaultdict(int)

        mp_ctx = mpmath.mp
        with mp_ctx.workprec(max_prec):
            for x in samples:
                x_mp = fa.utils.float2mpf(mp_ctx, x)
                j_mp = fa.utils.mpf2float(dtype, mp_ctx.besselj(order, x_mp))
                if backend == "scipy":
                    j_sc = sc.jv(dtype(order), x).astype(dtype)
                elif backend == "large_x":
                    j_sc = large_x_np(order, x)
                elif backend == "small_x":
                    j_sc = small_x_np(order, x)
                else:
                    assert 0, backend  # not implemented
                u = fa.utils.diff_ulp(j_sc, j_mp)
                ulp_sc_mp[u] += 1
                if u > 1000 and u < 1000000 and 0:
                    print(f"{u=}: {x=} {j_mp=} {j_sc=}")

        ur = defaultdict(int)
        for u in ulp_sc_mp:
            for u0, u1, l in ulp_ranges:
                if u >= u0 and u <= u1:
                    ur[l] += ulp_sc_mp[u]

        row = [label]
        col_width[0] = max(col_width[0], len(label))
        for i, (u0, u1, l) in enumerate(ulp_ranges):
            c = ur[l]
            if c == 0:
                row.append("-")
            else:
                row.append(f"{c}")
            col_width[i + 1] = max(col_width[i + 1], len(row[-1]))
        rows.append(row)

    lines = []

    row = ["x range"] + [l for _, _, l in ulp_ranges]
    for j in range(len(row)):
        col_width[j] = max(col_width[j], len(row[j]))

    lst = []
    for j, r in enumerate(row):
        fmt = "{:" + str(col_width[j] + 2) + "}"
        lst.append(fmt.format(r))
    lines.append(" | ".join(lst))

    lst = []
    for j in range(len(row)):
        lst.append("-" * (col_width[j] + 2))
    lines.append("-+-".join(lst))

    for i, row in enumerate(rows):
        lst = []
        for j, r in enumerate(row):
            fmt = "{:" + str(col_width[j] + 2) + "}"
            lst.append(fmt.format(r))
        lines.append(" | ".join(lst))

    print()
    print("\n".join(lines))


@pytest.mark.parametrize(
    "backend,a,b,c",
    [
        ("scipy", 2, 1, 4),
        ("jax", 2, 1, 4),
    ],
)
def test_hyp2f1(real_dtype, backend, a, b, c):
    import scipy.special as sc
    import mpmath

    if backend == "jax":
        try:
            import jax
        except ImportError as msg:
            pytest.skip(f"failed to import jax: {msg}")

    dtype = real_dtype
    fi = numpy.finfo(dtype)

    ulp_ranges = [
        (0, 0, "dULP=0"),
        (1, 1, "dULP=1"),
        (2, 2, "dULP=2"),
        (3, 9, "dULP=3..9"),
        (10, 1000, "dULP=10..1000"),
        (1001, 1000000, "dULP=1001..1000000"),
        (1000001, numpy.inf, "dULP > 1000000"),
    ]
    rows = []
    col_width = defaultdict(int)
    for start, stop, label in [
        (0, fi.tiny, "0..tiny"),
        (fi.tiny, fi.eps**2, "tiny..eps**2"),
        (fi.eps**2, fi.eps, "eps**2..eps"),
        (fi.eps, fi.eps**0.5, "eps..sqrt(eps)"),
        (fi.eps**0.5, 1, "sqrt(eps)..1"),
        # (1, 10, "1..10"),
        # (10, 100, "10..100"),
        # (1e2, 1 / fi.eps**0.5, "100..1/sqrt(eps)"),
        # (1 / fi.eps**0.5, 1 / fi.eps, "1/sqrt(eps)..1/eps"),
        # (1 / fi.eps, 1 / fi.eps**2, "1/eps..1/eps**2"),
        # (1 / fi.eps**2, fi.max**0.5, "1/eps..sqrt(largest)"),
        # (fi.max**0.5, fi.max, "sqrt(largest)..largest"),
    ]:
        if stop < start or numpy.isinf(dtype(start)) or numpy.isinf(dtype(stop)):
            continue
        size = 10000
        samples = fa.utils.real_samples(size, dtype=dtype, min_value=start, max_value=stop)

        max_prec = {numpy.float16: 24, numpy.float32: 149, numpy.float64: 1074}[dtype]

        ulp_sc_mp = defaultdict(int)

        mp_ctx = mpmath.mp
        with mp_ctx.workprec(max_prec):
            for x in samples:
                x_mp = fa.utils.float2mpf(mp_ctx, x)
                expected = fa.utils.mpf2float(dtype, mp_ctx.hyper([a, b], [c], x_mp))
                if backend == "scipy":
                    result = sc.hyp2f1(dtype(a), dtype(b), dtype(c), x).astype(dtype)
                elif backend == "jax":
                    import jax.scipy.special as jsc

                    result = numpy.asarray(jsc.hyp2f1(dtype(a), dtype(b), dtype(c), x))[()]
                else:
                    assert 0, backend  # not implemented
                u = fa.utils.diff_ulp(result, expected)
                ulp_sc_mp[u] += 1
                if u > 1000 and u < 1000000:
                    print(f"{u=}: {x=} {expected=} {result=} {a, b, c}")

        ur = defaultdict(int)
        for u in ulp_sc_mp:
            for u0, u1, l in ulp_ranges:
                if u >= u0 and u <= u1:
                    ur[l] += ulp_sc_mp[u]

        row = [label]
        col_width[0] = max(col_width[0], len(label))
        for i, (u0, u1, l) in enumerate(ulp_ranges):
            c_ = ur[l]
            if c_ == 0:
                row.append("-")
            else:
                row.append(f"{c_}")
            col_width[i + 1] = max(col_width[i + 1], len(row[-1]))
        rows.append(row)

    lines = []

    row = ["x range"] + [l for _, _, l in ulp_ranges]
    for j in range(len(row)):
        col_width[j] = max(col_width[j], len(row[j]))

    lst = []
    for j, r in enumerate(row):
        fmt = "{:" + str(col_width[j] + 2) + "}"
        lst.append(fmt.format(r))
    lines.append(" | ".join(lst))

    lst = []
    for j in range(len(row)):
        lst.append("-" * (col_width[j] + 2))
    lines.append("-+-".join(lst))

    for i, row in enumerate(rows):
        lst = []
        for j, r in enumerate(row):
            fmt = "{:" + str(col_width[j] + 2) + "}"
            lst.append(fmt.format(r))
        lines.append(" | ".join(lst))

    print()
    print("\n".join(lines))


@pytest.mark.parametrize(
    "backend,b",
    [
        ("scipy", 1),
        ("jax", 1),
        ("fa", 1),
    ],
)
def test_hyp0f1(real_dtype, backend, b):
    import scipy.special as sc
    import mpmath
    import functional_algorithms.generalized_hypergeometric_functions as ghf

    if backend == "jax":
        try:
            import jax
        except ImportError as msg:
            pytest.skip(f"failed to import jax: {msg}")

    dtype = real_dtype
    fi = numpy.finfo(dtype)

    ulp_ranges = [
        (0, 0, "dULP=0"),
        (1, 1, "dULP=1"),
        (2, 2, "dULP=2"),
        (3, 9, "dULP=3..9"),
        (10, 1000, "dULP=10..1000"),
        (1001, 1000000, "dULP=1001..1000000"),
        (1000001, numpy.inf, "dULP > 1000000"),
    ]
    rows = []
    col_width = defaultdict(int)
    for start, stop, label in [
        (0, fi.tiny, "0..tiny"),
        (fi.tiny, fi.eps**2, "tiny..eps**2"),
        (fi.eps**2, fi.eps, "eps**2..eps"),
        (fi.eps, fi.eps**0.5, "eps..sqrt(eps)"),
        (fi.eps**0.5, 1, "sqrt(eps)..1"),
        (1, 10, "1..10"),
        (10, 100, "10..100"),
        (1e2, 1 / fi.eps**0.5, "100..1/sqrt(eps)"),
        (1 / fi.eps**0.5, 1 / fi.eps, "1/sqrt(eps)..1/eps"),
        (1 / fi.eps, 1 / fi.eps**2, "1/eps..1/eps**2"),
        (1 / fi.eps**2, fi.max**0.5, "1/eps..sqrt(largest)"),
        (fi.max**0.5, fi.max / 10, "sqrt(largest)..largest/10"),
        (fi.max / 10, fi.max, "largest/10..largest"),
    ]:
        if stop < start or numpy.isinf(dtype(start)) or numpy.isinf(dtype(stop)):
            continue
        size = 100

        samples = -fa.utils.real_samples(size, dtype=dtype, min_value=start, max_value=stop)
        max_prec = {numpy.float16: 27, numpy.float32: 149, numpy.float64: 1074}[dtype]

        ulp_sc_mp = defaultdict(int)
        ulp_sc_mp_zeros = defaultdict(int)

        mp_ctx = mpmath.mp
        with mp_ctx.workprec(max_prec):

            results = None
            if backend in {"fa"}:  # backends that support vectorization
                if backend == "fa":
                    results = fa.special.hyp0f1(int(b), samples, enable_largest_correction=True, zero_indices=None)
                else:
                    assert 0, backend  # not implemented

            expected_lst = []
            for i, x in enumerate(samples):
                x_mp = fa.utils.float2mpf(mp_ctx, x)
                expected_mp = mp_ctx.hyper([], [b], x_mp)
                expected_ex = fa.utils.mpf2expansion(dtype, expected_mp)
                expected = sum(expected_ex[:-1], expected_ex[-1])
                expected_lst.append(expected)

            for i, x in enumerate(samples):
                expected = expected_lst[i]
                if results is not None:
                    result = results[i]
                elif backend == "scipy":
                    result = sc.hyp0f1(dtype(b), x).astype(dtype)
                elif backend == "jax":
                    import jax.scipy.special as jsc

                    result = numpy.asarray(jsc.hyp0f1(dtype(b), x))[()]
                else:
                    assert 0, backend  # not implemented
                u = fa.utils.diff_ulp(result, expected)

                # Fast estimate of zero closeness:
                w = 1
                is_close_to_zero = i >= w and i < len(expected_lst) - w and expected_lst[i - w] * expected_lst[i + w] < 0

                if u > 1 and not is_close_to_zero and backend == "fa":
                    # Slower but more robust estimate to zero closeness:
                    z = fa.utils.mpf2float(dtype, ghf.hyp0f1_closest_zero(b, x)[0])
                    w = dtype(1e-2 * abs(x) ** 0.61)
                    is_close_to_zero = abs(x - z) < w

                if is_close_to_zero:
                    ulp_sc_mp_zeros[u] += 1
                else:
                    ulp_sc_mp[u] += 1

                if u >= 1 and not is_close_to_zero and 0:
                    print(f"{u=}: {x=} {expected=} {result=} {b=}")

                if backend == "fa" and not is_close_to_zero:
                    if u > 1:
                        print(f"{u=}: {x=} {expected=} {result=} {label=}")
                    # assert u <= 1
                    # u > 1 is expected when x is close to a 0f1 zero
                    # that is not listed in zero_indices.
        ur = defaultdict(int)
        for u in ulp_sc_mp:
            for u0, u1, l in ulp_ranges:
                if u >= u0 and u <= u1:
                    ur[l] += ulp_sc_mp[u]

        ur_z = defaultdict(int)
        for u in ulp_sc_mp_zeros:
            for u0, u1, l in ulp_ranges:
                if u >= u0 and u <= u1:
                    ur_z[l] += ulp_sc_mp_zeros[u]

        row = [label]
        col_width[0] = max(col_width[0], len(label))
        for i, (u0, u1, l) in enumerate(ulp_ranges):
            c_ = ur[l]
            c_z = ur_z[l]
            s = "-" if c_ == 0 else str(c_)
            s_z = "-" if c_z == 0 else str(c_z)
            if s == "-":
                if s_z == "-":
                    row.append(f"-")
                else:
                    row.append(f"({s_z})")
            elif s_z == "-":
                row.append(f"{s}")
            else:
                row.append(f"{s} ({s_z})")
            col_width[i + 1] = max(col_width[i + 1], len(row[-1]))
        rows.append(row)

    lines = []

    row = ["(-x) range"] + [l for _, _, l in ulp_ranges]
    for j in range(len(row)):
        col_width[j] = max(col_width[j], len(row[j]))

    lst = []
    for j, r in enumerate(row):
        fmt = "{:" + str(col_width[j] + 2) + "}"
        lst.append(fmt.format(r))
    lines.append(" | ".join(lst))

    lst = []
    for j in range(len(row)):
        lst.append("-" * (col_width[j] + 2))
    lines.append("-+-".join(lst))

    for i, row in enumerate(rows):
        lst = []
        for j, r in enumerate(row):
            fmt = "{:" + str(col_width[j] + 2) + "}"
            lst.append(fmt.format(r))
        lines.append(" | ".join(lst))

    print()
    print("\n".join(lines))
