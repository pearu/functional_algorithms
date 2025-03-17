import numpy
import pytest
import os
from collections import defaultdict
import functional_algorithms as fa


@pytest.fixture(scope="function", params=[numpy.float32, numpy.float64])
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
                    j_sc = sc.jv(dtype(order), x)
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
