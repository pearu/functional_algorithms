"""See https://github.com/pearu/functional_algorithms/issues/70

The resolution to the issue is based on using Dekker's product.

References:
  https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf
  https://inria.hal.science/hal-04480440v1

"""

import mpmath
import numpy as np
from functional_algorithms import utils
from collections import defaultdict


def main():
    dtype = np.float32
    fi = np.finfo(dtype)

    def expr1(x, y):
        return x + x + x * x + y * y

    def expr_decker(x, y):
        x2_r1, x2_r2 = utils.double_twosum(x)  # same accuracy as using x + x
        xx_r1, xx_r2 = utils.square_dekker(x)
        yy_r1, yy_r2 = utils.square_dekker(y)
        if 0:
            r = x2_r1
            r += yy_r1
            r += xx_r1
            r += yy_r2
            r += xx_r2
            r += x2_r2
            return r
        s, t = utils.add_twosum(x2_r1, yy_r1)
        s, t2 = utils.add_twosum(s, xx_r1)
        s, t3 = utils.add_twosum(s, yy_r2)
        s, t4 = utils.add_twosum(s, xx_r2)
        s, t5 = utils.add_twosum(s, x2_r2)
        s = s + t + t2 + t3 + t4 + t5
        return s

    def argsort(seq):
        # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        return sorted(range(len(seq)), key=seq.__getitem__)

    def expr_decker_ordering(x, y):
        """
        x + x + y * y + x * x =
          = (x2h + x2l) + (yyh + yyl) + (xxh + xxl)
          = x2h + yyh + xxh + x2l + yyl + xxl
        """
        C = utils.get_veltkamp_splitter_constant(x)
        x2h, x2l = utils.double_fast2sum(x)  # 2sum: {0: 999685, 1: 315}
        x2h, x2l = x + x, type(x)(0)  # 2sum: {0: 999685, 1: 315}
        # => x2l can be eliminated

        xxh, xxl = utils.square_dekker(x, C=C)
        yyh, yyl = utils.square_dekker(y, C=C)

        seq = [xxh, yyh, x2l, x2h, xxl, yyl]
        # 2sum: {0: 999685, 1: 315}
        # 2sum, on curve: {0: 9283, 1: 650, 2: 64, 3: 3}
        # fast2sum: {0: 991685, 1: 8214, 8: 2, 2: 50, 6: 5, 4: 11, 3: 16, 9: 2, 5: 4, 13: 1, 20: 1, 206: 1, 63: 1, 15: 1, 60: 1, 10: 1, 7: 1, 23: 1, 14: 1, 32: 1}

        seq = [x2h, yyh, xxh, x2l, xxl, yyl]
        # 2sum, on curve: {0: 9344, 1: 601, 2: 55}, diff_ulp == 2 exists when r_decker==-3e-45 but r_mp=-0.0
        # 2sum: {0: 999685, 1: 315}
        # fast2sum: {0: 950732, 1: 49268}

        seq = [x2h, yyh, xxh, yyl, xxl, x2l]
        # 2sum, on curve: {0: 9344, 1: 601, 2: 55}
        # fast2sum: {0: 950732, 1: 49268}

        _seq = [x2h, yyh, xxh, yyl, x2l, xxl]
        # 2sum, on curve: {0: 9344, 1: 601, 2: 55}
        # 2sum: {0: 999685, 1: 315}
        # fast2sum: {0: 950732, 1: 49268}

        _seq = [yyh, x2h, xxh, yyl, x2l, xxl]
        # fast2sum: {0: 961695, 1: 38169, 4: 18, 512: 1, 2: 75, 8: 5, 16: 1, 12: 1, 3: 34, 128: 1}

        _seq = [yyh, x2h, xxh, x2l, xxl, yyl]
        # 2sum: {0: 999685, 1: 315}
        # fast2sum: {0: 961695, 1: 38169, 4: 18, 512: 1, 2: 75, 8: 5, 16: 1, 12: 1, 3: 34, 128: 1}

        # seq = list(reversed(seq))  # 2sum: {0: 999685, 1: 315}
        # => the order of applying add_2sum does matter on curve only
        # => the order of applying add_fast2sum does matter

        if 0:  # same accuracy as in elif block below
            s, t = utils.add_2sum(*seq[:2])
            t2 = type(t)(0)
            for n in seq[2:]:
                s, t1 = utils.add_2sum(s, n)
                t, t2_ = utils.add_2sum(t, t1)
                t2 += t2_
            return s + t + t2
        elif 1:
            s, t = utils.sum_2sum(seq)
            return s + t
        elif 1:
            s, t = utils.sum_fast2sum(seq)
            return s + t
        else:
            s, t = utils.add_2sum(x2h, yyh)
            s, t2 = utils.add_2sum(s, xxh)
            s, t3 = utils.add_2sum(s, x2l)
            s, t4 = utils.add_2sum(s, xxl)
            s, t5 = utils.add_2sum(s, yyl)
            s = s + t + t2 + t3 + t4 + t5
            return s

    def expr_decker2(x, y):
        C = utils.get_veltkamp_splitter_constant(x)
        xxh, xxl = utils.square_dekker(x, C=C)
        yyh, yyl = utils.square_dekker(y, C=C)
        return yyh + x + x + xxh + yyl + xxl

    def expr_mpmath(x, y):
        ctx = mpmath.mp
        ctx.prec = 5000
        x_mp = utils.float2mpf(ctx, x)
        y_mp = utils.float2mpf(ctx, y)

        r = x_mp + x_mp + y_mp * y_mp + x_mp * x_mp
        return utils.mpf2float(type(x), r)

    min_y = np.sqrt(dtype(2) * np.sqrt(fi.smallest_subnormal))
    min_y = 0

    max_ulp_error_naive = 0
    ulp_error_decker = defaultdict(int)

    def samples_on_curve():
        for y in utils.real_samples(size=10000, dtype=dtype, min_value=min_y, max_value=1):
            x = -dtype(0.5) * y * y
            yield x, y

    def samples_cross_curve():
        y = dtype(2e-19)
        for x in utils.real_samples(size=20, dtype=dtype, min_value=-1e-37, max_value=-1e-39):
            yield x, y

    def samples_on_plane():
        size = 1000
        for s in utils.complex_samples(
            (size, size),
            dtype=dtype,
            min_real_value=-np.sqrt(fi.max) * 0.99,
            max_real_value=np.sqrt(fi.max) * 0.99,
            min_imag_value=-np.sqrt(fi.max) * 0.01,
            max_imag_value=np.sqrt(fi.max) * 0.01,
        ).flatten():
            yield s.real, s.imag

    def samples_with_large_diff_ulp():
        for x, y in [
            (-1.1196316, -0.9936756),  # 512
            (-1.810698, 0.5806803),  # 128
            (-5.9593778e-05, 0.010990744),  # 33
            (-2.3108913e-10, 2.1596348e-05),  # 32
            (-4.1076538e-22, 2.901285e-11),  # 16
            (-1.810698, -0.47763407),  # 20
            (-1.810698, -0.6347373),  # 15
            (-7.0303718e-06, 0.0038440996),  # 9
            (-0.9939759, 0.00020737245),
        ]:
            yield dtype(x), dtype(y)

    samples_func = samples_on_curve
    samples_func = samples_on_plane
    # samples_func = samples_cross_curve
    samples_func = samples_with_large_diff_ulp

    for x, y in samples_func():
        r_naive = expr1(x, y)
        # expr_decker: max_ulp_error_naive=12272 dict(ulp_error_decker)={0: 941964, 1: 58036}
        r_decker = expr_decker_ordering(x, y)
        r_mp = expr_mpmath(x, y)

        d1 = utils.diff_ulp(r_naive, r_mp)
        d2 = utils.diff_ulp(r_decker, r_mp)

        if d2 > 2:
            print(f"{x, y=} {r_decker=} {r_mp=} {d2=}")

        max_ulp_error_naive = max(max_ulp_error_naive, d1)
        ulp_error_decker[d2] += 1

    print(f"{max_ulp_error_naive=} {dict(ulp_error_decker)=}")


if __name__ == "__main__":
    main()
