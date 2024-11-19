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
        r = x2_r1
        r += yy_r1
        r += xx_r1
        r += yy_r2
        r += xx_r2
        r += x2_r2
        return r

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

    for y in utils.real_samples(size=10000, dtype=dtype, min_value=min_y, max_value=1):
        x = -dtype(0.5) * y * y
        r_naive = expr1(x, y)
        r_decker = expr_decker(x, y)
        r_mp = expr_mpmath(x, y)

        d1 = utils.diff_ulp(r_naive, r_mp)
        d2 = utils.diff_ulp(r_decker, r_mp)

        max_ulp_error_naive = max(max_ulp_error_naive, d1)
        ulp_error_decker[d2] += 1

    print(f"{max_ulp_error_naive=} {dict(ulp_error_decker)=}")


if __name__ == "__main__":
    main()
