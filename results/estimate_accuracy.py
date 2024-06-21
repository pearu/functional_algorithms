import os
import numpy as np
import functional_algorithms as fa


def get_inputs():
    for func_name, dtype, parameters in [
        ("absolute", np.float32, {}),
        ("absolute", np.float64, {}),
        ("absolute", np.complex64, {}),
        ("absolute", np.complex128, {}),
        ("acos", np.float32, {}),
        ("acos", np.float64, {}),
        ("acos", np.complex64, {}),
        ("acos", np.complex128, {}),
        ("asin", np.float32, {}),
        ("asin", np.float64, {}),
        ("asin", np.complex64, {}),
        ("asin", np.complex128, {}),
        # *(("real_asinh", np.float32, dict(safe_min_limit=v)) for v in [None, 1, 10, 100, 1000]),
        # *(("real_asinh", np.float64, dict(safe_min_limit=v)) for v in [None, 1, 10, 100, 1000]),
        # *(("real_asinh", np.float32, dict(safe_max_limit_coefficient=v)) for v in [2, 1, 1/2, 1/4, 1/8]),
        # ("real_asinh_2", np.float32, {}),
        ("asinh", np.float32, {}),
        ("asinh", np.float64, {}),
        ("asinh", np.complex64, {}),
        ("asinh", np.complex128, {}),
        ("square", np.float32, {}),
        ("square", np.float64, {}),
        ("square", np.complex64, {}),
        ("square", np.complex128, {}),
    ]:
        if parameters:
            params = "[" + ", ".join(f"{k}={v}" for k, v in parameters.items()) + "]"
        else:
            params = ""
        row_prefix = f"| {func_name}{params} | {dtype.__name__} | "
        yield func_name, dtype, parameters, row_prefix


def main():
    target_file = os.path.join(os.path.dirname(__file__), "README.md")

    precomputed = dict()
    f = open(target_file, "r")
    for line in f.readlines():
        if line.startswith("|"):
            for func_name, dtype, parameters, row_prefix in get_inputs():
                if line.startswith(row_prefix):
                    precomputed[row_prefix] = line.rstrip()
    f.close()

    f = open(target_file, "w")
    size = 1000
    flush_subnormals = False
    print(
        """
# Accuracy of provided algorithms

The following table shows the counts of samples that produce function
values being different from expected values by the given ULP
difference (dULP). The expected values are obtained by evaluating
MPMath functions using multi-precision arithmetic.
""",
        file=f,
    )

    print("| Function | dtype | dULP=0 (exact) | dULP=1 | dULP=2 | dULP=3 | dULP>3 | errors    |", file=f)
    print("| -------- | ----- | ------------- | ----- | ----- | ----- | ----- | --------- |", file=f)

    for func_name, dtype, parameters, row_prefix in get_inputs():
        if row_prefix in precomputed:
            print(f"{func_name}: {dtype=} {parameters=}, using previous result.")
            print(precomputed[row_prefix], file=f)
            continue
        print(f"{func_name}: {dtype=} {parameters=}")
        ctx = fa.Context(paths=[fa.algorithms], parameters=parameters)
        graph = ctx.trace(getattr(fa.algorithms, func_name), dtype)
        graph2 = graph.implement_missing(fa.targets.numpy).simplify()
        func = fa.targets.numpy.as_function(graph2, debug=0)

        if func_name in {"asin", "asinh", "acos"}:
            extra_prec_multiplier = 20
        else:
            extra_prec_multiplier = 1
        reference = getattr(
            fa.utils.numpy_with_mpmath(extra_prec_multiplier=extra_prec_multiplier, flush_subnormals=flush_subnormals),
            dict(real_asinh="asinh", real_asinh_2="asinh").get(func_name, func_name),
        )

        if dtype in {np.complex64, np.complex128}:
            samples = fa.utils.complex_samples(
                (size, size), dtype=dtype, include_huge=True, include_subnormal=not flush_subnormals
            ).flatten()
        else:
            samples = fa.utils.real_samples(size * size, dtype=dtype, include_subnormal=not flush_subnormals).flatten()
        matches_with_reference, ulp_stats = fa.utils.validate_function(
            func, reference, samples, dtype, verbose=False, flush_subnormals=flush_subnormals, enable_progressbar=True
        )
        # assert matches_with_reference, (func_name, dtype, ulp_stats)

        cols = []
        total = sum(ulp_stats.values())
        for ulp in [0, 1, 2, 3]:
            if ulp not in ulp_stats:
                cols.append("-")
            else:
                cols.append(f"{ulp_stats[ulp]}")

        ulps = sum([ulp_stats[ulp] for ulp in ulp_stats if ulp > 3])
        if ulps:
            cols.append(f"{ulps}")
        else:
            cols.append("-")
        if ulp_stats.get(-1, 0):
            cols.append(f"{ulp_stats[-1]}")
        else:
            cols.append(f"-")
        print(row_prefix + " | ".join(cols) + " |", file=f)
        f.flush()

    print(f"Created {target_file}")


if __name__ == "__main__":
    main()
