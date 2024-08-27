import os
import numpy as np
import functional_algorithms as fa


def get_inputs():
    for func_name, dtype, parameters in [
        # ("absolute", np.float32, {}),  # disabling as the result is trivial
        # ("absolute", np.float64, {}),  # disabling as the result is trivial
        ("absolute", np.complex64, {}),
        ("absolute", np.complex128, {}),
        ("acos", np.float32, {}),
        ("acos", np.float64, {}),
        ("acos", np.complex64, {}),
        ("acos", np.complex128, {}),
        ("acosh", np.float32, {}),
        # *(("acosh", np.float32, dict(safe_max_limit_coefficient=v)) for v in [0.51, 0.5]),
        # *(("acosh", np.float64, dict(safe_max_limit_coefficient=v)) for v in [0.51, 0.5]),
        ("acosh", np.float64, {}),
        ("acosh", np.complex64, {}),
        ("acosh", np.complex128, {}),
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
        ("atan", np.float32, {}),
        ("atan", np.float64, {}),
        ("atan", np.complex64, {}),
        ("atan", np.complex128, {}),
        ("atanh", np.float32, {}),
        ("atanh", np.float64, {}),
        ("atanh", np.complex64, {}),
        ("atanh", np.complex128, {}),
        ("square", np.float32, {}),
        ("square", np.float64, {}),
        ("square", np.complex64, {}),
        ("square", np.complex128, {}),
        ("sqrt", np.float32, {}),
        ("sqrt", np.float64, {}),
        ("sqrt", np.complex64, {}),
        ("sqrt", np.complex128, {}),
        ("angle", np.complex64, {}),
        ("angle", np.complex128, {}),
        ("log1p", np.float32, {}),
        ("log1p", np.float64, {}),
        ("log1p", np.complex64, {}),
        ("log1p", np.complex128, {}),
    ]:
        validation_parameters = fa.utils.function_validation_parameters(func_name, dtype)
        max_bound_ulp_width = validation_parameters["max_bound_ulp_width"]
        if func_name == "log1p" and dtype is np.complex128:
            # the original max_bound_ulp_width value is required for log1p extra samples
            max_bound_ulp_width = 1
        for max_bound_ulp_width in range(1 if max_bound_ulp_width else 0, max_bound_ulp_width + 1):
            validation_parameters["max_bound_ulp_width"] = max_bound_ulp_width
            if parameters:
                params = "[" + ", ".join(f"{k}={v}" for k, v in parameters.items()) + "]"
            else:
                params = ""
            if max_bound_ulp_width > 0:
                row_prefix = f"| {func_name}<sup>{max_bound_ulp_width}</sup>{params} | {dtype.__name__} | "
            else:
                row_prefix = f"| {func_name}{params} | {dtype.__name__} | "
            yield func_name, dtype, parameters, validation_parameters, row_prefix


def main():
    target_file = os.path.join(os.path.dirname(__file__), "README.md")

    precomputed = dict()
    f = open(target_file, "r")
    for line in f.readlines():
        if line.startswith("|"):
            for func_name, dtype, parameters, validation_parameters, row_prefix in get_inputs():
                if line.startswith(row_prefix):
                    precomputed[row_prefix] = line.rstrip()
    f.close()

    f = open(target_file, "w")
    size = 1000
    flush_subnormals = False
    print(
        """
# Accuracy of provided algorithms

<sub>This file is generated using estimate_accuracy.py. Do not edit!</sub>

The reference values are obtained by evaluating MPMath functions using
multi-precision arithmetic.

The following table shows the counts of samples that produce function
values being
- different from expected values by the given ULP difference (dULP):
  ```
  ulp_diff(func(sample), reference(sample)) == dULP
  ```

- out of the reference values ulp-range:
  ```
  not (lower <= func(sample) <= upper)
  lower = minimal(reference(s) for s in surrounding(sample) if diff_ulp(s, sample) <= ulp_width)
  upper = maximal(reference(s) for s in surrounding(sample) if diff_ulp(s, sample) <= ulp_width)
  ```

When a counts value is attributed with a superscript, this indicates
the number of samples that lead to out-of-ulp-range results. When dULP
<= 3, out-of-ulp-range counts are acceptable as it typically indicates
that reference function is not sensitive to input perturbations, that
is, `lower == reference(sample) == upper` holds. On the other hand,
when the out-of-ulp-range counts is zero, dULP > 3 counts are
acceptable as it indicates that function's variability is very high
with respect to minimal variations in its input.

When `ulp_width` is specified, its value is indicated as a superscript
in function name. Notice the specified `ulp_width` is not the upper
limit in general: there may exist function-function dependent regions
in complex plane where `ulp_width` needs to be larger to pass the
"out-of-ulp-range counts is zero" test.

""",
        file=f,
    )

    print(
        "| Function | dtype | dULP=0 (exact) | dULP=1 | dULP=2 | dULP=3 | dULP>3 |",
        file=f,
    )
    print("| -------- | ----- | -------------- | ------ | ------ | ------ | ------ |", file=f)

    for func_name, dtype, parameters, validation_parameters, row_prefix in get_inputs():
        if row_prefix in precomputed:
            print(f"{row_prefix}- using previous result.")
            print(precomputed[row_prefix], file=f)
            continue
        print(f"{row_prefix}")
        ctx = fa.Context(paths=[fa.algorithms], parameters=parameters)
        graph = ctx.trace(getattr(fa.algorithms, func_name), dtype)
        graph2 = graph.implement_missing(fa.targets.numpy).simplify()
        func = fa.targets.numpy.as_function(graph2, debug=0)

        max_valid_ulp_count = validation_parameters["max_valid_ulp_count"]
        max_bound_ulp_width = validation_parameters["max_bound_ulp_width"]
        extra_prec_multiplier = validation_parameters["extra_prec_multiplier"]

        reference = getattr(
            fa.utils.numpy_with_mpmath(extra_prec_multiplier=extra_prec_multiplier, flush_subnormals=flush_subnormals),
            dict(real_asinh="asinh", real_asinh_2="asinh", acos="arccos", asin="arcsin", asinh="arcsinh", acosh="arccosh").get(
                func_name, func_name
            ),
        )

        if dtype in {np.complex64, np.complex128}:
            samples = fa.utils.complex_samples(
                (size, size), dtype=dtype, include_huge=True, include_subnormal=not flush_subnormals
            ).flatten()
        else:
            samples = fa.utils.real_samples(size * size, dtype=dtype, include_subnormal=not flush_subnormals).flatten()
        matches_with_reference, stats = fa.utils.validate_function(
            func,
            reference,
            samples,
            dtype,
            verbose=not False,
            flush_subnormals=flush_subnormals,
            enable_progressbar=True,
            workers=None,
            max_valid_ulp_count=max_valid_ulp_count,
            max_bound_ulp_width=max_bound_ulp_width,
        )
        # assert matches_with_reference, (func_name, dtype, ulp_stats)

        cols = []
        total = sum(stats["ulp"].values())
        for ulp in [0, 1, 2, 3]:
            if ulp not in stats["ulp"]:
                cols.append("-")
            else:
                t = f"{stats['ulp'][ulp]}"
                outrange = stats["outrange"][ulp]
                if outrange > 0:
                    t += f"<sup>{outrange}</sup>"
                cols.append(t)

        ulps = sum([stats["ulp"][ulp] for ulp in stats["ulp"] if ulp > 3])
        outrange = sum([stats["outrange"][ulp] for ulp in stats["outrange"] if ulp > 3])
        if ulps:
            t = f"{ulps}"
            if outrange > 0:
                t += f"<sup>{outrange}</sup>!!"
            cols.append(t)
        else:
            cols.append("-")
        print(row_prefix + " | ".join(cols) + " |", file=f)
        print(row_prefix + " | ".join(cols) + " |")
        f.flush()
    print(f"Created {target_file}")


if __name__ == "__main__":
    main()
