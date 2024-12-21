import functools
import os
import numpy as np
import functional_algorithms as fa
from collections import defaultdict


class Bucket:

    def __init__(self, samples):
        """
        Parameters
        ----------
        samples : ndarray
        Defines the shape of results accumulators.
        """
        self.samples = samples
        self.ulp_count = np.zeros(samples.shape, dtype=np.uint64)
        self.counts = np.zeros(samples.shape, dtype=np.uint64)

    def add(self, sample, dulp, result, expected):
        index = tuple(
            ind[-1] if ind.size else 0
            for ind in np.where((self.samples.real <= sample.real) & (self.samples.imag <= sample.imag))
        )
        self.ulp_count[index] += min(dulp, 1000)
        self.counts[index] += 1


def get_inputs():
    func_counts = defaultdict(int)

    for func_name, dtype, parameters in [
        # ("absolute", np.float32, {}),  # disabling as the result is trivial
        # ("absolute", np.float64, {}),  # disabling as the result is trivial
        ("absolute", np.complex64, {}),
        ("absolute", np.complex64, dict(use_native_absolute=True)),
        ("absolute", np.complex128, {}),
        ("acos", np.float32, {}),
        ("acos", np.float32, dict(use_native_acos=True)),
        ("acos", np.float64, {}),
        ("acos", np.complex64, {}),
        ("acos", np.complex64, dict(use_native_acos=True)),
        ("acos", np.complex128, {}),
        ("acosh", np.float32, {}),
        # *(("acosh", np.float32, dict(safe_max_limit_coefficient=v)) for v in [0.51, 0.5]),
        # *(("acosh", np.float64, dict(safe_max_limit_coefficient=v)) for v in [0.51, 0.5]),
        ("acosh", np.float32, dict(use_native_acosh=True)),
        ("acosh", np.float64, {}),
        ("acosh", np.complex64, {}),
        ("acosh", np.complex64, dict(use_native_acosh=True)),
        ("acosh", np.complex128, {}),
        ("asin", np.float32, {}),
        ("asin", np.float32, dict(use_native_asin=True)),
        ("asin", np.float64, {}),
        ("asin", np.complex64, {}),
        ("asin", np.complex64, dict(use_native_asin=True)),
        ("asin", np.complex128, {}),
        # *(("real_asinh", np.float32, dict(safe_min_limit=v)) for v in [None, 1, 10, 100, 1000]),
        # *(("real_asinh", np.float64, dict(safe_min_limit=v)) for v in [None, 1, 10, 100, 1000]),
        # *(("real_asinh", np.float32, dict(safe_max_limit_coefficient=v)) for v in [2, 1, 1/2, 1/4, 1/8]),
        # ("real_asinh_2", np.float32, {}),
        ("asinh", np.float32, {}),
        ("asinh", np.float32, dict(use_native_asinh=True)),
        ("asinh", np.float64, {}),
        ("asinh", np.complex64, {}),
        ("asinh", np.complex64, dict(use_native_asinh=True)),
        ("asinh", np.complex128, {}),
        # ("atan", np.float32, {}),  # real_atan is not implemented
        # ("atan", np.float64, {}),  # ditto
        ("atan", np.complex64, dict()),
        ("atan", np.complex64, dict(use_native_atan=True)),
        ("atan", np.complex128, {}),
        ("atanh", np.complex64, {}),
        ("atanh", np.complex64, dict(use_native_atanh=True)),
        ("atanh", np.complex128, {}),
        ("square", np.float32, {}),
        ("square", np.float32, dict(use_native_square=True)),
        ("square", np.float64, {}),
        ("square", np.complex64, {}),
        ("square", np.complex64, dict(use_native_square=True)),
        ("square", np.complex128, {}),
        # ("sqrt", np.float32, {}),     # real_sqrt is not implemented
        ("sqrt", np.float32, dict(use_native_sqrt=True)),
        # ("sqrt", np.float64, {}),     # real_sqrt is not implemented
        ("sqrt", np.complex64, dict(use_native_sqrt=True)),
        ("sqrt", np.complex64, {}),
        ("sqrt", np.complex128, {}),
        ("angle", np.complex64, {}),
        ("angle", np.complex128, {}),
        # ("log1p", np.float32, {}),  # real_log1p is not implemented
        # ("log1p", np.float64, {}),  # ditto
        ("log1p", np.complex64, {}),
        ("log1p", np.complex64, dict(use_fast2sum=True)),
        ("log1p", np.complex64, dict(use_native_log1p=True)),
        ("log1p", np.complex128, {}),
        # ("tan", np.float32, dict()),  # real_tan is not implemented
        ("tan", np.float32, dict(use_native_tan=True)),  # tan(x)
        ("tan", np.float32, dict(use_native_tan=True, use_upcast_tan=True)),  # float(tan(double(x)))
        ("real_naive_tan", np.float32, dict()),  # sin(x)/cos(x)
        (
            "real_naive_tan",
            np.float32,
            dict(use_upcast_sin=True, use_upcast_cos=True),
        ),  # float(sin(double(x))) / float(cos(double(x)))
        (
            "real_naive_tan",
            np.float32,
            dict(use_upcast_divide=True, use_upcast_sin=True, use_upcast_cos=True),
        ),  # float(tan(double(x)) / cos(double(x)))
        # ("tan", np.float64, {}),  # real_tan is not implemented
        # ("tan", np.complex64, {}),  # tan is not implemented
        ("tan", np.complex64, dict(use_native_tan=True)),
        ("tan", np.complex64, dict(use_native_tan=True, use_upcast_tan=True)),
        # ("tan", np.complex128, {}),  # tan is not implemented
        ("tanh", np.float32, dict(use_native_tanh=True)),
        ("tanh", np.float32, dict(use_native_tanh=True, use_upcast_tanh=True)),
        # ("tanh", np.float64, {}),  # real_tanh is not implemented
        # ("tanh", np.complex64, {}),  # tanh is not implemented
        # ("tanh", np.complex64, dict(use_upcast_tan=True, use_upcast_tanh=True, use_upcast_cos=True)),
        ("tanh", np.complex64, dict(use_native_tanh=True)),
        ("tanh", np.complex64, dict(use_native_tanh=True, use_upcast_tanh=True)),
        # ("tanh", np.complex128, {}),  # tanh is not implemented
    ]:
        validation_parameters = fa.utils.function_validation_parameters(func_name, dtype)
        max_bound_ulp_width = validation_parameters["max_bound_ulp_width"]

        def param2str(parameters):
            r = []
            for name, value in parameters.items():
                if name.startswith("use_"):
                    # skip use-parameters because these will be
                    # added to notes
                    continue
                r.append(f"{name}={value}")
            return "[" + ", ".join(r) + "]" if r else ""

        for max_bound_ulp_width in range(1 if max_bound_ulp_width else 0, max_bound_ulp_width + 1):
            func_counts[func_name, dtype] += 1
            func_count = func_counts[func_name, dtype]
            validation_parameters["max_bound_ulp_width"] = max_bound_ulp_width
            params = param2str(parameters)
            supscript = (
                f"<sup>{max_bound_ulp_width}</sup>"
                if max_bound_ulp_width > 0
                else (f"<sub>{func_count}</sub>" if func_count > 1 else "")
            )
            row_prefix = f"| {func_name}{supscript}{params} | {dtype.__name__} | "
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
    bucket_size = 50
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

Finally,
- "using native <function>" means "using the corresponding numpy <function>",
- "using upcast <function>" means that the function arguments are
  upcasted to a dtype with bits doubled, and the function results are
  downcasted to a dtype with bits split half.
""",
        file=f,
    )

    print(
        "| Function | dtype | dULP=0 (exact) | dULP=1 | dULP=2 | dULP=3 | dULP>3 | Notes |",
        file=f,
    )
    print("| -------- | ----- | -------------- | ------ | ------ | ------ | ------ | ----- |", file=f)

    for func_name, dtype, parameters, validation_parameters, row_prefix in get_inputs():
        if row_prefix in precomputed:
            print(f"{row_prefix}- using previous result.")
            print(precomputed[row_prefix], file=f)
            continue
        print(f"{row_prefix}")
        ctx = fa.Context(paths=[fa.algorithms], parameters=parameters)
        impl = getattr(fa.algorithms, func_name)

        if ctx.parameters.get(f"use_native_{func_name}", False):
            ctx.parameters["using"].add(f"native {func_name}")

            @functools.wraps(impl)
            def impl(ctx, *args, **kwargs):
                return getattr(ctx, func_name)(*args, **kwargs)

        if ctx.parameters.get(f"use_fast2sum", False):
            ctx.parameters["using"].add(f"fast2sum")

        graph = ctx.trace(impl, dtype)
        graph2 = graph.rewrite(
            fa.targets.numpy,  # implement missing functions
            ctx,  # applies use_upcast
            fa.rewrite,  # simplifies
        )
        func = fa.targets.numpy.as_function(graph2, debug=0)
        max_valid_ulp_count = validation_parameters["max_valid_ulp_count"]
        max_bound_ulp_width = validation_parameters["max_bound_ulp_width"]
        extra_prec_multiplier = validation_parameters["extra_prec_multiplier"]

        reference = getattr(
            fa.utils.numpy_with_mpmath(extra_prec_multiplier=extra_prec_multiplier, flush_subnormals=flush_subnormals),
            dict(
                real_asinh="asinh",
                real_asinh_2="asinh",
                acos="arccos",
                asin="arcsin",
                asinh="arcsinh",
                acosh="arccosh",
                real_naive_tan="tan",
            ).get(func_name, func_name),
        )

        if dtype in {np.complex64, np.complex128}:
            samples = fa.utils.complex_samples(
                (size, size), dtype=dtype, include_huge=True, include_subnormal=not flush_subnormals
            )
            step = size // bucket_size
            bucket = Bucket(samples[::step, ::step])
            samples = samples.flatten()
        else:
            samples = fa.utils.real_samples(size * size, dtype=dtype, include_subnormal=not flush_subnormals).flatten()
            bucket = None

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
            bucket=bucket,
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

        # notes column
        using = ctx.parameters.get("using")
        cols.append(f'using {", ".join(sorted(using))}' if using else "-")

        print(row_prefix + " | ".join(cols) + " |", file=f)
        print(row_prefix + " | ".join(cols) + " |")
        f.flush()

        if bucket is not None and bucket.ulp_count.sum() > 4:
            ulp_count = ((10 * bucket.ulp_count) // bucket.counts)[::-1]
            timage = fa.TextImage()
            timage.fill(0, 0, ulp_count <= 1000, symbol="C")
            timage.fill(0, 0, ulp_count <= 500, symbol="B")
            timage.fill(0, 0, ulp_count <= 100, symbol="A")
            timage.fill(0, 0, ulp_count == 0, symbol="=")
            for i in range(1, 10):
                timage.fill(0, 0, ulp_count == i, symbol=str(i))

            print(timage)

    print(f"Created {target_file}")


if __name__ == "__main__":
    main()
