import numpy as np
import functional_algorithms as fa

size = 1000
print("| Function | dtype | ULP=0 (exact) | ULP=1 | ULP=2 | ULP=3 | ULP>3 | errors    |")
print("| -------- | ----- | ------------- | ----- | ----- | ----- | ----- | --------- |")

for func_name, dtype in [
    ("absolute", np.float32),
    ("absolute", np.float64),
    ("absolute", np.complex64),
    ("absolute", np.complex128),
    ("asin", np.float32),
    ("asin", np.float64),
    ("asin", np.complex64),
    ("asin", np.complex128),
    ("square", np.float32),
    ("square", np.float64),
    ("square", np.complex64),
    ("square", np.complex128),
]:
    ctx = fa.Context(paths=[fa.algorithms])
    graph = ctx.trace(getattr(fa.algorithms, func_name), dtype)
    graph2 = graph.implement_missing(fa.targets.numpy).simplify()
    func = fa.targets.numpy.as_function(graph2, debug=0)

    if func_name == "asin":
        extra_prec_multiplier = 20
    else:
        extra_prec_multiplier = 1
    reference = getattr(fa.utils.numpy_with_mpmath(extra_prec_multiplier=extra_prec_multiplier), func_name)

    if dtype in {np.complex64, np.complex128}:
        samples = fa.utils.complex_samples((size, size), dtype=dtype, include_huge=True).flatten()
    else:
        samples = fa.utils.real_samples(size * size, dtype=dtype).flatten()
    matches_with_reference, ulp_stats = fa.utils.validate_function(func, reference, samples, dtype, verbose=False)
    # assert matches_with_reference, (func_name, dtype, ulp_stats)

    cols = [f"{func_name}", f"{dtype.__name__}"]
    total = sum(ulp_stats.values())
    for ulp in [0, 1, 2, 3]:
        if ulp not in ulp_stats:
            cols.append("-")
        else:
            cols.append(f"{100 * ulp_stats[ulp] / total:.3f} %")

    ulps = sum([ulp_stats[ulp] for ulp in ulp_stats if ulp > 3])
    if ulps:
        cols.append(f"{100 * ulps / total:.3f}")
    else:
        cols.append("-")
    if ulp_stats.get(-1, 0):
        cols.append(f"{ulp_stats[-1]}")
    else:
        cols.append(f"-")
    print("| " + " | ".join(cols) + " |")

print(f"Total number of samples is {size * size}")
