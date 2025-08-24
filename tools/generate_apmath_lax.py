def main():
    import os
    import numpy
    import functional_algorithms as fa
    import functional_algorithms.apmath_algorithms

    ctx = fa.Context(
        paths=[fa.apmath_algorithms],
        # enable_alt=True,
        # default_constant_type='dtype',
        # dtypes must be orderered starting with widest type
        parameters=dict(dtypes=[numpy.float64, numpy.float32, numpy.float16]),
    )
    target = fa.targets.lax

    output_fn = os.path.join(os.path.dirname(__file__), "..", "functional_algorithms", "apmath_lax.py")
    print(f"Generating {output_fn}")
    f = open(output_fn, "w")

    f.write(
        target.make_comment(
            f"""\
This file is generated using functional_algorithms tool ({fa.__version__}), see
  https://github.com/pearu/functional_algorithms
for more information."""
        )
        + "\n"
    )

    f.write(target.source_file_header + "\n")

    f.write("_np_dtypes = [" + ", ".join(["numpy." + dtype.__name__ for dtype in ctx.dtypes]) + "]\n")

    for func, args, kwargs in [
        (
            fa.apmath.two_sum,
            ("x:ArrayLike", "y:ArrayLike"),
            dict(fix_overflow=False, override_name="two_sum_unsafe", assume_fma=False),
        ),
        (
            fa.apmath.two_sum,
            ("x:ArrayLike", "y:ArrayLike"),
            dict(fix_overflow=True, override_name="two_sum_general", assume_fma=False),
        ),
        (
            fa.apmath.two_prod,
            ("x:ArrayLike", "y:ArrayLike"),
            dict(scale=False, fix_overflow=False, override_name="two_prod_unsafe", assume_fma=False),
        ),
        (
            fa.apmath.two_prod,
            ("x:ArrayLike", "y:ArrayLike"),
            dict(scale=True, fix_overflow=True, override_name="two_prod_general", assume_fma=False),
        ),
        (
            fa.apmath.fma,
            ("x:ArrayLike", "y:ArrayLike", "z:ArrayLike"),
            dict(
                fix_overflow=False,
                override_name="fma_unsafe",
                assume_fma=False,
                algorithm="apmath",
                functional=True,
                scale=False,
                size=None,
                possibly_zero_z=False,
            ),
        ),
        (
            fa.apmath.fma,
            ("x:ArrayLike", "y:ArrayLike", "z:ArrayLike"),
            dict(
                fix_overflow=True,
                override_name="fma_general",
                assume_fma=False,
                algorithm="a7",
                functional=True,
                scale=True,
                size=None,
                possibly_zero_z=True,
            ),
        ),
    ]:
        graph = ctx.trace(func, *args, **kwargs)
        doc = graph.props.get("__doc__")
        graph = graph.rewrite(target, fa.rewrite, fa.rewrite)
        if doc is not None:
            # rewrite may replace graph
            graph.props.update(__doc__=doc)
        doc = graph.props.get("__doc__")
        py = graph.tostring(target, tab="")
        f.write("\n\n" + py)

    f.close()


if __name__ == "__main__":
    main()
