import os
import functional_algorithms as fa

results_dir = os.path.dirname(__file__)

for target_name in dir(fa.targets):
    if target_name.startswith("_"):
        continue
    target = getattr(fa.targets, target_name)
    target_dir = os.path.join(results_dir, target_name)
    os.makedirs(target_dir, exist_ok=True)

    for func_name in dir(fa.algorithms):
        if func_name.startswith("_"):
            continue
        func = getattr(fa.algorithms, func_name)

        fn = os.path.join(target_dir, f"{func_name}{target.source_file_extension}")

        comment = target.make_comment(
            f"""
This file is generated using functional_algorithms tool ({fa.__version__}), see
  https://github.com/pearu/functional_algorithms
"""
        )

        ctx = fa.Context(paths=[fa.algorithms])
        graph = ctx.trace(func).implement_missing(target).simplify()
        src = graph.tostring(target)

        f = open(fn, "w")
        f.write(comment)
        f.write(src)
        f.close()
        print(f"Created {fn}")
