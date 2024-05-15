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
for more information.
"""
        )

        sources = []
        for i, atypes in enumerate(target.trace_arguments[func_name]):
            ctx = fa.Context(paths=[fa.algorithms])
            graph = ctx.trace(func, *atypes).implement_missing(target).simplify()._props(name=f"{func_name}_{i}")
            src = graph.tostring(target)
            sources.append(src)

        f = open(fn, "w")
        f.write(comment + "\n")
        f.write(target.source_file_header + "\n")
        f.write("\n\n".join(sources))
        f.close()
        print(f"Created {fn}")
