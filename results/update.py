import os
import functional_algorithms as fa

results_dir = os.path.dirname(__file__)

for target_name in dir(fa.targets):
    if target_name.startswith("_") or target_name == "base":
        continue
    target = getattr(fa.targets, target_name)
    target_dir = os.path.join(results_dir, target_name)
    os.makedirs(target_dir, exist_ok=True)

    for func_name in dir(fa.algorithms):
        if func_name.startswith("_"):
            continue
        if func_name.startswith("complex_") or func_name.startswith("real_"):
            continue
        if func_name not in target.trace_arguments:
            print(f"Please update {target.__name__}.trace_arguments for function `{func_name}`")

    for func_name in target.trace_arguments:
        func = getattr(fa.algorithms, func_name)

        fn = os.path.join(target_dir, f"{func_name}{target.source_file_extension}")

        comment = target.make_comment(
            f"""\
This file is generated using functional_algorithms tool ({fa.__version__}), see
  https://github.com/pearu/functional_algorithms
for more information."""
        )
        enable_alt = False
        default_constant_type = None
        if target_name == "xla_client":
            enable_alt = True
            default_constant_type = "FloatType"

        sources = []
        for i, atypes in enumerate(target.trace_arguments[func_name]):
            ctx = fa.Context(paths=[fa.algorithms], enable_alt=enable_alt, default_constant_type=default_constant_type)
            graph = ctx.trace(func, *atypes).implement_missing(target).simplify()
            graph.props.update(name=f"{func_name}_{i}")
            src = graph.tostring(target)
            sources.append(src)
        src = "\n\n".join(sources)
        src = f"{target.source_file_header}\n\n{src}"

        if os.path.isfile(fn):
            f = open(fn, "r")
            old_src = f.read()
            f.close()
            create = not old_src.endswith(src)
        else:
            create = True

        if create:
            f = open(fn, "w")
            f.write(comment + "\n\n")
            f.write(src)
            f.close()
            print(f"Created {fn}")
        else:
            print(f"Skipped {fn}")

        if hasattr(target, "try_compile"):
            if target.try_compile(fn):
                print(f"Try compile {fn} PASSED")
            else:
                print(f"Try compile {fn} FAILED")
