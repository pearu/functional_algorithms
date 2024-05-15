from functional_algorithms import expr, Context


def test_normalize_like():

    ctx = Context(paths=[])
    x = ctx.symbol("x", "complex")
    z = ctx.symbol("z")

    print(expr.normalize_like(abs(x)))
