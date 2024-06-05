from functional_algorithms import expr, Context


def test_normalize_like():

    ctx = Context(paths=[])
    x = ctx.symbol("x", "complex")
    assert expr.normalize_like(abs(x)) == "(abs x)"
