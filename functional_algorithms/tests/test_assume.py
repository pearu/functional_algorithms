import functional_algorithms as fa

assert_equal = fa.expr.assert_equal


def test_assume():
    ctx = fa.Context(default_constant_type="float")

    x = ctx.symbol("x")
    y = ctx.symbol("y")
    z = ctx.symbol("z")

    fa.assume((x > 0) & ((y < z) | (y > 0)))
