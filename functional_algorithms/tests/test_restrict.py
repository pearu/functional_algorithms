import functional_algorithms as fa

assert_equal = fa.expr.assert_equal


def test_symbol():
    ctx = fa.Context(default_constant_type="float")
    x = ctx.symbol("x")
    e_d = list(fa.restrict(x))
    assert len(e_d) == 1
    e, d = e_d[0]
    assert_equal(e, x)
    assert_equal(d, ctx.constant(True))


def test_log1p():
    ctx = fa.Context(default_constant_type="float")
    x = ctx.symbol("x")
    expr = ctx.log1p(x)
    e_d = list(fa.restrict(expr))
    assert len(e_d) == 3
    e, d = e_d[0]
    assert_equal(e, x)
    assert_equal(d, (abs(x) < ctx.constant("eps") / 2).rewrite(fa.rewrite))

    e, d = e_d[1]
    assert_equal(e, ctx.log(x))
    assert_equal(d, (abs(x) * ctx.constant("eps") > ctx.constant(4)).rewrite(fa.rewrite))

    e, d = e_d[2]
    assert_equal(e, expr)


def test_add():
    ctx = fa.Context(default_constant_type="float")
    x = ctx.symbol("x")
    y = ctx.symbol("y")
    expr = x + y
    e_d = list(fa.restrict(expr))
    assert len(e_d) == 3

    e, d = e_d[0]
    assert_equal(e, x)
    assert_equal(d, (abs(x) * ctx.constant("eps") > 4 * abs(y)).rewrite(fa.rewrite))

    e, d = e_d[1]
    assert_equal(e, y)
    assert_equal(d, (abs(y) * ctx.constant("eps") > 4 * abs(x)).rewrite(fa.rewrite))

    e, d = e_d[2]
    assert_equal(e, expr)
