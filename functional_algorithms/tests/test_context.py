from functional_algorithms import Context


def test_populate():
    ctx = Context()

    x = ctx.symbol("x")
    ctx(x)

    asin = ctx.asin
    print(asin(x))
