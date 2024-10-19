import numpy
import functional_algorithms as fa

assert_equal = fa.expr.assert_equal


def rewrite(x):
    return x.rewrite(fa.rewrite)


def swap(expr):
    if expr.kind == "le":
        x, y = expr.operands
        return expr.context.ge(y, x)
    elif expr.kind == "ge":
        x, y = expr.operands
        return expr.context.le(y, x)
    elif expr.kind == "lt":
        x, y = expr.operands
        return expr.context.gt(y, x)
    elif expr.kind == "gt":
        x, y = expr.operands
        return expr.context.lt(y, x)
    elif expr.kind == "eq":
        x, y = expr.operands
        return expr.context.eq(y, x)
    elif expr.kind == "ne":
        x, y = expr.operands
        return expr.context.ne(y, x)
    return expr


def test_is_subset_of_numbers():
    is_subset_of = fa.assumptions.is_subset_of

    ctx = fa.Context(default_constant_type="float", compare_float_type=numpy.float64)
    true = ctx.constant(True)
    false = ctx.constant(False)
    x = ctx.symbol("x").reference("x")
    y = ctx.symbol("y").reference("y")
    zero = x.context.constant(0.0, x).reference("zero")

    a = ctx.symbol("a").reference("a")
    b = ctx.symbol("b").reference("b")

    assert_equal(is_subset_of(x > 0, x > 0), true)
    assert_equal(is_subset_of(x > 0, x < 0), false)
    assert_equal(is_subset_of(x > 0, x > 1), false)
    assert_equal(is_subset_of(x > 0, x < 1), x < 1.0)
    assert_equal(is_subset_of(x > 0, x > -1), true)
    assert_equal(is_subset_of(x > 0, x < -1), false)

    assert_equal(is_subset_of(x > 0, x >= 0), true)
    assert_equal(is_subset_of(x > 0, x <= 0), false)
    assert_equal(is_subset_of(x > 0, x >= 1), false)
    assert_equal(is_subset_of(x > 0, x <= 1), x <= 1.0)
    assert_equal(is_subset_of(x > 0, x >= -1), true)
    assert_equal(is_subset_of(x > 0, x <= -1), false)

    assert_equal(is_subset_of(x > 0, x == 0), false)
    assert_equal(is_subset_of(x > 0, x == 1), x == 1.0)
    assert_equal(is_subset_of(x > 0, x == -1), false)

    assert_equal(is_subset_of(x > 0, x != 0), true)
    assert_equal(is_subset_of(x > 0, x != 1), false)
    assert_equal(is_subset_of(x > 0, x != -1), true)

    assert_equal(is_subset_of(x > 0, x == x), true)
    assert_equal(is_subset_of(x > 0, x != x), false)

    assert_equal(is_subset_of(x >= 0, x > 0), false)
    assert_equal(is_subset_of(x >= 0, x < 0), false)
    assert_equal(is_subset_of(x >= 0, x > 1), false)
    assert_equal(is_subset_of(x >= 0, x < 1), x < 1.0)
    assert_equal(is_subset_of(x >= 0, x > -1), true)
    assert_equal(is_subset_of(x >= 0, x < -1), false)

    assert_equal(is_subset_of(x >= 0, x >= 0), true)
    assert_equal(is_subset_of(x >= 0, x <= 0), x == 0.0)
    assert_equal(is_subset_of(x >= 0, x >= 1), false)
    assert_equal(is_subset_of(x >= 0, x <= 1), x <= 1.0)
    assert_equal(is_subset_of(x >= 0, x >= -1), true)
    assert_equal(is_subset_of(x >= 0, x <= -1), false)

    assert_equal(is_subset_of(x >= 0, x == 0), x == 0.0)
    assert_equal(is_subset_of(x >= 0, x == 1), x == 1.0)
    assert_equal(is_subset_of(x >= 0, x == -1), false)

    assert_equal(is_subset_of(x >= 0, x != 0), false)
    assert_equal(is_subset_of(x >= 0, x != 1), false)
    assert_equal(is_subset_of(x >= 0, x != -1), true)

    assert_equal(is_subset_of(x >= 0, x == x), true)
    assert_equal(is_subset_of(x >= 0, x != x), false)

    assert_equal(is_subset_of(x < 0, x > 0), false)
    assert_equal(is_subset_of(x < 0, x < 0), true)
    assert_equal(is_subset_of(x < 0, x > 1), false)
    assert_equal(is_subset_of(x < 0, x < 1), true)
    assert_equal(is_subset_of(x < 0, x > -1), swap(x > -1.0))
    assert_equal(is_subset_of(x < 0, x < -1), false)

    assert_equal(is_subset_of(x < 0, x >= 0), false)
    assert_equal(is_subset_of(x < 0, x <= 0), true)
    assert_equal(is_subset_of(x < 0, x >= 1), false)
    assert_equal(is_subset_of(x < 0, x <= 1), true)
    assert_equal(is_subset_of(x < 0, x >= -1), swap(x >= -1.0))
    assert_equal(is_subset_of(x < 0, x <= -1), false)

    assert_equal(is_subset_of(x < 0, x == 0), false)
    assert_equal(is_subset_of(x < 0, x == 1), false)
    assert_equal(is_subset_of(x < 0, x == -1), x == -1.0)

    assert_equal(is_subset_of(x < 0, x != 0), true)
    assert_equal(is_subset_of(x < 0, x != 1), true)
    assert_equal(is_subset_of(x < 0, x != -1), false)

    assert_equal(is_subset_of(x < 0, x == x), true)
    assert_equal(is_subset_of(x < 0, x != x), false)

    assert_equal(is_subset_of(x <= 0, x > 0), false)
    assert_equal(is_subset_of(x <= 0, x < 0), false)
    assert_equal(is_subset_of(x <= 0, x > 1), false)
    assert_equal(is_subset_of(x <= 0, x < 1), true)
    assert_equal(is_subset_of(x <= 0, x > -1), swap(x > -1.0))
    assert_equal(is_subset_of(x <= 0, x < -1), false)

    assert_equal(is_subset_of(x <= 0, x >= 0), x == 0.0)
    assert_equal(is_subset_of(x <= 0, x <= 0), true)
    assert_equal(is_subset_of(x <= 0, x >= 1), false)
    assert_equal(is_subset_of(x <= 0, x <= 1), true)
    assert_equal(is_subset_of(x <= 0, x >= -1), swap(x >= -1.0))
    assert_equal(is_subset_of(x <= 0, x <= -1), false)

    assert_equal(is_subset_of(x <= 0, x == 0), x == 0.0)
    assert_equal(is_subset_of(x <= 0, x == 1), false)
    assert_equal(is_subset_of(x <= 0, x == -1), x == -1.0)

    assert_equal(is_subset_of(x <= 0, x != 0), false)
    assert_equal(is_subset_of(x <= 0, x != 1), true)
    assert_equal(is_subset_of(x <= 0, x != -1), false)

    assert_equal(is_subset_of(x <= 0, x == x), true)
    assert_equal(is_subset_of(x <= 0, x != x), false)

    assert_equal(is_subset_of(x == 0, x > 0), false)
    assert_equal(is_subset_of(x == 0, x < 0), false)
    assert_equal(is_subset_of(x == 0, x > 1), false)
    assert_equal(is_subset_of(x == 0, x < 1), true)
    assert_equal(is_subset_of(x == 0, x > -1), true)
    assert_equal(is_subset_of(x == 0, x < -1), false)

    assert_equal(is_subset_of(x == 0, x >= 0), true)
    assert_equal(is_subset_of(x == 0, x <= 0), true)
    assert_equal(is_subset_of(x == 0, x >= 1), false)
    assert_equal(is_subset_of(x == 0, x <= 1), true)
    assert_equal(is_subset_of(x == 0, x >= -1), true)
    assert_equal(is_subset_of(x == 0, x <= -1), false)

    assert_equal(is_subset_of(x == 0, x == 0), true)
    assert_equal(is_subset_of(x == 0, x == 1), false)
    assert_equal(is_subset_of(x == 0, x == -1), false)

    assert_equal(is_subset_of(x == 0, x != 0), false)
    assert_equal(is_subset_of(x == 0, x != 1), true)
    assert_equal(is_subset_of(x == 0, x != -1), true)

    assert_equal(is_subset_of(x == 0, x == x), true)
    assert_equal(is_subset_of(x == 0, x != x), false)

    assert_equal(is_subset_of(x != 0, x > 0), swap(x > 0.0))
    assert_equal(is_subset_of(x != 0, x < 0), x < 0.0)
    assert_equal(is_subset_of(x != 0, x > 1), swap(x > 1.0))
    assert_equal(is_subset_of(x != 0, x < 1), x < 1.0)
    assert_equal(is_subset_of(x != 0, x > -1), swap(x > -1.0))
    assert_equal(is_subset_of(x != 0, x < -1), x < -1.0)

    assert_equal(is_subset_of(x != 0, x >= 0), swap(x > 0.0))
    assert_equal(is_subset_of(x != 0, x <= 0), x < 0.0)
    assert_equal(is_subset_of(x != 0, x >= 1), swap(x >= 1.0))
    assert_equal(is_subset_of(x != 0, x <= 1), x <= 1.0)
    assert_equal(is_subset_of(x != 0, x >= -1), swap(x >= -1.0))
    assert_equal(is_subset_of(x != 0, x <= -1), x <= -1.0)

    assert_equal(is_subset_of(x != 0, x == 0), false)
    assert_equal(is_subset_of(x != 0, x == 1), x == 1.0)
    assert_equal(is_subset_of(x != 0, x == -1), x == -1.0)

    assert_equal(is_subset_of(x != 0, x != 0), true)
    assert_equal(is_subset_of(x != 0, x != 1), x != 1.0)
    assert_equal(is_subset_of(x != 0, x != -1), x != -1.0)

    assert_equal(is_subset_of(x != 0, x == x), true)
    assert_equal(is_subset_of(x != 0, x != x), false)

    assert_equal(is_subset_of(x > a, x > b), (b <= a) & (b != ctx.constant("posinf", b)))
    assert_equal(is_subset_of(x > a, x > 0), swap(a >= ctx.constant(0.0, x)))
    assert_equal(is_subset_of(x < a, x < b), (a <= b) & (b != ctx.constant("neginf", b)))

    assert_equal(is_subset_of(x <= 0, x < 0), false)
    assert_equal(is_subset_of(x >= a, x > b), b < a)

    assert_equal(is_subset_of((a < x) & (a < b), x > 0), swap(a >= zero))

    assert_equal(is_subset_of((1 < x) & (x < 3), x > 0), true)
    assert_equal(is_subset_of((1 < x) & (x < 3), (1 < x) & (x < 5)), true)
    assert_equal(is_subset_of((1 <= x) & (x < 3), (1 < x) & (x < 5)), false)
    assert_equal(is_subset_of((1 < x) & (x < 3), (0 < x) & (x < 2)), false)


def test_is_subset_of_symbolic():
    is_subset_of = fa.assumptions.is_subset_of

    ctx = fa.Context(default_constant_type="float", compare_float_type=numpy.float64)
    true = ctx.constant(True)
    false = ctx.constant(False)
    x = ctx.symbol("x").reference("x")
    y = ctx.symbol("y").reference("y")
    zero = x.context.constant(0.0, x).reference("zero")
    eps = x.context.constant("eps", x).reference("eps")
    pi = x.context.constant("pi", x).reference("pi")
    smallest = x.context.constant("smallest", x).reference("smallest")
    largest = x.context.constant("largest", x).reference("largest")
    posinf = x.context.constant("posinf", x).reference("posinf")
    neginf = x.context.constant("neginf", x).reference("neginf")

    assert_equal(is_subset_of(x == smallest, x > 0), true)
    assert_equal(is_subset_of(x == smallest, x > smallest), false)
    assert_equal(is_subset_of(x == smallest, x > eps), false)
    assert_equal(is_subset_of(x == largest, x > eps), true)
    assert_equal(is_subset_of(x == posinf, x > eps), true)
    assert_equal(is_subset_of(x == neginf, x > eps), false)
    assert_equal(is_subset_of(x > eps, x >= eps), true)
    assert_equal(is_subset_of(x <= eps, x > -1), ctx.constant(-1.0, x) < x)
    assert_equal(is_subset_of(x >= eps, x < 1), x < 1.0)
    assert_equal(is_subset_of(x >= smallest, x < eps), x < eps)

    for c in [
        largest,
        ctx.constant(10, x),
        pi,
        eps,
        smallest,
        ctx.constant(0, x),
        -smallest,
        -eps,
        ctx.constant(-10, x),
        -largest,
        posinf,
        neginf,
    ]:
        for d in [
            largest,
            ctx.constant(20, x),
            pi,
            eps,
            smallest,
            ctx.constant(0, x),
            -smallest,
            -eps,
            ctx.constant(-10, x),
            -largest,
            posinf,
            neginf,
        ]:
            #
            assert_equal(is_subset_of(x < c, x < d), rewrite(false if d._is_neginf else c <= d))
            assert_equal(is_subset_of(x > -c, x > -d), rewrite(false if d._is_neginf else c <= d))
            assert_equal(is_subset_of(x < c, d < x), false if d._is_neginf else rewrite((d < x) & (c > d)))
            assert_equal(is_subset_of(x > -c, -d > x), false if d._is_neginf else rewrite((x < -d) & (c > d)))
            assert_equal(is_subset_of(c < x, x < d), false if d._is_posinf else rewrite((x < d) & (c < d)))
            assert_equal(is_subset_of(-c > x, x > -d), false if d._is_posinf else rewrite((-d < x) & (c < d)))
            assert_equal(is_subset_of(c < x, x > d), false if d._is_posinf else rewrite(c >= d))
            assert_equal(is_subset_of(-c > x, x < -d), false if d._is_posinf else rewrite(c >= d))
        #
        assert_equal(is_subset_of(x < c, x < posinf), true)
        assert_equal(is_subset_of(x > -c, x > neginf), true)
        assert_equal(is_subset_of(x < c, posinf < x), false)
        assert_equal(is_subset_of(x > -c, neginf > x), false)
        assert_equal(is_subset_of(c < x, x < posinf), false)
        assert_equal(is_subset_of(-c > x, x > neginf), false)
        assert_equal(is_subset_of(c < x, posinf < x), false)
        assert_equal(is_subset_of(-c > x, neginf > x), false)
        #
        assert_equal(is_subset_of(x < c, x <= posinf), true)
        assert_equal(is_subset_of(x > -c, x >= neginf), true)
        assert_equal(is_subset_of(x < c, posinf <= x), false)
        assert_equal(is_subset_of(x > -c, neginf >= x), false)
        assert_equal(is_subset_of(c < x, x <= posinf), true)
        assert_equal(is_subset_of(-c > x, x >= neginf), true)
        assert_equal(is_subset_of(c < x, posinf <= x), true)
        assert_equal(is_subset_of(-c > x, neginf >= x), true)
        #
        assert_equal(is_subset_of(x < c, x == posinf), false)
        assert_equal(is_subset_of(x > -c, x == neginf), false)
        assert_equal(is_subset_of(x < c, posinf == x), false)
        assert_equal(is_subset_of(x > -c, neginf == x), false)
        assert_equal(is_subset_of(c < x, x == posinf), false)
        assert_equal(is_subset_of(-c > x, x == neginf), false)
        assert_equal(is_subset_of(c < x, posinf == x), false)
        assert_equal(is_subset_of(-c > x, neginf == x), false)
        #
        assert_equal(is_subset_of(x < c, x != posinf), true)
        assert_equal(is_subset_of(x > -c, x != neginf), true)
        assert_equal(is_subset_of(x < c, posinf != x), true)
        assert_equal(is_subset_of(x > -c, neginf != x), true)
        assert_equal(is_subset_of(c < x, x != posinf), false if c is largest else true)
        assert_equal(is_subset_of(-c > x, x != neginf), false if c is largest else true)
        assert_equal(is_subset_of(c < x, posinf != x), false if c is largest else true)
        assert_equal(is_subset_of(-c > x, neginf != x), false if c is largest else true)
        #
        assert_equal(is_subset_of(x <= c, x < posinf), false if c._is_posinf else true)
        assert_equal(is_subset_of(x >= -c, x > neginf), false if c._is_posinf else true)
        assert_equal(is_subset_of(x <= c, posinf < x), false)
        assert_equal(is_subset_of(x >= -c, neginf > x), false)
        assert_equal(is_subset_of(c <= x, x < posinf), false)
        assert_equal(is_subset_of(-c >= x, x > neginf), false)
        assert_equal(is_subset_of(c <= x, posinf < x), false)
        assert_equal(is_subset_of(-c >= x, neginf > x), false)
        #
        assert_equal(is_subset_of(x <= c, x <= posinf), true)
        assert_equal(is_subset_of(x >= -c, x >= neginf), true)
        assert_equal(is_subset_of(x <= c, posinf <= x), x == posinf if c._is_posinf else false)
        assert_equal(is_subset_of(x >= -c, neginf >= x), x == neginf if c._is_posinf else false)
        assert_equal(is_subset_of(c <= x, x <= posinf), true)
        assert_equal(is_subset_of(-c >= x, x >= neginf), true)
        assert_equal(is_subset_of(c <= x, posinf <= x), true if c._is_posinf else false)
        assert_equal(is_subset_of(-c >= x, neginf >= x), true if c._is_posinf else false)
        #
        assert_equal(is_subset_of(x <= c, x == posinf), x == posinf if c._is_posinf else false)
        assert_equal(is_subset_of(x >= -c, x == neginf), x == neginf if c._is_posinf else false)
        assert_equal(is_subset_of(x <= c, posinf == x), x == posinf if c._is_posinf else false)
        assert_equal(is_subset_of(x >= -c, neginf == x), x == neginf if c._is_posinf else false)
        assert_equal(is_subset_of(c <= x, x == posinf), true if c._is_posinf else x == posinf)
        assert_equal(is_subset_of(-c >= x, x == neginf), true if c._is_posinf else x == neginf)
        assert_equal(is_subset_of(c <= x, posinf == x), true if c._is_posinf else x == posinf)
        assert_equal(is_subset_of(-c >= x, neginf == x), true if c._is_posinf else x == neginf)
        #
        assert_equal(is_subset_of(x <= c, x != posinf), false if c._is_posinf else true)
        assert_equal(is_subset_of(x >= -c, x != neginf), false if c._is_posinf else true)
        assert_equal(is_subset_of(x <= c, posinf != x), false if c._is_posinf else true)
        assert_equal(is_subset_of(x >= -c, neginf != x), false if c._is_posinf else true)
        assert_equal(is_subset_of(c <= x, x != posinf), false)
        assert_equal(is_subset_of(-c >= x, x != neginf), false)
        assert_equal(is_subset_of(c <= x, posinf != x), false)
        assert_equal(is_subset_of(-c >= x, neginf != x), false)
        #
        assert_equal(is_subset_of(x == c, x < posinf), false if c._is_posinf else true)
        assert_equal(is_subset_of(x == -c, x > neginf), false if c._is_posinf else true)
        assert_equal(is_subset_of(x == c, posinf < x), false)
        assert_equal(is_subset_of(x == -c, neginf > x), false)
        assert_equal(is_subset_of(c == x, x < posinf), false if c._is_posinf else true)
        assert_equal(is_subset_of(-c == x, x > neginf), false if c._is_posinf else true)
        assert_equal(is_subset_of(c == x, posinf < x), false)
        assert_equal(is_subset_of(-c == x, neginf > x), false)
        #
        assert_equal(is_subset_of(x == c, x <= posinf), true)
        assert_equal(is_subset_of(x == -c, x >= neginf), true)
        assert_equal(is_subset_of(x == c, posinf <= x), true if c._is_posinf else false)
        assert_equal(is_subset_of(x == -c, neginf >= x), true if c._is_posinf else false)
        assert_equal(is_subset_of(c == x, x <= posinf), true)
        assert_equal(is_subset_of(-c == x, x >= neginf), true)
        assert_equal(is_subset_of(c == x, posinf <= x), true if c._is_posinf else false)
        assert_equal(is_subset_of(-c == x, neginf >= x), true if c._is_posinf else false)
        #
        assert_equal(is_subset_of(x == c, x == posinf), true if c._is_posinf else false)
        assert_equal(is_subset_of(x == -c, x == neginf), true if c._is_posinf else false)
        assert_equal(is_subset_of(x == c, posinf == x), true if c._is_posinf else false)
        assert_equal(is_subset_of(x == -c, neginf == x), true if c._is_posinf else false)
        assert_equal(is_subset_of(c == x, x == posinf), true if c._is_posinf else false)
        assert_equal(is_subset_of(-c == x, x == neginf), true if c._is_posinf else false)
        assert_equal(is_subset_of(c == x, posinf == x), true if c._is_posinf else false)
        assert_equal(is_subset_of(-c == x, neginf == x), true if c._is_posinf else false)
        #
        assert_equal(is_subset_of(x == c, x != posinf), false if c._is_posinf else true)
        assert_equal(is_subset_of(x == -c, x != neginf), false if c._is_posinf else true)
        assert_equal(is_subset_of(x == c, posinf != x), false if c._is_posinf else true)
        assert_equal(is_subset_of(x == -c, neginf != x), false if c._is_posinf else true)
        assert_equal(is_subset_of(c == x, x != posinf), false if c._is_posinf else true)
        assert_equal(is_subset_of(-c == x, x != neginf), false if c._is_posinf else true)
        assert_equal(is_subset_of(c == x, posinf != x), false if c._is_posinf else true)
        assert_equal(is_subset_of(-c == x, neginf != x), false if c._is_posinf else true)
        #
        assert_equal(is_subset_of(x != c, x < posinf), true if c._is_posinf else x < posinf)
        assert_equal(is_subset_of(x != -c, x > neginf), true if c._is_posinf else neginf < x)
        assert_equal(is_subset_of(x != c, posinf < x), false)
        assert_equal(is_subset_of(x != -c, neginf > x), false)
        assert_equal(is_subset_of(c != x, x < posinf), true if c._is_posinf else x < posinf)
        assert_equal(is_subset_of(-c != x, x > neginf), true if c._is_posinf else neginf < x)
        assert_equal(is_subset_of(c != x, posinf < x), false)
        assert_equal(is_subset_of(-c != x, neginf > x), false)
        #
        assert_equal(is_subset_of(x != c, x <= posinf), true)
        assert_equal(is_subset_of(x != -c, x >= neginf), true)
        assert_equal(is_subset_of(x != c, posinf <= x), false)
        assert_equal(is_subset_of(x != -c, neginf >= x), false)
        assert_equal(is_subset_of(c != x, x <= posinf), true)
        assert_equal(is_subset_of(-c != x, x >= neginf), true)
        assert_equal(is_subset_of(c != x, posinf <= x), false)
        assert_equal(is_subset_of(-c != x, neginf >= x), false)
        #
        assert_equal(is_subset_of(x != c, x == posinf), false if c._is_posinf else x == posinf)
        assert_equal(is_subset_of(x != -c, x == neginf), false if c._is_posinf else x == neginf)
        assert_equal(is_subset_of(x != c, posinf == x), false if c._is_posinf else x == posinf)
        assert_equal(is_subset_of(x != -c, neginf == x), false if c._is_posinf else x == neginf)
        assert_equal(is_subset_of(c != x, x == posinf), false if c._is_posinf else x == posinf)
        assert_equal(is_subset_of(-c != x, x == neginf), false if c._is_posinf else x == neginf)
        assert_equal(is_subset_of(c != x, posinf == x), false if c._is_posinf else x == posinf)
        assert_equal(is_subset_of(-c != x, neginf == x), false if c._is_posinf else x == neginf)
        #
        assert_equal(is_subset_of(x != c, x != posinf), true if c._is_posinf else x != posinf)
        assert_equal(is_subset_of(x != -c, x != neginf), true if c._is_posinf else x != neginf)
        assert_equal(is_subset_of(x != c, posinf != x), true if c._is_posinf else x != posinf)
        assert_equal(is_subset_of(x != -c, neginf != x), true if c._is_posinf else x != neginf)
        assert_equal(is_subset_of(c != x, x != posinf), true if c._is_posinf else x != posinf)
        assert_equal(is_subset_of(-c != x, x != neginf), true if c._is_posinf else x != neginf)
        assert_equal(is_subset_of(c != x, posinf != x), true if c._is_posinf else x != posinf)
        assert_equal(is_subset_of(-c != x, neginf != x), true if c._is_posinf else x != neginf)


def test_register_assumptions():
    ctx = fa.Context(default_constant_type="float")

    x = ctx.symbol("x")
    y = ctx.symbol("y")

    fa.assume(((x > 0) | (x < 1)) & (x * x < 1) & (y > 2))

    assert (x * x < 1).key in x.props["_assumptions"]
    assert ((x > 0) | (x < 1)).key in x.props["_assumptions"]
    assert (y > 2).key in y.props["_assumptions"]
    assert (y > 2).key not in x.props["_assumptions"]

    fa.assume(x + y < 2)

    assert (x * x < 1).key in x.props["_assumptions"]
    assert (x + y < 2).key in x.props["_assumptions"]
    assert (x + y < 2).key in y.props["_assumptions"]


def test_use_assumptions():

    ctx = fa.Context(default_constant_type="float")
    x = ctx.symbol("x")

    assert fa.check(x > 0) is None

    fa.assume(x > 0)

    assert fa.check(x > -1) is False
    assert fa.check(x > 0) is True
    assert fa.check(x > 1) is True
    assert fa.check(x >= 0) is False
    assert fa.check(x < 0) is False
    assert fa.check(x < 1) is None


def test_abs_check():

    ctx = fa.Context(default_constant_type="float")
    x = ctx.symbol("x")

    assert fa.check(abs(x) >= 0) is True
    assert fa.check(abs(x) >= 1) is True
    assert fa.check(abs(x) < 1) is True
    assert fa.check(abs(x) < 0) is False
    assert fa.check(abs(x) < -1) is False
    assert fa.check(abs(x) == 0) is True
    assert fa.check(abs(x) == -1) is False
    assert fa.check(abs(x) == 1) is True
    assert fa.check(abs(x) != 1) is True
    assert fa.check(abs(x) == -1) is False


def test_assume_nonlogical():

    ctx = fa.Context(default_constant_type="float")
    x = ctx.symbol("x")

    # assert fa.assume(abs(x)) == [abs(x) >= 0]
    fa.assume(abs(x) < ctx.constant("posinf", x))
    d = fa.assume(abs(x) * abs(x))
    for item in d:
        print(111, item.tostring(fa.targets.symbolic))
    return

    assert fa.assume(x) == []
    assert fa.assume(abs(x)) == [abs(x) >= 0]
