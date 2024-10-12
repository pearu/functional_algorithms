import functional_algorithms as fa
import numpy


assert_equal = fa.expr.assert_equal


def test_constant():
    ctx = fa.Context()
    true = ctx.constant(True)
    assert true.get_type().kind == "boolean"
    assert true.get_type().bits is None

    x = ctx.constant(1.23)
    assert x.get_type().kind == "float"
    assert x.get_type().bits is None

    x = ctx.constant(numpy.float32(1.23))
    assert x.get_type().kind == "float"
    assert x.get_type().bits == 32

    x = ctx.constant(numpy.float64(1.23))
    assert x.get_type().kind == "float"
    assert x.get_type().bits == 64

    x = ctx.constant(numpy.complex64(1.23 + 2j))
    assert x.get_type().kind == "complex"
    assert x.get_type().bits == 64


def test_numpy_constant():
    ctx = fa.Context(default_constant_type=numpy.float32)

    x = ctx.constant(1.23)
    assert x.get_type().kind == "float"
    assert x.get_type().bits == 32

    v = numpy.float32(1.23)
    x = ctx.constant(v)
    assert x.get_type().kind == "float"
    assert x.get_type().bits == 32

    v = numpy.float64(1.23)
    x = ctx.constant(v)
    assert x.get_type().kind == "float"
    assert x.get_type().bits == 32

    x = ctx.constant(1)
    assert x.get_type().kind == "float"
    assert x.get_type().bits == 32

    x = ctx.constant(v, ctx.symbol(None, numpy.float64))
    assert x.get_type().kind == "float"
    assert x.get_type().bits == 64


def test_symbol():
    ctx = fa.Context()

    x = ctx.symbol("x")
    assert x.get_type().kind == "float"
    assert x.get_type().bits is None

    x = ctx.symbol("x", float)
    assert x.get_type().kind == "float"
    assert x.get_type().bits is None

    x = ctx.symbol("x", bool)
    assert x.get_type().kind == "boolean"
    assert x.get_type().bits is None


def test_symbol_boolean():
    ctx = fa.Context(default_constant_type=bool)

    x = ctx.symbol("x")
    assert x.get_type().kind == "boolean"
    assert x.get_type().bits is None

    x = ctx.symbol("x", float)
    assert x.get_type().kind == "float"
    assert x.get_type().bits is None


def test_symbol_numpy():
    ctx = fa.Context(default_constant_type=numpy.float32)

    x = ctx.symbol("x")
    assert x.get_type().kind == "float"
    assert x.get_type().bits == 32

    x = ctx.symbol("x", bool)
    assert x.get_type().kind == "boolean"
    assert x.get_type().bits is None


def test_rewrite():
    ctx = fa.Context()
    x = ctx.symbol("x")
    r = x + 0
    assert r.kind == "add"
    x1 = r.rewrite(fa.rewrite)
    assert x1 is x


def test_rewrite_numpy():
    ctx = fa.Context(default_constant_type=numpy.float32)
    x = ctx.constant(2)
    y = ctx.constant(numpy.float32(1))
    r = x + y
    assert r.kind == "add"
    r1 = r.rewrite(fa.rewrite)
    assert r1.kind == "constant"
    assert r1.operands[0] == 3

    r = ctx.sqrt(x)
    assert r.kind == "sqrt"
    r = r.rewrite(fa.rewrite)
    assert r.kind == "constant"
    assert r.operands[0] == numpy.sqrt(numpy.float32(x.operands[0]))

    r = ctx.sqrt(y)
    assert r.kind == "sqrt"
    r = r.rewrite(fa.rewrite)
    assert r.kind == "constant"
    assert r.operands[0] == numpy.sqrt(y.operands[0])


def test_substitute():
    ctx = fa.Context()

    x = ctx.symbol("x")
    y = ctx.symbol("y")
    c = ctx.constant(5, x)
    eps = ctx.constant("eps", x)

    def subs(expr, **dct):
        r = expr.rewrite(fa.rewrite.Substitute.fromdict(dct))
        return r

    assert subs(x, x=y) is y
    assert subs(x, x=1.2) is ctx.constant(1.2, x)
    assert subs(x, x=y) is y
    assert subs(x + x, x=y) is y + y
    assert subs(x + x, x=c) is c + c
    assert subs(x + y, x=c) is c + y
    assert subs(x + y, x=y, y=c) is y + c
    assert subs(x + y, y=c, x=y) is y + c
    assert subs(x, x="eps") is eps
    assert subs(x, x="a") is ctx.symbol("a", x.get_type())


def test_substitute_numpy():
    fi = numpy.finfo(numpy.float32)
    ctx = fa.Context(default_constant_type=numpy.float32)

    x = ctx.symbol("x")
    eps = ctx.constant("eps", x)

    def subs(expr, **dct):
        r = expr.rewrite(fa.rewrite.Substitute.fromdict(dct))
        # print(r)
        return r

    assert subs(x + eps, x=2) is ctx.constant(2, x) + eps
    assert subs(x + eps, x=2).rewrite(fa.rewrite) is ctx.constant(numpy.float32(2), x)
    assert subs(x + eps * 2, x=2).rewrite(fa.rewrite) is ctx.constant(numpy.float32(2 + 2 * fi.eps), x)
    assert subs(x, x="pi").rewrite(fa.rewrite) is ctx.constant(numpy.float32(numpy.pi), x)


def test_is_property():

    ctx = fa.Context()

    x = ctx.symbol("x")

    assert x._is("positive") is None
    assert x._is("nonpositive") is None
    assert x._is("negative") is None
    assert x._is("nonnegative") is None
    assert x._is("finite") is None
    assert x._is("zero") is None
    assert x._is("nonzero") is None
    assert x._is("one") is None

    for r in [ctx.square(x), x * x]:
        assert r._is("positive") is None
        assert r._is("nonpositive") is None
        assert r._is("negative") is False
        assert r._is("nonnegative") is True
        assert r._is("finite") is None
        assert r._is("zero") is None
        assert r._is("nonzero") is None
        assert r._is("one") is None

    r = ctx.constant(5, x)
    assert r._is("positive") is True
    assert r._is("nonpositive") is False
    assert r._is("negative") is False
    assert r._is("nonnegative") is True
    assert r._is("finite") is True
    assert r._is("zero") is False
    assert r._is("nonzero") is True
    assert r._is("one") is False

    r = ctx.constant("posinf", x)
    assert r._is("positive") is True
    assert r._is("nonpositive") is False
    assert r._is("negative") is False
    assert r._is("nonnegative") is True
    assert r._is("finite") is False
    assert r._is("zero") is False
    assert r._is("nonzero") is True
    assert r._is("one") is False

    r = ctx.constant(1, x)
    assert r._is("zero") is False
    assert r._is("nonzero") is True
    assert r._is("one") is True

    for name in ["smallest_subnormal", "smallest", "eps", "largest"]:
        r = ctx.constant(name, x)
        assert r._is("positive") is True
        assert r._is("nonpositive") is False
        assert r._is("negative") is False
        assert r._is("nonnegative") is True
        assert r._is("finite") is True
        assert r._is("zero") is False
        assert r._is("nonzero") is True
        assert r._is("one") is False

    r = ctx.constant(0, x)
    assert r._is("positive") is False
    assert r._is("nonpositive") is True
    assert r._is("negative") is False
    assert r._is("nonnegative") is True
    assert r._is("finite") is True
    assert r._is("zero") is True
    assert r._is("nonzero") is False
    assert r._is("one") is False

    r = ctx.constant(-5, x)
    assert r._is("positive") is False
    assert r._is("nonpositive") is True
    assert r._is("negative") is True
    assert r._is("nonnegative") is False
    assert r._is("finite") is True
    assert r._is("zero") is False
    assert r._is("nonzero") is True
    assert r._is("one") is False

    r = ctx.constant("neginf", x)
    assert r._is("positive") is False
    assert r._is("nonpositive") is True
    assert r._is("negative") is True
    assert r._is("nonnegative") is False
    assert r._is("finite") is False
    assert r._is("zero") is False
    assert r._is("nonzero") is True
    assert r._is("one") is False

    for name in ["undefined", "nan"]:
        r = ctx.constant("undefined", x)
        assert r._is("positive") is None
        assert r._is("nonpositive") is None
        assert r._is("negative") is None
        assert r._is("nonnegative") is None
        assert r._is("finite") is None
        assert r._is("zero") is None
        assert r._is("nonzero") is None
        assert r._is("one") is None

    r = ctx.constant(5, x) + ctx.constant("eps", x)
    assert r.kind == "add"
    assert r._is("positive") is True
    assert r._is("nonpositive") is False
    assert r._is("negative") is False
    assert r._is("nonnegative") is True
    assert r._is("finite") is True
    assert r._is("zero") is False
    assert r._is("nonzero") is True
    assert r._is("one") is None

    r = ctx.constant(-5, x) + ctx.constant("eps", x)
    assert r.kind == "add"
    assert r._is("positive") is None
    assert r._is("nonpositive") is None
    assert r._is("negative") is None
    assert r._is("nonnegative") is None
    assert r._is("finite") is True
    assert r._is("zero") is None
    assert r._is("nonzero") is None
    assert r._is("one") is None

    r = ctx.constant(5, x) - ctx.constant("eps", x)
    assert r.kind == "subtract"
    assert r._is("positive") is None
    assert r._is("nonpositive") is None
    assert r._is("negative") is None
    assert r._is("nonnegative") is None
    assert r._is("finite") is True
    assert r._is("zero") is None
    assert r._is("nonzero") is None
    assert r._is("one") is None

    r = ctx.constant(-5, x) - ctx.constant("eps", x)
    assert r.kind == "subtract"
    assert r._is("positive") is False
    assert r._is("nonpositive") is True
    assert r._is("negative") is True
    assert r._is("nonnegative") is False
    assert r._is("finite") is True
    assert r._is("zero") is False
    assert r._is("nonzero") is True
    assert r._is("one") is False

    r = ctx.constant("eps", x) - ctx.constant(-5, x)
    assert r.kind == "subtract"
    assert r._is("positive") is True
    assert r._is("nonpositive") is False
    assert r._is("negative") is False
    assert r._is("nonnegative") is True
    assert r._is("finite") is True
    assert r._is("zero") is False
    assert r._is("nonzero") is True
    assert r._is("one") is None

    for kind, r in [
        ("multiply", ctx.constant(5, x) * ctx.constant(6, x)),
        ("multiply", ctx.constant(-5, x) * ctx.constant(-6, x)),
        ("divide", ctx.constant(5, x) / ctx.constant(6, x)),
        ("divide", ctx.constant(-5, x) / ctx.constant(-6, x)),
    ]:
        assert r.kind == kind
        assert r._is("positive") is True
        assert r._is("nonpositive") is False
        assert r._is("negative") is False
        assert r._is("nonnegative") is True
        assert r._is("finite") is True
        assert r._is("zero") is False
        assert r._is("nonzero") is True
        assert r._is("one") is None

    for kind, r in [
        ("multiply", ctx.constant(-5, x) * ctx.constant(6, x)),
        ("multiply", ctx.constant(5, x) * ctx.constant(-6, x)),
        ("divide", ctx.constant(-5, x) / ctx.constant(6, x)),
        ("divide", ctx.constant(5, x) / ctx.constant(-6, x)),
    ]:
        assert r.kind == kind
        assert r._is("positive") is False
        assert r._is("nonpositive") is True
        assert r._is("negative") is True
        assert r._is("nonnegative") is False
        assert r._is("finite") is True
        assert r._is("zero") is False
        assert r._is("nonzero") is True
        assert r._is("one") is False

    for kind in ["square", "sqrt", "absolute"]:
        r = getattr(ctx, kind)(ctx.constant(5, x))
        assert r.kind == kind
        assert r._is("positive") is True, kind
        assert r._is("nonpositive") is False
        assert r._is("negative") is False
        assert r._is("nonnegative") is True
        assert r._is("finite") is True
        assert r._is("zero") is False
        assert r._is("nonzero") is True
        assert r._is("one") is False

    for kind in ["square", "sqrt", "absolute"]:
        r = getattr(ctx, kind)(ctx.constant(1, x))
        assert r.kind == kind
        assert r._is("zero") is False
        assert r._is("nonzero") is True
        assert r._is("one") is True

    for kind in ["square", "sqrt", "absolute"]:
        r = getattr(ctx, kind)(ctx.constant(0, x))
        assert r.kind == kind
        assert r._is("zero") is True
        assert r._is("nonzero") is False
        assert r._is("one") is False

    for kind in ["square", "absolute"]:
        r = getattr(ctx, kind)(ctx.constant(-5, x))
        assert r.kind == kind
        assert r._is("positive") is True, kind
        assert r._is("nonpositive") is False
        assert r._is("negative") is False
        assert r._is("nonnegative") is True
        assert r._is("finite") is True
        assert r._is("zero") is False
        assert r._is("nonzero") is True
        assert r._is("one") is False

    r = ctx.sqrt(ctx.constant(-5, x))
    assert r.kind == "sqrt"
    assert r._is("positive") is None
    assert r._is("nonpositive") is None
    assert r._is("negative") is None
    assert r._is("nonnegative") is None
    assert r._is("finite") is None
    assert r._is("zero") is None
    assert r._is("nonzero") is None
    assert r._is("one") is None


def test_rewrite_relop_consistency():

    for data in [fa.rewrite._constant_relop_constant, fa.rewrite._constant_relop_any, fa.rewrite._any_relop_any]:
        for (lhs, rhs), (ge, gt, le, lt, eq, ne) in data.items():
            # lhs rop rhs   is equivalent to    rhs swap_rop lhs
            swap_le, swap_lt, swap_ge, swap_gt, swap_eq, swap_ne = data[(rhs, lhs)]
            assert ge == swap_ge
            assert gt == swap_gt
            assert le == swap_le
            assert lt == swap_lt
            assert eq == swap_eq
            assert ne == swap_ne

            # lhs rop rhs   is equivalent to    not (lhs inverse_rop rhs)
            inverse_ge = lt
            inverse_gt = le
            inverse_le = gt
            inverse_lt = ge
            inverse_eq = ne
            inverse_ne = eq
            assert (inverse_ge is None and ge is None) or (inverse_ge is not None and ge is not None)
            assert ge is None or ge == (not inverse_ge)
            assert (inverse_gt is None and gt is None) or (inverse_gt is not None and gt is not None)
            assert gt is None or gt == (not inverse_gt)
            assert (inverse_le is None and le is None) or (inverse_le is not None and le is not None)
            assert le is None or le == (not inverse_le)
            assert (inverse_lt is None and lt is None) or (inverse_lt is not None and lt is not None)
            assert lt is None or lt == (not inverse_lt)
            assert (inverse_eq is None and eq is None) or (inverse_eq is not None and eq is not None)
            assert eq is None or eq == (not inverse_eq)
            assert (inverse_ne is None and ne is None) or (inverse_ne is not None and ne is not None)
            assert ne is None or ne == (not inverse_ne)


def test_relops():
    def rewrite(expr):
        r1 = expr.rewrite(fa.rewrite)
        expr2 = fa.Expr(
            expr.context, dict(lt="gt", le="ge", gt="lt", ge="le", eq="eq", ne="ne")[expr.kind], expr.operands[::-1]
        )
        r2_ = expr2.rewrite(fa.rewrite)
        if r2_.kind == "constant":
            r2 = r2_
        else:
            r2 = fa.Expr(
                expr.context, dict(lt="gt", le="ge", gt="lt", ge="le", eq="eq", ne="ne")[r2_.kind], r2_.operands[::-1]
            )
        assert r1 is r2
        return r1

    ctx = fa.Context()
    true = ctx.constant(True)
    false = ctx.constant(False)
    x = ctx.symbol("x")

    assert rewrite(x > 0).kind == "gt"
    assert rewrite(x >= 0).kind == "ge"
    assert rewrite(x <= 0).kind == "le"
    assert rewrite(x < 0).kind == "lt"
    assert rewrite(x == 0).kind == "eq"
    assert rewrite(x != 0).kind == "ne"
    assert rewrite(x > 1).kind == "gt"
    assert rewrite(x >= 1).kind == "ge"
    assert rewrite(x <= 1).kind == "le"
    assert rewrite(x < 1).kind == "lt"
    assert rewrite(x == 1).kind == "eq"
    assert rewrite(x != 1).kind == "ne"
    assert rewrite(x == 1).kind == "eq"

    for r in [ctx.square(x), x * x]:
        assert rewrite(r > 0).kind == "gt"
        assert rewrite(r >= 0) is true
        assert rewrite(r <= 0).kind == "le"
        assert rewrite(r < 0) is false
        assert rewrite(r == 0).kind == "eq"
        assert rewrite(r != 0).kind == "ne"
        assert rewrite(r == 1).kind == "eq"

    for r in [ctx.square(x), x * x]:
        assert rewrite(r > 1).kind == "gt"
        assert rewrite(r >= 1).kind == "ge"
        assert rewrite(r <= 1).kind == "le"
        assert rewrite(r < 1).kind == "lt"
        assert rewrite(r == 1).kind == "eq"
        assert rewrite(r != 1).kind == "ne"

    for r in [
        ctx.constant(1, x),
        ctx.constant(5, x),
        ctx.constant("posinf", x),
        ctx.constant("largest", x),
        ctx.constant("eps", x),
        ctx.constant("smallest", x),
        ctx.constant("smallest_subnormal", x),
    ]:
        assert rewrite(r > 0) is true
        assert rewrite(r >= 0) is true
        assert rewrite(r <= 0) is false
        assert rewrite(r < 0) is false
        assert rewrite(r == 0) is false
        assert rewrite(r != 0) is true

    for r in [
        ctx.constant(-5, x),
        ctx.constant("neginf", x),
    ]:
        assert rewrite(r > 0) is false
        assert rewrite(r >= 0) is false
        assert rewrite(r <= 0) is true
        assert rewrite(r < 0) is true
        assert rewrite(r == 0) is false
        assert rewrite(r != 0) is true
        assert rewrite(r > 1) is false
        assert rewrite(r >= 1) is false
        assert rewrite(r <= 1) is true
        assert rewrite(r < 1) is true
        assert rewrite(r == 1) is false
        assert rewrite(r != 1) is true

    for r in [
        ctx.constant(5, x),
        ctx.constant("posinf", x),
        ctx.constant("largest", x),
    ]:
        assert rewrite(r > 1) is true
        assert rewrite(r >= 1) is true
        assert rewrite(r <= 1) is false
        assert rewrite(r < 1) is false
        assert rewrite(r == 1) is false
        assert rewrite(r != 1) is true

    for r in [
        ctx.constant(0, x),
        ctx.constant("eps", x),
        ctx.constant("smallest", x),
        ctx.constant("smallest_subnormal", x),
    ]:
        assert rewrite(r > 1) is false
        assert rewrite(r >= 1) is false
        assert rewrite(r <= 1) is true
        assert rewrite(r < 1) is true
        assert rewrite(r == 1) is false
        assert rewrite(r != 1) is true

    r = ctx.constant(5, x) + ctx.constant("eps", x)
    assert r.kind == "add"
    assert rewrite(r > 0) is true
    assert rewrite(r >= 0) is true
    assert rewrite(r <= 0) is false
    assert rewrite(r < 0) is false
    assert rewrite(r == 0) is false
    assert rewrite(r != 0) is true

    r = ctx.constant(-5, x) + ctx.constant("eps", x)
    assert r.kind == "add"
    assert rewrite(r > 0).kind == "gt"
    assert rewrite(r >= 0).kind == "ge"
    assert rewrite(r <= 0).kind == "le"
    assert rewrite(r < 0).kind == "lt"
    assert rewrite(r == 0).kind == "eq"
    assert rewrite(r != 0).kind == "ne"

    r = ctx.constant(5, x) - ctx.constant("eps", x)
    assert r.kind == "subtract"
    assert rewrite(r > 0).kind == "gt"
    assert rewrite(r >= 0).kind == "ge"
    assert rewrite(r <= 0).kind == "le"
    assert rewrite(r < 0).kind == "lt"
    assert rewrite(r == 0).kind == "eq"
    assert rewrite(r != 0).kind == "ne"

    r = ctx.constant(-5, x) - ctx.constant("eps", x)
    assert r.kind == "subtract"
    assert rewrite(r > 0) is false
    assert rewrite(r >= 0) is false
    assert rewrite(r <= 0) is true
    assert rewrite(r < 0) is true
    assert rewrite(r == 0) is false
    assert rewrite(r != 0) is true

    for kind, r in [
        ("multiply", ctx.constant(5, x) * ctx.constant(6, x)),
        ("multiply", ctx.constant(-5, x) * ctx.constant(-6, x)),
        ("divide", ctx.constant(5, x) / ctx.constant(6, x)),
        ("divide", ctx.constant(-5, x) / ctx.constant(-6, x)),
    ]:
        assert r.kind == kind
        assert rewrite(r > 0) is true
        assert rewrite(r >= 0) is true
        assert rewrite(r <= 0) is false
        assert rewrite(r < 0) is false
        assert rewrite(r == 0) is false
        assert rewrite(r != 0) is true

    for kind, r in [
        ("multiply", ctx.constant(-5, x) * ctx.constant(6, x)),
        ("multiply", ctx.constant(5, x) * ctx.constant(-6, x)),
        ("divide", ctx.constant(-5, x) / ctx.constant(6, x)),
        ("divide", ctx.constant(5, x) / ctx.constant(-6, x)),
    ]:
        assert r.kind == kind
        assert rewrite(r > 0) is false
        assert rewrite(r >= 0) is false
        assert rewrite(r <= 0) is true
        assert rewrite(r < 0) is true
        assert rewrite(r == 0) is false
        assert rewrite(r == 1) is false
        assert rewrite(r != 0) is true

    for kind in ["square", "sqrt", "absolute"]:
        r = getattr(ctx, kind)(ctx.constant(5, x))
        assert r.kind == kind
        assert rewrite(r > 0) is true
        assert rewrite(r >= 0) is true
        assert rewrite(r <= 0) is false
        assert rewrite(r < 0) is false
        assert rewrite(r == 0) is false
        assert rewrite(r == 1) is false
        assert rewrite(r != 0) is true

        if kind != "sqrt":
            r = getattr(ctx, kind)(ctx.constant(-5, x))
            assert r.kind == kind
            assert rewrite(r > 0) is true
            assert rewrite(r >= 0) is true
            assert rewrite(r <= 0) is false
            assert rewrite(r < 0) is false
            assert rewrite(r == 0) is false
            assert rewrite(r == 1) is false
            assert rewrite(r != 0) is true

    for kind in ["square", "sqrt", "absolute"]:
        r = getattr(ctx, kind)(ctx.constant(1, x))
        assert r.kind == kind
        assert rewrite(r == 0) is false
        assert rewrite(r == 1) is true
        assert rewrite(r != 0) is true

        r = getattr(ctx, kind)(ctx.constant(0, x))
        assert r.kind == kind
        assert rewrite(r == 0) is true
        assert rewrite(r == 1) is false
        assert rewrite(r != 0) is false


def test_logical_op_flatten():
    ctx = fa.Context(default_constant_type="boolean")

    x = ctx.symbol("x")
    y = ctx.symbol("y")
    z = ctx.symbol("z")
    w = ctx.symbol("w")

    assert tuple(fa.rewrite.op_flatten(x, kind="logical_and")) == (x,)
    assert tuple(fa.rewrite.op_flatten(x & y)) == (x, y)
    assert tuple(fa.rewrite.op_flatten(x & y & x)) == (x, y, x)
    assert tuple(fa.rewrite.op_flatten(x & y & x & x)) == (x, y, x, x)

    assert tuple(fa.rewrite.op_flatten(x & y, commutative=True)) == (x, y)
    assert tuple(fa.rewrite.op_flatten(x & y & x, commutative=True)) == (x, x, y)
    assert tuple(fa.rewrite.op_flatten(x & y & x & x, commutative=True)) == (x, x, x, y)

    assert tuple(fa.rewrite.op_flatten(x & y, idempotent=True)) == (x, y)
    assert tuple(fa.rewrite.op_flatten(x & y & x, idempotent=True)) == (x, y, x)
    assert tuple(fa.rewrite.op_flatten(x & y & x & x, idempotent=True)) == (x, y, x)

    assert tuple(fa.rewrite.op_flatten(x & y, commutative=True, idempotent=True)) == (x, y)
    assert tuple(fa.rewrite.op_flatten(x & y & x, commutative=True, idempotent=True)) == (x, y)
    assert tuple(fa.rewrite.op_flatten(x & y & x & x, commutative=True, idempotent=True)) == (x, y)

    assert tuple(fa.rewrite.op_flatten(x, kind="logical_or")) == (x,)
    assert tuple(fa.rewrite.op_flatten(x | y)) == (x, y)
    assert tuple(fa.rewrite.op_flatten(x | y | x)) == (x, y, x)
    assert tuple(fa.rewrite.op_flatten(x | y | x | x)) == (x, y, x, x)

    assert tuple(fa.rewrite.op_flatten(x | y, commutative=True)) == (x, y)
    assert tuple(fa.rewrite.op_flatten(x | y | x, commutative=True)) == (x, x, y)
    assert tuple(fa.rewrite.op_flatten(x | y | x | x, commutative=True)) == (x, x, x, y)

    assert tuple(fa.rewrite.op_flatten(x | y, idempotent=True)) == (x, y)
    assert tuple(fa.rewrite.op_flatten(x | y | x, idempotent=True)) == (x, y, x)
    assert tuple(fa.rewrite.op_flatten(x | y | x | x, idempotent=True)) == (x, y, x)

    assert tuple(fa.rewrite.op_flatten(x | y, commutative=True, idempotent=True)) == (x, y)
    assert tuple(fa.rewrite.op_flatten(x | y | x, commutative=True, idempotent=True)) == (x, y)
    assert tuple(fa.rewrite.op_flatten(x | y | x | x, commutative=True, idempotent=True)) == (x, y)

    assert tuple(fa.rewrite.op_flatten((x | y) & z)) == (x | y, z)
    assert tuple(fa.rewrite.op_flatten((x & z) | (x & z))) == ((x & z), (x & z))


def test_logical_op_expand():
    ctx = fa.Context(default_constant_type="boolean")

    x = ctx.symbol("x")
    y = ctx.symbol("y")
    z = ctx.symbol("z")
    w = ctx.symbol("w")
    p = ctx.symbol("p")

    kwargs = dict(kind="logical_and")
    assert_equal(tuple(fa.rewrite.op_expand((x | y) & z, **kwargs)), (x & z, y & z))
    assert_equal(tuple(fa.rewrite.op_expand(w & (x | y), **kwargs)), (w & x, w & y))
    assert_equal(tuple(fa.rewrite.op_expand(w & (x | y) & z, **kwargs)), (w & x & z, w & y & z))
    assert_equal(tuple(fa.rewrite.op_expand((x | y) & (z | w), **kwargs)), (x & z, x & w, y & z, y & w))
    assert_equal(tuple(fa.rewrite.op_expand((x | y) & p & (z | w), **kwargs)), (x & p & z, x & p & w, y & p & z, y & p & w))

    assert_equal(tuple(fa.rewrite.op_expand((x | y) & x, **kwargs)), (x & x, y & x))
    assert_equal(tuple(fa.rewrite.op_expand((x | y) & x, idempotent=True, **kwargs)), (x, y & x))

    assert_equal(tuple(fa.rewrite.op_expand((x | y) & x & y, idempotent=True, **kwargs)), (x & y, y & x & y))
    assert_equal(tuple(fa.rewrite.op_expand((x | y) & x & y, idempotent=True, commutative=True, **kwargs)), (x & y, x & y))
    assert_equal(
        tuple(fa.rewrite.op_expand((x | y) & x & y, idempotent=True, commutative=True, over_idempotent=True, **kwargs)),
        (x & y,),
    )


def test_logical_op_collect():

    ctx = fa.Context(default_constant_type="boolean")

    x = ctx.symbol("x")
    y = ctx.symbol("y")
    z = ctx.symbol("z")
    w = ctx.symbol("w")

    kwargs = dict(over_kind="logical_or")
    assert_equal(fa.rewrite.op_collect((x & z) | (y & z), **kwargs), (x | y) & z)
    assert_equal(fa.rewrite.op_collect((z & x & z) | (y & z), **kwargs), ((z & x) | y) & z)
    assert_equal(fa.rewrite.op_collect((z & x & z) | (y & z), commutative=True, idempotent=True, **kwargs), (x | y) & z)

    assert_equal(fa.rewrite.op_collect((x & z) | (y & z) | z, **kwargs), (x | y) & z)
    assert_equal(fa.rewrite.op_collect((z & x) | (y & z), **kwargs), (z & x) | (y & z))
    assert_equal(fa.rewrite.op_collect((z & x) | (y & z), commutative=True, **kwargs), (x | y) & z)

    assert_equal(fa.rewrite.op_collect((x & z) | (y & z) | (x & z), **kwargs), (x | y | x) & z)
    assert_equal(fa.rewrite.op_collect((x & z) | (y & z) | (x & z), commutative=True, **kwargs), (x | y | x) & z)
    assert_equal(fa.rewrite.op_collect((x & z) | (y & z) | (x & z), over_commutative=True, **kwargs), (x | x | y) & z)
    assert_equal(
        fa.rewrite.op_collect((x & z) | (y & z) | (x & z), over_commutative=True, over_idempotent=True, **kwargs), (x | y) & z
    )
