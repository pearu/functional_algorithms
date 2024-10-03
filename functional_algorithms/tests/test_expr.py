import functional_algorithms as fa
import numpy


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
