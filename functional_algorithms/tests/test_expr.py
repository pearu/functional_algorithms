import functional_algorithms as fa
import numpy


def test_constant():
    ctx = fa.Context()
    true = ctx.constant(True)
    assert true.get_type().kind == "boolean"
    assert true.get_type().bits == None

    x = ctx.constant(1.23)
    assert x.get_type().kind == "float"
    assert x.get_type().bits == None

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
    assert x.get_type().bits == None

    x = ctx.symbol("x", float)
    assert x.get_type().kind == "float"
    assert x.get_type().bits == None

    x = ctx.symbol("x", bool)
    assert x.get_type().kind == "boolean"
    assert x.get_type().bits == None


def test_symbol_boolean():
    ctx = fa.Context(default_constant_type=bool)

    x = ctx.symbol("x")
    assert x.get_type().kind == "boolean"
    assert x.get_type().bits == None

    x = ctx.symbol("x", float)
    assert x.get_type().kind == "float"
    assert x.get_type().bits == None


def test_symbol_numpy():
    ctx = fa.Context(default_constant_type=numpy.float32)

    x = ctx.symbol("x")
    assert x.get_type().kind == "float"
    assert x.get_type().bits == 32

    x = ctx.symbol("x", bool)
    assert x.get_type().kind == "boolean"
    assert x.get_type().bits == None


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
