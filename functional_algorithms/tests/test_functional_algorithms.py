from functional_algorithms import Context, targets, algorithms, utils


class TestImplementations:

    @staticmethod
    def square(ctx, x):
        if x.is_complex:
            x_sq = ctx.complex(ctx.square(x.real).reference("x_real_square") - ctx.square(x.imag), 2 * x.real * x.imag)
        else:
            x_sq = x * x
        return ctx(x_sq)

    @staticmethod
    def hypot(ctx, x, y):
        mx = ctx.maximum(abs(x), abs(y))
        mn = ctx.minimum(abs(x), abs(y))
        result = mx * ctx.sqrt(ctx.square(ctx.div(mn, mx, ref="mn_over_mx")) + 1)
        return ctx(result)

    @staticmethod
    def readme_square(ctx, z):
        if z.is_complex:
            x = abs(z.real)
            y = abs(z.imag)
            real = ctx.select(x == y, 0, ((x - y) * (y + y)).reference("real_part"))
            imag = 2 * (x * y)
            r = ctx.complex(real.reference(), imag.reference())
            return ctx(r)
        return z * z


def test_myhypot_stablehlo():

    ctx = Context(paths=[TestImplementations])

    graph = ctx.trace(TestImplementations.hypot)
    graph1 = graph.implement_missing(targets.stablehlo)
    graph1.props.update(name="CHLO_MyHypot")

    hlo = graph1.tostring(targets.stablehlo)

    assert (
        hlo
        == """\
def : Pat<(CHLO_MyHypot NonComplexElementType:$x, NonComplexElementType:$y),
  (StableHLO_MulOp
    (StableHLO_MaxOp:$mx
      (StableHLO_AbsOp:$abs_x $x),
      (StableHLO_AbsOp:$abs_y $y)),
    (StableHLO_SqrtOp
      (StableHLO_AddOp
        (StableHLO_MulOp
          (StableHLO_DivOp:$mn_over_mx
            (StableHLO_MinOp $abs_x, $abs_y),
            $mx),
          $mn_over_mx),
        (StableHLO_ConstantLike<"1"> $x))))>;"""
    )


def test_myhypot_python():

    ctx = Context(paths=[TestImplementations])

    graph = ctx.trace(TestImplementations.hypot)

    graph1 = graph.implement_missing(targets.python)
    graph1.props.update(name="myhypot")
    py = graph1.tostring(targets.python, tab="")

    print(py)

    assert py == utils.format_python(
        """\
def myhypot(x: float, y: float) -> float:
  abs_x: float = abs(x)
  abs_y: float = abs(y)
  mx: float = max(abs_x, abs_y)
  mn_over_mx: float = (min(abs_x, abs_y)) / (mx)
  return (mx) * (math.sqrt(((mn_over_mx) * (mn_over_mx)) + (1)))"""
    )


def test_square_python():

    ctx = Context(paths=[TestImplementations])

    graph = ctx.trace(TestImplementations.square, "x")

    graph1 = graph.implement_missing(targets.python)
    py = graph1.tostring(targets.python, tab="")

    print(py)

    assert py == utils.format_python(
        """\
def square(x: float) -> float:
  return (x) * (x)"""
    )


def test_complex_square_python():

    ctx = Context(paths=[TestImplementations])

    graph = ctx.trace(TestImplementations.square, complex)

    graph1 = graph.implement_missing(targets.python)
    py = graph1.tostring(targets.python, tab="")

    print(py)

    assert py == utils.format_python(
        """\
def square(x: complex) -> complex:
  _square_1_x: float = (x).real
  x_real_square: float = (_square_1_x) * (_square_1_x)
  _square_2_x: float = (x).imag
  return complex((x_real_square) - ((_square_2_x) * (_square_2_x)), ((2) * (_square_1_x)) * (_square_2_x))"""
    )


def test_complex_square_stablehlo():

    ctx = Context(paths=[TestImplementations])

    graph = ctx.trace(TestImplementations.square, complex)

    graph1 = graph.implement_missing(targets.stablehlo)
    shlo = graph1.tostring(targets.stablehlo, tab="")

    print(shlo)

    assert (
        shlo
        == """\
def : Pat<(CHLO_Square ComplexElementType:$x),
  (StableHLO_ComplexOp
    (StableHLO_SubtractOp
      (StableHLO_MulOp:$x_real_square
        (StableHLO_RealOp:$_square_1_x $x),
        $_square_1_x),
      (StableHLO_MulOp
        (StableHLO_ImagOp:$_square_2_x $x),
        $_square_2_x)),
    (StableHLO_MulOp
      (StableHLO_MulOp
        (StableHLO_ConstantLike<"2"> $_square_1_x),
        $_square_1_x),
      $_square_2_x))>;"""
    )


def test_readme_square_python():

    ctx = Context(paths=[TestImplementations])

    graph = ctx.trace(TestImplementations.readme_square, complex)

    print(graph)

    graph1 = graph.implement_missing(targets.python)
    py = graph1.tostring(targets.python, tab="")

    print(py)

    assert py == utils.format_python(
        """\
def readme_square(z: complex) -> complex:
  real_z: float = (z).real
  x: float = abs(real_z)
  y: float = abs((z).imag)
  real_part: float = ((x) - (y)) * ((y) + (y))
  real: float = (0) if ((x) == (y)) else (real_part)
  imag: float = (2) * ((x) * (y))
  return complex(real, imag)"""
    )
