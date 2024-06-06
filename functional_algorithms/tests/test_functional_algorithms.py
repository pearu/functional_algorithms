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
        result = mx * ctx.sqrt(ctx.square(ctx.div(mn, mx).reference("mn_over_mx")) + 1)
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

    @staticmethod
    def safe_min(ctx, x):
        m = ctx.alt.constant("smallest")
        return ctx.constant(ctx.alt.sqrt(m) / 4, x)


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

    assert py == utils.format_python(
        """\
def myhypot(x: float, y: float) -> float:
  abs_x: float = abs(x)
  abs_y: float = abs(y)
  mx: float = max(abs_x, abs_y)
  mn_over_mx: float = (min(abs_x, abs_y)) / (mx)
  return (mx) * (math.sqrt(((mn_over_mx) * (mn_over_mx)) + (1)))"""
    )


def test_myhypot_xla_client():

    ctx = Context(paths=[TestImplementations])
    constant_ctx = Context()

    graph = ctx.trace(TestImplementations.hypot)

    graph1 = graph.implement_missing(targets.xla_client)

    graph1.props.update(name="myhypot")
    py = graph1.tostring(targets.xla_client, tab="")

    assert py == utils.format_cpp(
        """\
XlaOp myhypot(XlaOp x, XlaOp y) {
  XlaOp abs_x = Abs(x);
  XlaOp abs_y = Abs(y);
  XlaOp mx = Max(abs_x, abs_y);
  XlaOp mn_over_mx = Div(Min(abs_x, abs_y), mx);
  return Mul(mx, Sqrt(Add(Square(mn_over_mx), ScalarLike(x, 1))));
}"""
    )


def test_myhypot_cpp():

    ctx = Context(paths=[TestImplementations])

    graph = ctx.trace(TestImplementations.hypot)

    graph1 = graph.implement_missing(targets.cpp)
    graph1.props.update(name="myhypot")
    py = graph1.tostring(targets.cpp, tab="")

    assert py == utils.format_cpp(
        """\
double myhypot(double x, double y) {
  double abs_x = std::abs(x);
  double abs_y = std::abs(y);
  double mx = std::max(abs_x, abs_y);
  double mn_over_mx = (std::min(abs_x, abs_y)) / (mx);
  return (mx) * (std::sqrt(((mn_over_mx) * (mn_over_mx)) + (1)));
}"""
    )


def test_square_python():

    ctx = Context(paths=[TestImplementations])

    graph = ctx.trace(TestImplementations.square, "x")

    graph1 = graph.implement_missing(targets.python)
    py = graph1.tostring(targets.python, tab="")

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

    assert py == utils.format_python(
        """\
def square(x: complex) -> complex:
  real_x: float = (x).real
  x_real_square: float = (real_x) * (real_x)
  imag_x: float = (x).imag
  return complex((x_real_square) - ((imag_x) * (imag_x)), ((2) * (real_x)) * (imag_x))"""
    )


def test_complex_square_stablehlo():

    ctx = Context(paths=[TestImplementations])

    graph = ctx.trace(TestImplementations.square, complex)

    graph1 = graph.implement_missing(targets.stablehlo)
    shlo = graph1.tostring(targets.stablehlo, tab="")

    assert (
        shlo
        == """\
def : Pat<(CHLO_Square ComplexElementType:$x),
  (StableHLO_ComplexOp
    (StableHLO_SubtractOp
      (StableHLO_MulOp:$x_real_square
        (StableHLO_RealOp:$real_x $x),
        $real_x),
      (StableHLO_MulOp
        (StableHLO_ImagOp:$imag_x $x),
        $imag_x)),
    (StableHLO_MulOp
      (StableHLO_MulOp
        (StableHLO_ConstantLike<"2"> $real_x),
        $real_x),
      $imag_x))>;"""
    )


def test_readme_square_python():

    ctx = Context(paths=[TestImplementations])

    graph = ctx.trace(TestImplementations.readme_square, complex)

    graph1 = graph.implement_missing(targets.python)
    py = graph1.tostring(targets.python, tab="")

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


def test_readme_square_numpy_debug_0():

    ctx = Context(paths=[TestImplementations])

    graph = ctx.trace(TestImplementations.readme_square, complex)
    graph1 = graph.implement_missing(targets.numpy)
    py = graph1.tostring(targets.numpy, tab="", debug=0)

    assert py == utils.format_python(
        """\
def readme_square(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        real_z: numpy.float64 = (z).real
        x: numpy.float64 = numpy.abs(real_z)
        y: numpy.float64 = numpy.abs((z).imag)
        real_part: numpy.float64 = ((x) - (y)) * ((y) + (y))
        real: numpy.float64 = (numpy.float64(0)) if (numpy.equal(x, y, dtype=numpy.bool_)) else (real_part)
        imag: numpy.float64 = (numpy.float64(2)) * ((x) * (y))
        result = make_complex(real, imag)
        return result"""
    )


def test_readme_square_numpy_debug_1():

    ctx = Context(paths=[TestImplementations])

    graph = ctx.trace(TestImplementations.readme_square, complex)
    graph1 = graph.implement_missing(targets.numpy)
    py = graph1.tostring(targets.numpy, tab="", debug=1)

    assert py == utils.format_python(
        """\
def readme_square(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        real_z: numpy.float64 = (z).real
        assert real_z.dtype == numpy.float64, (real_z.dtype, numpy.float64)
        x: numpy.float64 = numpy.abs(real_z)
        assert x.dtype == numpy.float64, (x.dtype, numpy.float64)
        y: numpy.float64 = numpy.abs((z).imag)
        assert y.dtype == numpy.float64, (y.dtype, numpy.float64)
        real_part: numpy.float64 = ((x) - (y)) * ((y) + (y))
        assert real_part.dtype == numpy.float64, (real_part.dtype, numpy.float64)
        real: numpy.float64 = (numpy.float64(0)) if (numpy.equal(x, y, dtype=numpy.bool_)) else (real_part)
        assert real.dtype == numpy.float64, (real.dtype, numpy.float64)
        imag: numpy.float64 = (numpy.float64(2)) * ((x) * (y))
        assert imag.dtype == numpy.float64, (imag.dtype, numpy.float64)
        result = make_complex(real, imag)
        assert result.dtype == numpy.complex128, (result.dtype,)
        return result"""
    )


def test_readme_square_numpy_debug_2():

    ctx = Context(paths=[TestImplementations])

    graph = ctx.trace(TestImplementations.readme_square, complex)
    graph1 = graph.implement_missing(targets.numpy)
    py = graph1.tostring(targets.numpy, tab="", debug=2)

    assert py == utils.format_python(
        """\
def readme_square(z: numpy.complex128) -> numpy.complex128:
    with warnings.catch_warnings(action="ignore"):
        z = numpy.complex128(z)
        print("z=", z)
        real_z: numpy.float64 = (z).real
        print("real_z=", real_z)
        assert real_z.dtype == numpy.float64, (real_z.dtype, numpy.float64)
        x: numpy.float64 = numpy.abs(real_z)
        print("x=", x)
        assert x.dtype == numpy.float64, (x.dtype, numpy.float64)
        y: numpy.float64 = numpy.abs((z).imag)
        print("y=", y)
        assert y.dtype == numpy.float64, (y.dtype, numpy.float64)
        real_part: numpy.float64 = ((x) - (y)) * ((y) + (y))
        print("real_part=", real_part)
        assert real_part.dtype == numpy.float64, (real_part.dtype, numpy.float64)
        real: numpy.float64 = (numpy.float64(0)) if (numpy.equal(x, y, dtype=numpy.bool_)) else (real_part)
        print("real=", real)
        assert real.dtype == numpy.float64, (real.dtype, numpy.float64)
        imag: numpy.float64 = (numpy.float64(2)) * ((x) * (y))
        print("imag=", imag)
        assert imag.dtype == numpy.float64, (imag.dtype, numpy.float64)
        result = make_complex(real, imag)
        print("result=", result)
        assert result.dtype == numpy.complex128, (result.dtype,)
        return result"""
    )


def test_safe_min_xla_client():

    ctx = Context(paths=[TestImplementations], enable_alt=True, default_constant_type="MyDType")
    graph = ctx.trace(TestImplementations.safe_min, "y:XlaOp")
    graph1 = graph.implement_missing(targets.xla_client)

    py = graph1.tostring(targets.xla_client, tab="")

    assert py == utils.format_cpp(
        """\
template <typename MyDType>
XlaOp safe_min(XlaOp y) {
  return ScalarLike(
      y,
      (std::sqrt(std::numeric_limits<MyDType>::min())) / (4));
}"""
    )
