import numpy
import sys
import warnings

from .. import utils
from . import numpy as this_module
from .base import PrinterBase, modifier_base


constant_target = this_module

source_file_header = utils.format_python(
    """
import numpy
import warnings

def make_complex(r, i):
  if r.dtype == numpy.float32 and i.dtype == numpy.float32:
    return numpy.array([r, i]).view(numpy.complex64)[0]
  elif i.dtype == numpy.float64 and i.dtype == numpy.float64:
    return numpy.array([r, i]).view(numpy.complex128)[0]
  raise NotImplementedError((r.dtype, i.dtype))
"""
)


def __rewrite_modifier__(expr):
    return modifier_base(this_module, expr)


def upcast_func(target, expr):
    assert expr.kind == "upcast", expr.kind
    (x,) = expr.operands
    t = target.get_type(x)
    t_new = {
        "numpy.float16": "numpy.float32",
        "numpy.float32": "numpy.float64",
        "numpy.float64": "numpy.float128",
        "numpy.complex32": "numpy.complex64",
        "numpy.complex64": "numpy.complex128",
        "numpy.complex128": "numpy.complex256",
        "numpy.int8": "numpy.int16",
        "numpy.int16": "numpy.int32",
        "numpy.int32": "numpy.int64",
    }[t]
    s = target.tostring(x)
    return f"{t_new}({s})"


def downcast_func(target, expr):
    assert expr.kind == "downcast", expr.kind
    (x,) = expr.operands
    t = target.get_type(x)
    t_new = {
        "numpy.float32": "numpy.float16",
        "numpy.float64": "numpy.float32",
        "numpy.float128": "numpy.float64",
        "numpy.complex64": "numpy.complex32",
        "numpy.complex128": "numpy.complex64",
        "numpy.complex256": "numpy.complex128",
        "numpy.int16": "numpy.int8",
        "numpy.int32": "numpy.int16",
        "numpy.int64": "numpy.int32",
    }[t]
    s = target.tostring(x)
    return f"{t_new}({s})"


trace_arguments = dict(
    absolute=[(":complex128",), (":complex64",)],
    asin_acos_kernel=[(":complex128",), (":complex64",)],
    acos=[(":complex128",), (":complex64",), (":float64",), (":float32",)],
    acosh=[(":complex128",), (":complex64",), (":float64",), (":float32",)],
    asin=[(":complex128",), (":complex64",), (":float64",), (":float32",)],
    asinh=[(":complex128",), (":complex64",), (":float64",), (":float32",)],
    atan=[(":complex128",), (":complex64",), (":float64",), (":float32",)],
    atanh=[(":complex128",), (":complex64",), (":float64",), (":float32",)],
    log1p=[(":complex128",), (":complex64",), (":float64",), (":float32",)],
    hypot=[(":float32", ":float32"), (":float64", ":float64")],
    square=[(":complex128",), (":complex64",), (":float64",), (":float32",)],
    sqrt=[
        (":complex128",),
        (":complex64",),
    ],
)


source_file_extension = ".py"


def make_comment(message):
    return "# " + "\n# ".join(message.splitlines()) + "\n"


kind_to_target = dict(
    absolute="numpy.abs({0})",
    negative="-({0})",
    positive="+({0})",
    add="({0}) + ({1})",
    subtract="({0}) - ({1})",
    multiply="({0}) * ({1})",
    divide="({0}) / ({1})",
    remainder="({0}) %% ({1})",
    floor_divide="({0}) // ({1})",
    pow="({0}) ** ({1})",
    logical_and="({0}) and ({1})",
    logical_or="({0}) or ({1})",
    logical_xor=NotImplemented,
    logical_not="not ({0})",
    bitwise_invert="~({0})",
    bitwise_and="({0}) & ({1})",
    bitwise_or="({0}) | ({1})",
    bitwise_xor="({0}) ^ ({1})",
    bitwise_left_shift="({0}) << ({1})",
    bitwise_right_shift="({0}) >> ({1})",
    maximum="max({0}, {1})",
    minimum="min({0}, {1})",
    acos="numpy.arccos({0})",
    acosh="numpy.arccosh({0})",
    asin="numpy.arcsin({0})",
    asinh="numpy.arcsinh({0})",
    atan="numpy.arctan({0})",
    atanh="numpy.arctanh({0})",
    atan2="numpy.arctan2({0}, {1})",
    cos="numpy.cos({0})",
    cosh="numpy.cosh({0})",
    sin="numpy.sin({0})",
    sinh="numpy.sinh({0})",
    tan="numpy.tan({0})",
    tanh="numpy.tanh({0})",
    exp="numpy.exp({0})",
    expm1="numpy.expm1({0})",
    log="numpy.log({0})",
    log1p="numpy.log1p({0})",
    log2="numpy.log2({0})",
    log10="numpy.log10({0})",
    ceil="numpy.ceil({0})",
    floor="numpy.floor({0})",
    copysign="numpy.copysign({0}, {1})",
    round=NotImplemented,
    sign="numpy.sign({0})",
    truncate="numpy.trunc({0})",
    conjugate="({0}).conjugate()",
    real="({0}).real",
    imag="({0}).imag",
    complex="make_complex({0}, {1})",
    hypot="numpy.hypot({0}, {1})",
    square="numpy.square({0})",
    sqrt="numpy.sqrt({0})",
    select="({1}) if ({0}) else ({2})",
    lt="({0}) < ({1})",
    le="({0}) <= ({1})",
    gt="({0}) > ({1})",
    ge="({0}) >= ({1})",
    eq="numpy.equal({0}, {1}, dtype=numpy.bool_)",
    ne="({0}) != ({1})",
    nextafter="numpy.nextafter({0}, {1})",
    upcast=upcast_func,
    downcast=downcast_func,
    is_finite="numpy.isfinite({0})",
)

constant_to_target = dict(
    smallest_subnormal="numpy.finfo({type}).smallest_subnormal",
    smallest="numpy.finfo({type}).smallest_normal",
    eps="numpy.finfo({type}).eps",
    largest="numpy.finfo({type}).max",
    posinf="{type}(numpy.inf)",
    neginf="-{type}(numpy.inf)",
    pi="{type}(numpy.pi)",
    nan="{type}(numpy.nan)",
)

type_to_target = dict(
    integer8="numpy.int8",
    integer16="numpy.int16",
    integer32="numpy.int32",
    integer64="numpy.int64",
    integer="numpy.int64",
    float32="numpy.float32",
    float64="numpy.float64",
    float="numpy.float64",
    float128="numpy.float128",
    complex64="numpy.complex64",
    complex128="numpy.complex128",
    complex="numpy.complex128",
    boolean="numpy.bool_",
)


def as_function(graph, debug=0):
    """Return function graph as Python callable."""
    assert graph.kind == "apply"
    d = dict(
        sys=sys,
        numpy=numpy,
        make_complex=utils.make_complex,
        finfo_float32=numpy.finfo(numpy.float32),
        finfo_float64=numpy.finfo(numpy.float64),
        warnings=warnings,
    )
    np = graph.tostring(this_module, debug=debug)
    if debug >= 2:
        print(np)
    exec(np, d)
    return d[graph.operands[0].operands[0]]


class Printer(PrinterBase):
    """Printer for Python target"""

    force_cast_arguments = True

    kind_to_target = kind_to_target

    type_to_target = type_to_target

    constant_to_target = constant_to_target

    def make_assignment(self, typ, var, value):
        if typ is None:
            return f"{var} = {value}"
        return f"{var}: {typ} = {value}"

    def make_constant(self, like, value):
        typ = self.get_type(like)
        s = str(value)
        s = {"inf": "numpy.inf", "-inf": "-numpy.inf", "nan": "numpy.nan"}.get(s, s)
        return f"{typ}({s})"

    def make_argument(self, arg):
        assert arg.kind == "symbol", arg.kind
        typ = self.get_type(arg)
        return f"{arg}: {typ}"

    def show_value(self, var):
        return f'print("{var}=", {var})'

    def check_dtype(self, var, dtype):
        return f"assert {var}.dtype == {dtype}, ({var}.dtype, {dtype})"

    def make_apply(self, expr, name, tab=""):
        sargs = ", ".join(map(self.make_argument, expr.operands[1:-1]))
        body = self.tostring(expr.operands[-1])
        body_type = self.get_type(expr.operands[-1])
        lines = []
        lines.append(f"{tab}def {name}({sargs}) -> {body_type}:")
        lines.append(f'{tab}  with warnings.catch_warnings(action="ignore"):')
        for a in self.assignments:
            lines.append(f"{tab}    {a}")
        lines.append(f"{tab}    result = {body}")
        if self.debug >= 2:
            lines.append(f'{tab}    print("result=", result)')
        if self.debug >= 1:
            lines.append(f"{tab}    assert result.dtype == {body_type}, (result.dtype,)")
        lines.append(f"{tab}    return result")
        return utils.format_python("\n".join(lines))
