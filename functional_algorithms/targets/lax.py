import numpy
import sys
import warnings

from .. import utils
from . import lax as this_module
from .base import PrinterBase, modifier_base


constant_target = this_module

source_file_header = utils.format_python(
    """
import numpy
from jax._src import lax
from jax._src import numpy as jnp
from jax._src.api import jit
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import promote_args_inexact
"""
)


def __rewrite_modifier__(expr):
    return modifier_base(this_module, expr)


def list_func(target, expr):
    assert expr.kind == "list"
    s = ", ".join([target.tostring(item) for item in expr.operands])
    return f"[{s}]"


def dtype_index_func(target, expr):
    assert expr.kind == "dtype_index"
    x = expr.operands[0]
    # See tools/generate_apmath_lax.py for an example how to define _np_dtypes
    return f"_np_dtypes.index({x.ref}.dtype.type)"


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
    log=[
        (":complex128",),
        (":complex64",),
    ],
    log10=[
        (":complex128",),
        (":complex64",),
    ],
    log2=[
        (":complex128",),
        (":complex64",),
    ],
    exp=[
        (":complex128",),
        (":complex64",),
    ],
)


source_file_extension = ".py"


def make_comment(message):
    return "# " + "\n# ".join(message.splitlines()) + "\n"


kind_to_target = dict(
    absolute="lax.abs({0})",
    negative="lax.neg({0})",
    positive="+({0})",
    add="lax.add({0}, {1})",
    subtract="lax.sub({0}, {1})",
    multiply="lax.mul({0}, {1})",
    divide="lax.div({0}), ({1})",
    remainder="({0}) %% ({1})",
    floor_divide="({0}) // ({1})",
    pow="lax.pow({0}, {1})",
    logical_and="jnp.logical_and({0}, {1})",
    logical_or="jnp.logical_or({0}, {1})",
    logical_xor=NotImplemented,
    logical_not="jnp.logical_not({0})",
    bitwise_invert="~({0})",
    bitwise_and="({0}) & ({1})",
    bitwise_or="({0}) | ({1})",
    bitwise_xor="({0}) ^ ({1})",
    bitwise_left_shift="({0}) << ({1})",
    bitwise_right_shift="({0}) >> ({1})",
    maximum="lax.max({0}, {1})",
    minimum="lax.min({0}, {1})",
    acos="jnp.arccos({0})",
    acosh="jnp.arccosh({0})",
    asin="jnp.arcsin({0})",
    asinh="jnp.arcsinh({0})",
    atan="jnp.arctan({0})",
    atanh="jnp.arctanh({0})",
    atan2="jnp.arctan2({0}, {1})",
    cos="jnp.cos({0})",
    cosh="jnp.cosh({0})",
    sin="jnp.sin({0})",
    sinh="jnp.sinh({0})",
    tan="jnp.tan({0})",
    tanh="jnp.tanh({0})",
    exp="jnp.exp({0})",
    exp2="jnp.exp2({0})",
    expm1="jnp.expm1({0})",
    log="jnp.log({0})",
    log1p="jnp.log1p({0})",
    log2="jnp.log2({0})",
    log10="jnp.log10({0})",
    ceil="jnp.ceil({0})",
    floor="lax.floor({0})",
    copysign="jnp.copysign({0}, {1})",
    round=NotImplemented,
    sign="lax.sign({0})",
    truncate="jnp.trunc({0})",
    conjugate="lax.conj({0})",
    real="lax.real({0})",
    imag="lax.imag({0})",
    # complex="make_complex({0}, {1})",
    hypot="jnp.hypot({0}, {1})",
    square="lax.square({0})",
    sqrt="lax.sqrt({0})",
    # select="({1}) if ({0}) else ({2})",
    select="jnp.where({0}, {1}, {2})",  # jax.select requires broadcasted cases, hence using jnp.where
    lt="lax.lt({0}, {1})",
    # lt="jnp.less({0}, {1})",
    le="lax.le({0}, {1})",
    gt="lax.gt({0}, {1})",
    ge="lax.ge({0}, {1})",
    eq="lax.eq({0}, {1})",
    ne="lax.ne({0}, {1})",
    nextafter="jnp.nextafter({0}, {1})",
    # upcast=upcast_func,
    # downcast=downcast_func,
    is_finite="jnp.isfinite({0})",
    list=list_func,
    item="{0}[{1}]",
    dtype_index=dtype_index_func,
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
    array="Array",
    integer8="numpy.int8",
    integer16="numpy.int16",
    integer32="numpy.int32",
    integer64="numpy.int64",
    integer="numpy.int64",
    float16="numpy.float16",
    float32="numpy.float32",
    float64="numpy.float64",
    float="numpy.float64",
    float128="numpy.float128",
    complex64="numpy.complex64",
    complex128="numpy.complex128",
    complex="numpy.complex128",
    boolean="numpy.bool_",
)


def as_function(graph, debug=0, numpy=numpy, force_cast_arguments=None):
    """Return function graph as Python callable."""
    assert graph.kind == "apply"
    d = dict(
        # sys=sys,
        # numpy=numpy,
        # make_complex=utils.make_complex,
        # finfo_float32=numpy.finfo(numpy.float32),
        # finfo_float64=numpy.finfo(numpy.float64),
        # warnings=warnings,
    )
    np = graph.tostring(this_module, debug=debug, force_cast_arguments=force_cast_arguments)
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
        # assert value.kind != "list"
        return f"{var} = {value}"
        if typ is None:
            return f"{var} = {value}"
        return f"{var}: {typ} = {value}"

    def make_constant(self, like, value):
        typ = self.get_type(like)
        s = str(value)
        s = {"inf": "jnp.inf", "-inf": "-jnp.inf", "nan": "jnp.nan"}.get(s, s)
        if typ.startswith("numpy."):
            return f"{typ}({s})"
        if typ in {"Array", "ArrayLike"}:
            return f"numpy.array({s}, dtype={like.operands[0]}.dtype.type)"
        return f"({s})"

    def get_type(self, expr):
        t = expr.get_type()
        typ = super().get_type(expr)

        if t.kind == "array" and typ == "Array":
            ctx = expr.context
            same_dtype_cache = ctx.parameters.get("same_dtype_cache", {})
            if expr.key in same_dtype_cache:
                return "ArrayLike"
        return typ

    def make_argument(self, arg):
        if arg.kind == "list":
            at = ", ".join([str(self.get_type(a_)) for a_ in arg.operands])
            return f"{arg.ref}: list[{at}]"
        assert arg.kind == "symbol", arg.kind
        typ = self.get_type(arg)
        return f"{arg}: {typ}"

    def init_arguments(self, funcname, args):
        assignments = []

        if not args:
            return assignments

        ctx = args[0].context
        same_dtype_cache = ctx.parameters.get("same_dtype_cache", {})

        already_promoted = set()
        for a in args:
            self.defined_refs.add(a.ref)

            if a.key in same_dtype_cache and a.key not in already_promoted:
                lst = [a.ref]
                already_promoted.add(a.key)
                for a_ in args:
                    if a_.key in same_dtype_cache[a.key]:
                        lst.append(a_.ref)
                        already_promoted.add(a_.key)
                lst = ", ".join(lst)
                assignments.append(f'{lst}, = promote_args_inexact("{funcname}", {lst})')

        return assignments

    def make_getitem(self, var, index):
        return f"{var.ref}[{index}]"

    def show_value(self, var):
        return f'print("{var}=", {var})'

    def check_dtype(self, var, dtype):
        return f"assert {var}.dtype == {dtype}, ({var}.dtype, {dtype})"

    def make_apply(self, expr, name, tab=""):
        sargs = ", ".join(map(self.make_argument, expr.operands[1:-1]))
        body = self.tostring(expr.operands[-1])
        btype = expr.operands[-1].get_type()
        body_type = self.get_type(expr.operands[-1])
        lines = []
        lines.append(f"{tab}@jit")
        lines.append(f"{tab}def {name}({sargs}) -> {body_type}:")
        doc = expr.props.get("__doc__", None)
        if doc:
            lines.append(f"{tab}    '''{doc}'''\n")
        for a in self.assignments:
            lines.append(f"{tab}    {a}")
        lines.append(f"{tab}    return {body}")
        return utils.format_python("\n".join(lines))
