import warnings
import sys
import math
from . import python as this_module
from .. import utils
from .base import PrinterBase, modifier_base

constant_target = this_module

source_file_header = utils.format_python("")


def __rewrite_modifier__(expr):
    return modifier_base(this_module, expr)


trace_arguments = dict()


source_file_extension = ".txt"  # not applicable


def make_comment(message):
    return "# " + "\n# ".join(message.splitlines()) + "\n"


kind_to_target = dict(
    absolute="abs({0})",
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
    logical_xor="({0}) != ({1})",
    logical_not="not ({0})",
    bitwise_invert="~({0})",
    bitwise_and="({0}) & ({1})",
    bitwise_or="({0}) | ({1})",
    bitwise_xor="({0}) ^ ({1})",
    bitwise_left_shift="({0}) << ({1})",
    bitwise_right_shift="({0}) >> ({1})",
    maximum="max({0}, {1})",
    minimum="min({0}, {1})",
    acos="acos({0})",
    acosh="acosh({0})",
    # asin="asin({0})",
    asinh="asinh({0})",
    atan="atan({0})",
    atanh="atanh({0})",
    atan2="atan2({0}, {1})",
    cos="cos({0})",
    cosh="cosh({0})",
    sin="sin({0})",
    sinh="sinh({0})",
    tan="tan({0})",
    tanh="tanh({0})",
    exp="exp({0})",
    expm1="expm1({0})",
    log="log({0})",
    log1p="log1p({0})",
    log2="log2({0})",
    log10="log10({0})",
    ceil="ceil({0})",
    floor="floor({0})",
    copysign="copysign({0}, {1})",
    round=NotImplemented,
    sign="(0 if {0} == 0 else copysign(1, {0}))",
    truncate="trunc({0})",
    conjugate="({0}).conjugate()",
    real="({0}).real",
    imag="({0}).imag",
    complex="complex({0}, {1})",
    hypot="hypot({0}, {1})",
    square="square({0})",
    sqrt="sqrt({0})",
    select="({1}) if ({0}) else ({2})",
    lt="{0} < {1}",
    le="{0} <= {1}",
    gt="{0} > {1}",
    ge="{0} >= {1}",
    eq="{0} == {1}",
    ne="{0} != {1}",
)

constant_to_target = dict(
    smallest="smallest",
    largest="largest",
    posinf="inf",
    neginf="-inf",
    pi="pi",
    eps="eps",
)

type_to_target = dict(integer="int", float="float", complex="complex", boolean="bool", float32="float32")


def as_function(graph):
    """Return function graph as Python callable."""
    assert graph.kind == "apply"
    d = dict()
    exec(graph.tostring(this_module), d)
    return d[graph.operands[0].operands[0]]


class Printer(PrinterBase):
    """Printer for symbolic target"""

    enable_assignments = False

    kind_to_target = kind_to_target

    type_to_target = type_to_target

    constant_to_target = constant_to_target

    def make_assignment(self, typ, var, value):
        if typ is None:
            return f"{var} = {value}"
        return f"{var}: {typ} = {value}"

    def make_constant(self, like, value):
        return str(value)

    def show_value(self, var):
        return f'print("{var}=", {var})'

    def make_apply(self, expr, name, tab=""):
        args = expr.operands[1:-1]
        body = self.tostring(expr.operands[-1])
        body_type = type_to_target[str(expr.operands[-1].get_type())]
        sargs = []
        for a in args:
            self.defined_refs.add(a.ref)
            typ = type_to_target[str(a.operands[1])]
            sargs.append(f"{self.tostring(a)}: {typ}")
        sargs = ", ".join(sargs)

        lines = []
        lines.append(f"{tab}def {name}({sargs}) -> {body_type}:")
        for a in self.assignments:
            lines.append(f"{tab}  {a}")
        lines.append(f"{tab}  return {body}")
        return utils.format_python("\n".join(lines))
