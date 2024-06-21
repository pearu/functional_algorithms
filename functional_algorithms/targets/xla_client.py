import warnings

from . import xla_client as this_module
from . import cpp as constant_target
from .base import PrinterBase
from .. import utils

source_file_header = """
#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>
"""

trace_arguments = dict(
    absolute=[(":complex",)],
    hypot=[(":float", ":float")],
    complex_acos=[(":complex", ":complex")],
    real_acos=[(":float", ":float")],
    complex_asin=[(":complex", ":complex")],
    real_asin=[(":float", ":float")],
    complex_asinh=[(":complex", ":complex")],
    real_asinh=[(":float", ":float")],
    square=[(":float", ":float"), (":complex", ":complex")],
    acos=[(":float", ":float"), (":complex", ":complex")],
    asin=[(":float", ":float"), (":complex", ":complex")],
    asinh=[(":float", ":float"), (":complex", ":complex")],
)

source_file_extension = ".cc"


def make_comment(message):
    lst = []
    for line in message.splitlines():
        if line.strip():
            lst.append("// " + line)
        else:
            lst.append("//")
    return "\n".join(lst) + "\n"


kind_to_target = dict(
    abs="Abs({0})",
    negative="Neg({0})",
    positive="({0})",
    add="Add({0}, {1})",
    subtract="Sub({0}, {1})",
    multiply="Mul({0}, {1})",
    divide="Div({0}, {1})",
    reminder="Rem({0}, {1})",
    floor_divide=NotImplemented,
    pow="Pow({0}, {1})",
    logical_and="And({0}, {1})",
    logical_or="Or({0}, {1})",
    logical_xor="Xor({0}, {1})",
    logical_not="Not({0})",
    bitwise_invert=NotImplemented,
    bitwise_and="({0}) & ({1})",
    bitwise_or="({0}) | ({1})",
    bitwise_xor="({0}) ^ ({1})",
    bitwise_not="~({0})",
    bitwise_left_shift="({0}) << ({1})",
    bitwise_right_shift="({0}) >> ({1})",
    maximum="Max({0}, {1})",
    minimum="Min({0}, {1})",
    acos="Acos({0})",
    acosh="Acosh({0})",
    asin="Asin({0})",
    asinh="Asinh({0})",
    atan="Atan({0})",
    atanh="Atanh({0})",
    atan2="Atan2({0}, {1})",
    cos="Cos({0})",
    cosh="Cosh({0})",
    sin="Sin({0})",
    sinh="Sinh({0})",
    tan="Tan({0})",
    tanh="Tanh({0})",
    exp="Exp({0})",
    expm1="Expm1({0})",
    log="Log({0})",
    log1p="Log1p({0})",
    log2="Log2({0})",
    log10="Log10({0})",
    ceil="Ceil({0})",
    floor="Floot({0})",
    copysign=NotImplemented,
    round="Round({0})",
    sign="Sign({0})",
    trunc=NotImplemented,
    conj=NotImplemented,
    real="Real({0})",
    imag="Imag({0})",
    complex="Complex({0}, {1})",
    hypot=NotImplemented,
    square="Square({0})",
    sqrt="Sqrt({0})",
    select="Select({0}, {1}, {2})",
    lt="Lt({0}, {1})",
    le="Le({0}, {1})",
    gt="Gt({0}, {1})",
    ge="Ge({0}, {1})",
    eq="Eq({0}, {1})",
    ne="Ne({0}, {1})",
    is_finite="IsFinite({0})",
    is_inf="IsInf({0})",
    is_posinf="IsPosInf({0})",
    is_neginf="IsNegInf({0})",
    is_nan="IsNan({0})",
    is_negzero="IsNegZero({0})",
    nextafter="NextAfter({0}, {1})",
)

constant_to_target = dict()


class Printer(PrinterBase):
    """Printer for XLA Client C++ target"""

    constant_target = constant_target

    kind_to_target = kind_to_target

    constant_to_target = constant_to_target

    type_to_target = dict(
        float="XlaOp",
        complex="XlaOp",
        boolean="XlaOp",
        type="XlaOp",
    )

    def make_assignment(self, typ, var, value):
        return f"{typ} {var} = {value};"

    def make_constant(self, like, value):
        return f"ScalarLike({like.ref}, {value})"

    def make_argument(self, arg):
        typ = self.get_type(arg)
        return f"{typ} {arg}"

    def make_apply(self, expr, name, tab=""):
        sargs = ", ".join(map(self.make_argument, expr.operands[1:-1]))
        body = self.tostring(expr.operands[-1])
        body_type = self.get_type(expr.operands[-1])
        lines = []
        if expr.context.default_like is not None:
            lines.append(f"{tab}template <typename {expr.context.default_like.operands[1]}>")
        lines.append(f"{tab}{body_type} {name}({sargs}) {{")
        for a in self.assignments:
            lines.append(f"{tab}    {a}")
        lines.append(f"{tab}    return {body};")
        lines.append(f"{tab}}}")
        return utils.format_cpp("\n".join(lines))
