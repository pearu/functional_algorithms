import warnings
import sys
import math
from . import python as this_module
from .. import utils

source_file_header = utils.format_python(
    """
import math
import sys
"""
)


trace_arguments = dict(
    asin=[(":complex",)],
    hypot=[(":float", ":float")],
    square=[(":float",), (":complex",)],
)


source_file_extension = ".py"


def make_comment(message):
    return "# " + "\n# ".join(message.splitlines()) + "\n"


kind_to_target = dict(
    abs="abs({0})",
    negative="-({0})",
    positive="+({0})",
    add="({0}) + ({1})",
    subtract="({0}) - ({1})",
    multiply="({0}) * ({1})",
    divide="({0}) / ({1})",
    reminder="({0}) %% ({1})",
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
    bitwise_not="!({0})",
    bitwise_left_shift="({0}) << ({1})",
    bitwise_right_shift="({0}) >> ({1})",
    maximum="max({0}, {1})",
    minimum="min({0}, {1})",
    acos="math.acos({0})",
    acosh="math.acosh({0})",
    asin="math.asin({0})",
    asinh="math.asinh({0})",
    atan="math.atan({0})",
    atanh="math.atanh({0})",
    atan2="math.atan2({0}, {1})",
    cos="math.cos({0})",
    cosh="math.cosh({0})",
    sin="math.sin({0})",
    sinh="math.sinh({0})",
    tan="math.tan({0})",
    tanh="math.tanh({0})",
    exp="math.exp({0})",
    expm1="math.expm1({0})",
    log="math.log({0})",
    log1p="math.log1p({0})",
    log2="math.log2({0})",
    log10="math.log10({0})",
    ceil="math.ceil({0})",
    floor="math.floor({0})",
    copysign="math.copysign({0})",
    round=NotImplemented,
    sign=NotImplemented,
    trunc="math.trunc({0})",
    conj="({0}).conjugate()",
    real="({0}).real",
    imag="({0}).imag",
    complex="complex({0}, {1})",
    hypot=NotImplemented,
    square=NotImplemented,
    sqrt="math.sqrt({0})",
    select="({1}) if ({0}) else ({2})",
    lt="({0}) < ({1})",
    le="({0}) <= ({1})",
    gt="({0}) > ({1})",
    ge="({0}) >= ({1})",
    eq="({0}) == ({1})",
    ne="({0}) != ({1})",
)

constant_to_target = dict(smallest="sys.float_info.min", largest="sys.float_info.max", posinf="math.inf", neginf="-math.inf")

type_to_target = dict(integer="int", float="float", complex="complex", boolean="bool")


def as_function(graph):
    """Return function graph as Python callable."""
    assert graph.kind == "apply"
    d = dict(sys=sys, math=math)
    exec(graph.tostring(this_module), d)
    return d[graph.operands[0].operands[0]]


class Printer:
    """Printer for Python target"""

    def __init__(self, need_ref, debug=1):
        self.need_ref = need_ref
        self.defined_refs = set()
        self.assignments = []
        self.debug = debug

    def tostring(self, expr, tab=""):
        if expr.kind == "symbol":
            return str(expr.operands[0])
        elif expr.kind == "constant":
            value = expr.operands[0]
            if isinstance(value, str):
                target_value = constant_to_target.get(value, NotImplemented)
                if target_value is not NotImplemented:
                    if value not in self.defined_refs:
                        self.assignments.append(f"{value}: {expr.get_type()} = {target_value}")
                        self.defined_refs.add(value)
                else:
                    warnings.warn(f"python constant_to_target does not implement {value}")
                return value
            return str(value)
        elif expr.kind == "apply":
            name = expr.props.get("name", expr.operands[0])
            if not isinstance(name, str):
                assert name.kind == "symbol", name
                name = name.operands[0]
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
        elif expr.ref in self.defined_refs:
            assert self.need_ref.get(expr.ref), expr.ref
            return expr.ref

        tmpl = kind_to_target.get(expr.kind, NotImplemented)
        if tmpl is NotImplemented:
            raise NotImplementedError(f"python operator for {expr.kind}")
        result = tmpl.format(*[self.tostring(operand) for operand in expr.operands])

        if self.need_ref.get(expr.ref):
            t = type_to_target[str(expr.get_type())]
            self.assignments.append(f"{expr.ref}: {t} = {result}")
            result = expr.ref

        if expr.ref is not None:
            self.defined_refs.add(expr.ref)

        return result
