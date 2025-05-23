import warnings

from . import stablehlo as this_module
from .base import modifier_base

constant_target = this_module

source_file_header = ""


trace_arguments = dict(
    absolute=[(":complex",)],
    asin_acos_kernel=[(":complex",)],
    acos=[(":float",), (":complex",)],
    acosh=[(":float",), (":complex",)],
    asin=[(":float",), (":complex",)],
    asinh=[(":float",), (":complex",)],
    atan=[(":float",), (":complex",)],
    atanh=[(":float",), (":complex",)],
    log1p=[(":float",), (":complex",)],
    hypot=[(":float", ":float")],
    square=[(":float",), (":complex",)],
    sqrt=[(":complex",)],
    log=[(":complex",)],
    log10=[(":complex",)],
    log2=[(":complex",)],
    exp=[(":complex",)],
)


source_file_extension = ".td"


def __rewrite_modifier__(expr):
    return modifier_base(this_module, expr)


def make_comment(message):
    lst = []
    for line in message.splitlines():
        if line.strip():
            lst.append("// " + line)
        else:
            lst.append("//")
    return "\n".join(lst) + "\n"


kind_to_target = dict(
    absolute="StableHLO_AbsOp",
    negative="StableHLO_NegOp",
    positive="StableHLO_PosOp",
    add="StableHLO_AddOp",
    subtract="StableHLO_SubtractOp",
    multiply="StableHLO_MulOp",
    divide="StableHLO_DivOp",
    remainder=NotImplemented,
    floor_divide=NotImplemented,
    pow=NotImplemented,
    logical_and="StableHLO_AndOp",
    logical_or="StableHLO_OrOp",
    logical_xor="StableHLO_XorOp",
    logical_not="StableHLO_NotOp",
    bitwise_invert=NotImplemented,
    bitwise_and=NotImplemented,
    bitwise_or=NotImplemented,
    bitwise_xor=NotImplemented,
    bitwise_left_shift="StableHLO_ShiftLeftOp",
    bitwise_right_shift="StableHLO_ShiftRightArithmeticOp",
    maximum="StableHLO_MaxOp",
    minimum="StableHLO_MinOp",
    asin_acos_kernel="CHLO_AsinAcosKernelOp",
    acos="CHLO_AcosOp",
    acosh="CHLO_AcoshOp",
    asin="CHLO_AsinOp",
    asinh="CHLO_AsinhOp",
    atan="CHLO_AtanOp",
    atanh="CHLO_AtanhOp",
    atan2="StableHLO_Atan2Op",
    cos="StableHLO_CosineOp",
    cosh=NotImplemented,
    sin="StableHLO_SineOp",
    sinh=NotImplemented,
    tan=NotImplemented,
    tanh=NotImplemented,
    exp="StableHLO_ExpOp",
    expm1="StableHLO_Expm1Op",
    log="StableHLO_LogOp",
    log1p="StableHLO_Log1pOp",
    log2=NotImplemented,
    log10=NotImplemented,
    ceil=NotImplemented,
    floor=NotImplemented,
    copysign=NotImplemented,
    round=NotImplemented,
    sign="StableHLO_SignOp",
    truncate=NotImplemented,
    conjugate=NotImplemented,
    real="StableHLO_RealOp",
    imag="StableHLO_ImagOp",
    complex="StableHLO_ComplexOp",
    hypot=NotImplemented,
    square=NotImplemented,
    sqrt="StableHLO_SqrtOp",
    select="StableHLO_SelectOp",
    lt=None,
    le=None,
    gt=None,
    ge=None,
    eq=None,
    ne=None,
    nextafter="CHLO_NextAfterOp",
    is_finite="StableHLO_IsFiniteOp",
)


constant_to_target = dict(
    largest="StableHLO_ConstantLikeMaxFiniteValue",
    smallest="StableHLO_ConstantLikeSmallestNormalizedValue",
    posinf="StableHLO_ConstantLikePosInfValue",
    neginf="StableHLO_ConstantLikeNegInfValue",
    pi='StableHLO_ConstantLike<"M_PI">',
)


class Printer:
    """Printer for stablehlo"""

    constant_target = None

    def __init__(self, need_ref, debug=1):
        self.need_ref = need_ref
        self.defined_refs = set()
        self.debug = debug

    def tostring(self, expr, tab=""):

        if expr.kind == "apply":
            name = expr.operands[0]
            args = expr.operands[1:-1]
            body = expr.operands[-1]

            sargs = []
            for a in args:
                self.defined_refs.add(a.ref)
                typ = "ComplexElementType" if a.is_complex else "NonComplexElementType"
                sargs.append(f"{typ}:${a.ref}")
            sargs = ", ".join(sargs)

            lines = []
            expander_name = expr.props.get("expander_name", "")
            chlo_name = expr.props.get("name", f"CHLO_{name.ref.title()}")
            lines.append(f"{tab}def {expander_name}: Pat<({chlo_name} {sargs}),")

            for line in self.tostring(body, tab=tab + "  ").splitlines():
                sline = line.lstrip()
                if sline.startswith(")"):
                    lines[-1] += sline
                else:
                    lines.append(line)
            lines[-1] += ">;"
            return "\n".join(lines)

        if expr.kind == "symbol":
            return f"{tab}${expr.ref}"

        if expr.ref in self.defined_refs:
            assert self.need_ref.get(expr.ref), expr.ref
            return f"{tab}${expr.ref}"

        self.defined_refs.add(expr.ref)

        ref = f":${expr.ref}" if self.need_ref.get(expr.ref) else ""

        if expr.kind == "constant":
            value, like = expr.operands
            if like.ref in self.defined_refs:
                like_val = f"${like.ref}"
            else:
                warnings.warn(
                    f"undefined reference {like} in {expr} (when a constant is used as left operand,"
                    " its ref value must be specified explicitly)"
                )
                like_val = self.tostring(like)
            if isinstance(value, str):
                v = constant_to_target.get(value, NotImplemented)
                if v is not NotImplemented:
                    return f"{tab}({v}{ref} {like_val})"
                else:
                    warnings.warn(f"Constant `{value}` is not implemented in {this_module.__name__}.constant_to_target")
            return f'{tab}(StableHLO_ConstantLike<"{value}">{ref} {like_val})'

        elif expr.kind in {"lt", "le", "gt", "ge", "eq", "ne"}:
            lines = []
            lines.append(f"{tab}(StableHLO_CompareOp{ref}")
            for operand in expr.operands:
                for line in self.tostring(operand, tab=tab + " ").splitlines():
                    lines.append(line)
                lines[-1] += ","
            lines.append(f'{tab}  StableHLO_ComparisonDirectionValue<"{expr.kind.upper()}">,')
            lines.append(f"{tab}  (STABLEHLO_DEFAULT_COMPARISON_TYPE)")
            lines.append(f"{tab})")
            return "\n".join(lines)

        else:
            hlo = kind_to_target.get(expr.kind, NotImplemented)
            if hlo is NotImplemented:
                raise NotImplementedError(f"stablehlo operator for {expr.kind}")
            lines = []
            lines.append(f"{tab}({hlo}{ref}")
            short_args = []
            operand_lines = []
            for operand in expr.operands:
                op_lines = self.tostring(operand, tab=tab + "  ").splitlines()
                if len(op_lines) == 1:
                    line = op_lines[0].strip()
                    if line.startswith("$"):
                        short_args.append(line)
                operand_lines.extend(op_lines)
                operand_lines[-1] += ","
            operand_lines[-1] = operand_lines[-1][:-1]

            if len(short_args) == len(expr.operands):
                lines[-1] += " " + ", ".join(short_args) + ")"
            else:
                lines.extend(operand_lines)
                lines.append(f"{tab})")

            return "\n".join(lines)
