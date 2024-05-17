import warnings

from . import stablehlo as this_module

source_file_header = ""


trace_arguments = dict(
    asin=[(":complex",)],
    hypot=[(":float", ":float")],
    square=[(":float",), (":complex",)],
)


source_file_extension = ".td"


def make_comment(message):
    return "// " + "\n// ".join(message.splitlines()) + "\n"


kind_to_target = dict(
    abs="StableHLO_AbsOp",
    negative="StableHLO_NegOp",
    positive="StableHLO_PosOp",
    add="StableHLO_AddOp",
    subtract="StableHLO_SubtractOp",
    multiply="StableHLO_MulOp",
    divide="StableHLO_DivOp",
    reminder=NotImplemented,
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
    bitwise_not=NotImplemented,
    bitwise_left_shift=NotImplemented,
    bitwise_right_shift=NotImplemented,
    maximum="StableHLO_MaxOp",
    minimum="StableHLO_MinOp",
    acos=NotImplemented,
    acosh=NotImplemented,
    asin=NotImplemented,
    asinh=NotImplemented,
    atan=NotImplemented,
    atanh=NotImplemented,
    atan2="StableHLO_Atan2Op",
    cos=NotImplemented,
    cosh=NotImplemented,
    sin=NotImplemented,
    sinh=NotImplemented,
    tan=NotImplemented,
    tanh=NotImplemented,
    exp=NotImplemented,
    expm1=NotImplemented,
    log="StableHLO_LogOp",
    log1p="StableHLO_Log1pOp",
    log2=NotImplemented,
    log10=NotImplemented,
    ceil=NotImplemented,
    floor=NotImplemented,
    copysign=NotImplemented,
    round=NotImplemented,
    sign=NotImplemented,
    trunc=NotImplemented,
    conj=NotImplemented,
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
)


constant_to_target = dict(
    largest="StableHLO_ConstantLikeMaxFiniteValue",
    smallest="StableHLO_ConstantLikeSmallestNormalizedValue",
    posinf="StableHLO_ConstantLikePosInfValue",
    neginf="StableHLO_ConstantLikeNegInfValue",
)


class Printer:
    """Printer for stablehlo"""

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
            chlo_name = expr.props.get("name", f"CHLO_{name.ref.title()}")
            lines.append(f"{tab}def : Pat<({chlo_name} {sargs}),")

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
            if like.ref not in self.defined_refs:
                raise RuntimeError(
                    f"undefined reference {like} in {expr} (when a constant is used as left operand, its ref value must be specified explicitly)"
                )
            if isinstance(value, str):
                v = constant_to_target.get(value, NotImplemented)
                if v is not NotImplemented:
                    return f"{tab}({v} ${like.ref})"
                else:
                    warnings.warn(f"Constant `{value}` is not implemented in {this_module.__name__}.constant_to_target")
            return f'{tab}(StableHLO_ConstantLike<"{value}">{ref} ${like.ref})'

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
