import warnings

from . import xla_client as this_module
from .base import PrinterBase
from .. import utils

constant_target = this_module

source_file_header = """
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
"""

trace_arguments = dict(
    square=[(":float32",), (":float64",), (":complex64",), (":complex128",)],
    absolute=[(":float32",), (":float64",), (":complex64",), (":complex128",)],
    hypot=[(":float32", ":float32"), (":float64", ":float64")],
    acos=[(":float32",), (":float64",), (":complex64",), (":complex128",)],
    asin=[(":float32",), (":float64",), (":complex64",), (":complex128",)],
    asinh=[(":float32",), (":float64",), (":complex64",), (":complex128",)],
    # complex_asin=[(":complex", ":complex")],
    # real_asin=[(":float", ":float")],
    # square=[(":float", ":float"), (":complex", ":complex")],
    # asin=[(":float", ":float"), (":complex", ":complex")],
)

source_file_extension = ".cpp"


def make_comment(message):
    lst = []
    for line in message.splitlines():
        if line.strip():
            lst.append("// " + line)
        else:
            lst.append("//")
    return "\n".join(lst) + "\n"


kind_to_target = dict(
    abs="std::abs({0})",
    negative="-({0})",
    positive="({0})",
    add="({0}) + ({1})",
    subtract="({0}) - ({1})",
    multiply="({0}) * ({1})",
    divide="({0}) / ({1})",
    reminder="({0}) % ({1})",
    floor_divide=NotImplemented,
    # pow="Pow({0}, {1})",
    logical_and="({0}) && ({1})",
    logical_or="({0}) || ({1})",
    # logical_xor="Xor({0}, {1})",
    logical_not="!({0})",
    bitwise_invert=NotImplemented,
    bitwise_and="({0}) & ({1})",
    bitwise_or="({0}) | ({1})",
    bitwise_xor="({0}) ^ ({1})",
    bitwise_not="~({0})",
    bitwise_left_shift="({0}) << ({1})",
    bitwise_right_shift="({0}) >> ({1})",
    maximum="std::max({0}, {1})",
    minimum="std::min({0}, {1})",
    acos="std::acos({0})",
    acosh="std::acosh({0})",
    asin="std::asin({0})",
    asinh="std::asinh({0})",
    atan="std::atan({0})",
    atanh="std::atanh({0})",
    atan2="std::atan2({0}, {1})",
    cos="std::cos({0})",
    cosh="std::cosh({0})",
    sin="std::sin({0})",
    sinh="std::sinh({0})",
    tan="std::tan({0})",
    tanh="std::tanh({0})",
    exp="std::exp({0})",
    expm1="std::expm1({0})",
    log="std::log({0})",
    log1p="std::log1p({0})",
    log2="std::log2({0})",
    log10="std::log10({0})",
    ceil="std::ceil({0})",
    floor="std::floot({0})",
    copysign=NotImplemented,
    round="std::round({0})",
    sign="({0} == 0 ? {0} : std::copysign(1, {0}))",
    trunc=NotImplemented,
    conj=NotImplemented,
    real="({0}).real()",
    imag="({0}).imag()",
    complex="std::complex<{typeof_0}>({0}, {1})",
    hypot=NotImplemented,
    # square="std::square({0})",
    sqrt="std::sqrt({0})",
    select="(({0}) ? ({1}) : ({2}))",
    lt="({0}) < ({1})",
    le="({0}) <= ({1})",
    gt="({0}) > ({1})",
    ge="({0}) >= ({1})",
    eq="({0}) == ({1})",
    ne="({0}) != ({1})",
    # is_finite="IsFinite({0})",
    # is_inf="IsInf({0})",
    # is_posinf="IsPosInf({0})",
    # is_neginf="IsNegInf({0})",
    # is_nan="IsNan({0})",
    # is_negzero="IsNegZero({0})",
    # nextafter="NextAfter({0}, {1})",
)

constant_to_target = dict(
    smallest="std::numeric_limits<{type}>::min()",
    largest="std::numeric_limits<{type}>::max()",
    posinf="std::numeric_limits<{type}>::infinity()",
    neginf="-std::numeric_limits<{type}>::infinity()",
)


class Printer(PrinterBase):
    """Printer for C++ target"""

    kind_to_target = kind_to_target

    constant_to_target = constant_to_target

    type_to_target = dict(
        integer8="int8_t",
        integer16="int16_t",
        integer32="int32_t",
        integer64="int64_t",
        integer="int64_t",
        float32="float",
        float64="double",
        float="double",
        complex64="std::complex<float>",
        complex128="std::complex<double>",
        complex="std::complex<double>",
        boolean="bool",
    )

    def make_assignment(self, typ, var, value):
        return f"{typ} {var} = {value};"

    def make_constant(self, like, value):
        return f"{value}"

    def make_argument(self, arg):
        typ = self.get_type(arg)
        return f"{typ} {arg}"

    def make_apply(self, expr, name, tab=""):
        sargs = ", ".join(map(self.make_argument, expr.operands[1:-1]))
        body = self.tostring(expr.operands[-1])
        body_type = self.get_type(expr.operands[-1])
        lines = []
        lines.append(f"{tab}{body_type} {name}({sargs}) {{")
        for a in self.assignments:
            lines.append(f"{tab}    {a}")
        lines.append(f"{tab}    return {body};")
        lines.append(f"{tab}}}")
        return utils.format_cpp("\n".join(lines))


def try_compile(filename):
    import subprocess
    import pathlib
    import tempfile

    _, outfilename = tempfile.mkstemp()

    command = ["g++", "-c", filename, "-o", outfilename]

    try:
        p = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=None,
            stdin=subprocess.PIPE,
            universal_newlines=True,
        )
    except FileNotFoundError as e:
        print('Failed to run "%s": %s"' % (" ".join(command), e))
        return False
    except OSError as e:
        print('Failed to run "%s" - %s"' % (" ".join(command), e.strerror))
        pathlib.Path(outfilename).unlink(missing_ok=True)
        return False

    stdout, stderr = p.communicate()
    status = p.returncode == 0
    if not status:
        print(stdout)
    pathlib.Path(outfilename).unlink(missing_ok=True)
    return status
