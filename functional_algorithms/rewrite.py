import math
import numpy
from . import expr as _expr
from .utils import number_types, value_types, float_types, complex_types, boolean_types


class Printer:

    def __init__(self):
        self.names = dict()
        self.inames = dict()
        self.name_counter = 0

    def getname(self, expr):
        uid = expr._serialized

        if uid in self.inames:
            return self.inames[uid]

        name = f"_M{self.name_counter}_"
        self.name_counter += 1

        self.inames[uid] = name
        self.names[name] = expr
        return name

    def tostring(self, expr):
        if not isinstance(expr, _expr.Expr):
            return str(expr)

        if expr.kind == "symbol":
            return self.getname(expr)
        elif expr.kind == "apply":
            name = expr.operands[0]
            args = expr.operands[1:-1]
            body = expr.operands[-1]
            sname = self.tostring(name)
            sargs = ", ".join([self.getname(a) for a in args])
            return f"apply({sname}, {sargs}, {self.tostring(body)})"
        elif expr.kind == "constant" and 0:
            return self.tostring(expr.operands[0])
        else:
            ops = ", ".join(map(self.tostring, expr.operands))
            return f"{expr.kind}({ops})"


_constant_relop_constant = {
    # >=      >     <=     <     ==       !=
    ("posinf", "posinf"): (True, False, True, False, True, False),
    ("posinf", "neginf"): (True, True, False, False, False, True),
    ("posinf", "largest"): (True, True, False, False, False, True),
    ("posinf", "eps"): (True, True, False, False, False, True),
    ("posinf", "smallest"): (True, True, False, False, False, True),
    ("posinf", "smallest_subnormal"): (True, True, False, False, False, True),
    ("posinf", 1): (True, True, False, False, False, True),
    ("posinf", 0): (True, True, False, False, False, True),
    ("neginf", "posinf"): (False, False, True, True, False, True),
    ("neginf", "neginf"): (True, False, True, False, True, False),
    ("neginf", "largest"): (False, False, True, True, False, True),
    ("neginf", "eps"): (False, False, True, True, False, True),
    ("neginf", "smallest"): (False, False, True, True, False, True),
    ("neginf", "smallest_subnormal"): (False, False, True, True, False, True),
    ("neginf", 1): (False, False, True, True, False, True),
    ("neginf", 0): (False, False, True, True, False, True),
    ("largest", "posinf"): (False, False, True, True, False, True),
    ("largest", "neginf"): (True, True, False, False, False, True),
    ("largest", "largest"): (True, False, True, False, True, False),
    ("largest", "eps"): (True, True, False, False, False, True),
    ("largest", "smallest"): (True, True, False, False, False, True),
    ("largest", "smallest_subnormal"): (True, True, False, False, False, True),
    ("largest", 1): (True, True, False, False, False, True),
    ("largest", 0): (True, True, False, False, False, True),
    ("smallest", "posinf"): (False, False, True, True, False, True),
    ("smallest", "neginf"): (True, True, False, False, False, True),
    ("smallest", "largest"): (False, False, True, True, False, True),
    ("smallest", "eps"): (False, False, True, True, False, True),
    ("smallest", "smallest"): (True, False, True, False, True, False),
    ("smallest", "smallest_subnormal"): (True, True, False, False, False, True),
    ("smallest", 1): (False, False, True, True, False, True),
    ("smallest", 0): (True, True, False, False, False, True),
    ("smallest_subnormal", "posinf"): (False, False, True, True, False, True),
    ("smallest_subnormal", "neginf"): (True, True, False, False, False, True),
    ("smallest_subnormal", "largest"): (False, False, True, True, False, True),
    ("smallest_subnormal", "eps"): (False, False, True, True, False, True),
    ("smallest_subnormal", "smallest"): (False, False, True, True, False, True),
    ("smallest_subnormal", "smallest_subnormal"): (True, False, True, False, True, False),
    ("smallest_subnormal", 1): (False, False, True, True, False, True),
    ("smallest_subnormal", 0): (True, True, False, False, False, True),
    ("eps", "posinf"): (False, False, True, True, False, True),
    ("eps", "neginf"): (True, True, False, False, False, True),
    ("eps", "largest"): (False, False, True, True, False, True),
    ("eps", "eps"): (True, False, True, False, True, False),
    ("eps", "smallest"): (True, True, False, False, False, True),
    ("eps", "smallest_subnormal"): (True, True, False, False, False, True),
    ("eps", 1): (False, False, True, True, False, True),
    ("eps", 0): (True, True, False, False, False, True),
    (1, "posinf"): (False, False, True, True, False, True),
    (1, "neginf"): (True, True, False, False, False, True),
    (1, "largest"): (False, False, True, True, False, True),
    (1, "eps"): (True, True, False, False, False, True),
    (1, "smallest"): (True, True, False, False, False, True),
    (1, "smallest_subnormal"): (True, True, False, False, False, True),
    (1, 1): (True, False, True, False, True, False),
    (1, 0): (True, True, False, False, False, True),
    (0, "posinf"): (False, False, True, True, False, True),
    (0, "neginf"): (True, True, False, False, False, True),
    (0, "largest"): (False, False, True, True, False, True),
    (0, "eps"): (False, False, True, True, False, True),
    (0, "smallest"): (False, False, True, True, False, True),
    (0, "smallest_subnormal"): (False, False, True, True, False, True),
    (0, 1): (False, False, True, True, False, True),
    (0, 0): (True, False, True, False, True, False),
}


_constant_relop_any = {
    # >=      >     <=     <     ==       !=
    ("posinf", "positive"): (True, None, None, False, None, None),
    ("neginf", "positive"): (False, False, True, True, False, True),
    ("largest", "positive"): (None, None, None, None, None, None),
    ("smallest", "positive"): (None, False, True, None, None, None),
    ("smallest_subnormal", "positive"): (None, False, True, None, None, None),
    ("eps", "positive"): (None, None, None, None, None, None),
    (1, "positive"): (None, None, None, None, None, None),
    (0, "positive"): (False, False, True, True, False, True),
    ("posinf", "nonnegative"): (True, None, None, False, None, None),
    ("neginf", "nonnegative"): (False, False, True, True, False, True),
    ("largest", "nonnegative"): (None, None, None, None, None, None),
    ("smallest", "nonnegative"): (None, False, True, None, None, None),
    ("smallest_subnormal", "nonnegative"): (None, False, True, None, None, None),
    ("eps", "nonnegative"): (None, None, None, None, None, None),
    (1, "nonnegative"): (None, None, None, None, None, None),
    (0, "nonnegative"): (None, False, True, None, None, None),
    ("posinf", "negative"): (True, True, False, False, False, True),
    ("neginf", "negative"): (None, False, True, None, None, None),
    ("largest", "negative"): (True, True, False, False, False, True),
    ("smallest", "negative"): (True, True, False, False, False, True),
    ("smallest_subnormal", "negative"): (True, True, False, False, False, True),
    ("eps", "negative"): (True, True, False, False, False, True),
    (1, "negative"): (True, True, False, False, False, True),
    (0, "negative"): (True, True, False, False, False, True),
    ("posinf", "nonpositive"): (True, True, False, False, False, True),
    ("neginf", "nonpositive"): (None, False, True, None, None, None),
    ("largest", "nonpositive"): (True, True, False, False, False, True),
    ("smallest", "nonpositive"): (True, True, False, False, False, True),
    ("smallest_subnormal", "nonpositive"): (True, True, False, False, False, True),
    ("eps", "nonpositive"): (True, True, False, False, False, True),
    (1, "nonpositive"): (True, True, False, False, False, True),
    (0, "nonpositive"): (True, None, None, False, None, None),
    ("posinf", "finite"): (True, True, False, False, False, True),
    ("neginf", "finite"): (False, False, True, True, False, True),
    ("largest", "finite"): (True, None, None, False, None, None),
    ("smallest", "finite"): (None, None, None, None, None, None),
    ("smallest_subnormal", "finite"): (None, None, None, None, None, None),
    ("eps", "finite"): (None, None, None, None, None, None),
    (1, "finite"): (None, None, None, None, None, None),
    (0, "finite"): (None, None, None, None, None, None),
}

for (_lhs, _rhs), (_ge, _gt, _le, _lt, _eq, _ne) in dict(_constant_relop_any.items()).items():
    _constant_relop_any[_rhs, _lhs] = (_le, _lt, _ge, _gt, _eq, _ne)

_any_relop_any = {
    # >=      >     <=     <     ==       !=
    ("finite", "finite"): (None, None, None, None, None, None),
    ("finite", "positive"): (None, None, None, None, None, None),
    ("finite", "negative"): (None, None, None, None, None, None),
    ("finite", "nonpositive"): (None, None, None, None, None, None),
    ("finite", "nonnegative"): (None, None, None, None, None, None),
    ("positive", "finite"): (None, None, None, None, None, None),
    ("positive", "positive"): (None, None, None, None, None, None),
    ("positive", "negative"): (True, True, False, False, False, True),
    ("positive", "nonpositive"): (True, True, False, False, False, True),
    ("positive", "nonnegative"): (None, None, None, None, None, None),
    ("negative", "finite"): (None, None, None, None, None, None),
    ("negative", "negative"): (None, None, None, None, None, None),
    ("negative", "positive"): (False, False, True, True, False, True),
    ("negative", "nonnegative"): (False, False, True, True, False, True),
    ("negative", "nonpositive"): (None, None, None, None, None, None),
    ("nonnegative", "finite"): (None, None, None, None, None, None),
    ("nonnegative", "positive"): (None, None, None, None, None, None),
    ("nonnegative", "nonnegative"): (None, None, None, None, None, None),
    ("nonnegative", "negative"): (True, True, False, False, False, True),
    ("nonnegative", "nonpositive"): (True, True, False, False, False, True),
    ("nonpositive", "finite"): (None, None, None, None, None, None),
    ("nonpositive", "nonpositive"): (None, None, None, None, None, None),
    ("nonpositive", "negative"): (None, None, None, None, None, None),
    ("nonpositive", "positive"): (False, False, True, True, False, True),
    ("nonpositive", "nonnegative"): (False, False, True, True, False, True),
}


class Rewriter:

    patterns = {
        "lt(constant(0, _M0_), multiply(divide(sqrt(constant(largest, _M0_)), constant(8, _M0_)),"
        " constant(1000000000000.0, _M0_)))": (
            "ctx.constant(True, ctx.symbol(None, 'boolean'))",
            "constant(True, _M1_)",
        ),
        "eq(maximum(constant(1, _M0_), abs(_M0_)), minimum(constant(1, _M0_), abs(_M0_)))": (
            "ctx.eq(ctx.constant(1, _M0_), abs(_M0_))",
            "eq(constant(1, _M0_), abs(_M0_))",
        ),
        "select(logical_and(ge(abs(_M0_), multiply(divide(sqrt(constant(largest, _M0_)), constant(8, _M0_)),"
        " constant(1e-06, _M0_))), logical_not(eq(abs(_M0_), constant(posinf, _M0_)))),"
        " divide(constant(0, _M0_), abs(_M0_)), constant(0, _M0_))": (
            "ctx.constant(0, _M0_)",
            "constant(0, _M0_)",
        ),
    }

    def __init__(self):
        self._printer = Printer()

    def __call__(self, expr):
        result = getattr(self, expr.kind, self._notimpl)(expr)
        if result is not None:
            return result

        # Apply static pattern rewrites:
        s = self._printer.tostring(expr)
        if s in self.patterns:
            replacement, expected = self.patterns[s]
            r = eval(replacement, dict(ctx=expr.context, **self._printer.names))
            rs = self._printer.tostring(r)
            if rs == expected:
                return r
            raise ValueError(f"expected rewrite of `{s}` is `{expected}`, got `{rs}`")

    def _todo(self, expr):
        print(f'TODO: rewrite {expr.kind}({", ".join(op.kind for op in expr.operands)})')

    def _notimpl(self, expr):
        print(f'NOTIMPL: rewrite {expr.kind}({", ".join(op.kind for op in expr.operands)})')

    def _eval(self, like, opname, *args):
        typ = like.get_type()
        if typ.kind in {"float", "complex", "integer", "boolean"} and typ.bits is not None:
            dtype = typ.asdtype()
            if dtype is not None:
                op = getattr(numpy, opname)
                return like.context.constant(op(*map(dtype, args)), like)
        if typ.kind == "float":
            if opname == "square":
                op = lambda x: x * x
            else:
                op = getattr(math, opname)
            return like.context.constant(op(*args), like)

    def absolute(self, expr):
        (x,) = expr.operands

        if x.kind == "absolute":
            return x

        if x.kind == "constant":
            value, like = x.operands
            if isinstance(value, value_types):
                return x.context.constant(abs(value), like)

    def apply(self, expr):
        pass

    def asin_acos_kernel(self, expr):
        pass

    def acos(self, expr):
        pass

    def acosh(self, expr):
        pass

    def asin(self, expr):
        pass

    def asinh(self, expr):
        pass

    def atan(self, expr):
        pass

    def atan2(self, expr):
        pass

    def atanh(self, expr):
        pass

    def sin(self, expr):
        pass

    def sinh(self, expr):
        pass

    def cos(self, expr):
        pass

    def cosh(self, expr):
        pass

    def tan(self, expr):
        pass

    def tanh(self, expr):
        pass

    def exp(self, expr):
        pass

    def _binary_op(self, expr, op):
        x, y = expr.operands

        if x.kind == "constant" and y.kind == "constant":
            xvalue, xlike = x.operands
            yvalue, ylike = y.operands
            if isinstance(xvalue, number_types) and isinstance(yvalue, number_types):
                r = op(xvalue, yvalue)
                return expr.context.constant(r, xlike)

    def add(self, expr):
        result = self._binary_op(expr, lambda x, y: x + y)
        if result is not None:
            return result

        x, y = expr.operands
        for x_, y_ in [(x, y), (y, x)]:
            if x_.kind == "constant":
                value, like = x_.operands
                if isinstance(value, number_types) and value == 0:
                    return y_

    def subtract(self, expr):
        result = self._binary_op(expr, lambda x, y: x - y)

        if result is not None:
            return result

        x, y = expr.operands
        for x_, y_, s in [(x, y, -1), (y, x, 1)]:
            if x_.kind == "constant":
                value, like = x_.operands
                if isinstance(value, number_types) and value == 0:
                    return -y_ if s == -1 else y_

    def multiply(self, expr):
        result = self._binary_op(expr, lambda x, y: x * y)

        if result is not None:
            return result

        x, y = expr.operands
        for x_, y_ in [(x, y), (y, x)]:
            if x_.kind == "constant":
                value, like = x_.operands
                if isinstance(value, number_types) and value == 1:
                    return y_

    def minimum(self, expr):
        return self._binary_op(expr, lambda x, y: min(x, y))

    def maximum(self, expr):
        return self._binary_op(expr, lambda x, y: max(x, y))

    def divide(self, expr):
        x, y = expr.operands
        if y.kind == "constant":
            value, like = y.operands
            if isinstance(value, number_types) and value == 1:
                return x

    def complex(self, expr):
        pass

    def constant(self, expr):
        value, like = expr.operands
        typ = like.get_type()
        if typ.kind in {"float", "complex"} and typ.bits is not None:
            dtype = typ.asdtype()
            if dtype is not None:
                if isinstance(value, number_types):
                    if not isinstance(value, dtype):
                        return expr.context.constant(dtype(value), like)
                elif isinstance(value, str):
                    if value == "posinf":
                        return expr.context.constant(dtype(numpy.inf), like)
                    if value == "neginf":
                        return expr.context.constant(-dtype(numpy.inf), like)
                    if value == "pi":
                        return expr.context.constant(dtype(numpy.pi), like)
                    if value in {"undefined", "nan"}:
                        return expr.context.constant(dtype(numpy.nan), like)
                    fi = numpy.finfo(dtype)
                    if value == "eps":
                        return expr.context.constant(dtype(fi.eps), like)
                    if value == "largest":
                        return expr.context.constant(dtype(fi.max), like)
                    if value == "smallest":
                        return expr.context.constant(dtype(fi.smallest_normal), like)
                    if value == "smallest_subnormal":
                        return expr.context.constant(dtype(fi.smallest_subnormal), like)
        elif typ.kind == "float" and typ.bits is None:
            if not isinstance(value, float) and isinstance(value, number_types):
                return expr.context.constant(float(value), like)
        elif typ.kind == "complex" and typ.bits is None:
            if not isinstance(value, complex) and isinstance(value, number_types):
                return expr.context.constant(complex(value), like)
        elif typ.kind == "boolean" and typ.bits is not None:
            if isinstance(value, boolean_types) and not isinstance(value, bool):
                return expr.context.constant(bool(value), like)

    def upcast(self, expr):
        (x,) = expr.operands
        if x.kind == "downcast":
            return x.operands[0]

    def downcast(self, expr):
        (x,) = expr.operands
        if x.kind == "upcast":
            return x.operands[0]

    def log(self, expr):
        (x,) = expr.operands
        if x.kind == "constant":
            value, like = x.operands
            if isinstance(value, number_types) and value == 1:
                return x.context.constant(0, like)

    def log1p(self, expr):
        (x,) = expr.operands
        if x.kind == "constant":
            value, like = x.operands
            if isinstance(value, number_types) and value == 0:
                return x

    def logical_and(self, expr):
        x, y = expr.operands
        ctx = x.context
        for x_, y_ in [(x, y), (y, x)]:
            if x_.kind == "constant":
                value, like = x_.operands
                if isinstance(value, bool):
                    return y_ if value else ctx.constant(False, ctx.symbol(None, "boolean"))

    def logical_or(self, expr):
        x, y = expr.operands
        ctx = x.context
        for x_, y_ in [(x, y), (y, x)]:
            if x_.kind == "constant":
                value, like = x_.operands
                if isinstance(value, bool):
                    if value:
                        return ctx.constant(True, ctx.symbol(None, "boolean"))
                    return y_

    def logical_not(self, expr):
        (x,) = expr.operands
        if x.kind == "constant":
            value, like = x.operands
            if isinstance(value, bool):
                return x.context.constant(not value, like)

    def negative(self, expr):
        (x,) = expr.operands
        if x.kind == "constant":
            value, like = x.operands
            if isinstance(value, number_types):
                return x.context.constant(-value, like)

        if x.kind == "negative":
            return x.operands[0]

    def conjugate(self, expr):

        (x,) = expr.operands

        if x.kind == "constant":
            value, like = x.operands
            if isinstance(value, float_types):
                return x
            if isinstance(value, complex_types):
                return x.context.constant(value.conjugate(), like)

        if x.kind == "complex":
            real, imag = x.operands
            return x.context.complex(real, -imag)

        if x.kind == "conjugate":
            return x

    def real(self, expr):

        (x,) = expr.operands

        if x.kind == "constant":
            self._todo(expr)

        if x.kind == "complex":
            return x.operands[0]

    def imag(self, expr):

        (x,) = expr.operands

        if x.kind == "constant":
            self._todo(expr)

        if x.kind == "complex":
            return x.operands[1]

    def _compare(self, expr, relop, relop_index, swap_relop_index):
        x, y = expr.operands
        if x.kind == "constant":
            xvalue, xlike = x.operands
            if y.kind == "constant":
                yvalue, ylike = y.operands

                if isinstance(xvalue, _expr.Expr) or isinstance(yvalue, _expr.Expr):
                    r = self._compare(relop(xvalue, yvalue), relop, relop_index, swap_relop_index)
                    if isinstance(r, bool):
                        return expr.context.constant(r)
                else:
                    r = _constant_relop_constant.get((xvalue, yvalue))
                    if r is not None:
                        return expr.context.constant(r[relop_index])

                if isinstance(xvalue, value_types) and isinstance(yvalue, value_types):
                    r = bool(relop(xvalue, yvalue))
                    return expr.context.constant(r)

            elif isinstance(xvalue, number_types):
                for prop in ["positive", "negative", "nonpositive", "nonnegative", "finite"]:
                    if y._is(prop):
                        r = _constant_relop_any.get((xvalue, prop))
                        if r is not None:
                            r = r[relop_index]
                            if r is not None:
                                return expr.context.constant(r)

        elif y.kind == "constant":
            yvalue, ylike = y.operands
            if isinstance(yvalue, number_types):
                for prop in ["positive", "negative", "nonpositive", "nonnegative", "finite"]:
                    if x._is(prop):
                        r = _constant_relop_any.get((yvalue, prop))
                        if r is not None:
                            r = r[swap_relop_index]
                            if r is not None:
                                return expr.context.constant(r)
        else:
            for xprop, yprop in [
                ("positive", "negative"),
                ("positive", "nonnegative"),
                ("positive", "nonpositive"),
                ("negative", "positive"),
                ("negative", "nonpositive"),
                ("negative", "nonnegative"),
                ("nonpositive", "negative"),
                ("nonpositive", "nonnegative"),
                ("nonpositive", "positive"),
                ("nonnegative", "positive"),
                ("nonnegative", "nonpositive"),
                ("nonnegative", "negative"),
            ]:
                if x._is(xprop) and y._is(yprop):
                    r = _any_relop_any.get((xprop, yprop))
                    if r is not None:
                        r = r[relop_index]
                        if r is not None:
                            return expr.context.constant(r)

    def ge(self, expr):
        return self._compare(expr, lambda x, y: x >= y, 0, 2)

    def gt(self, expr):
        return self._compare(expr, lambda x, y: x > y, 1, 3)

    def le(self, expr):
        return self._compare(expr, lambda x, y: x <= y, 2, 0)

    def lt(self, expr):
        return self._compare(expr, lambda x, y: x < y, 3, 1)

    def eq(self, expr):
        return self._compare(expr, lambda x, y: x == y, 4, 4)

    def ne(self, expr):
        return self._compare(expr, lambda x, y: x != y, 5, 5)

    def select(self, expr):
        cond, x, y = expr.operands

        if cond.kind == "constant":
            value, kind = cond.operands
            if isinstance(value, bool):
                return x if value else y

            # self._todo(expr)

    def sqrt(self, expr):
        (x,) = expr.operands
        if x.kind == "constant":
            value, like = x.operands
            if isinstance(value, number_types):
                if value == 0 or value == 1:
                    return x
                return self._eval(like, "sqrt", value)

    def sign(self, expr):
        (x,) = expr.operands
        if x.kind == "constant":
            value, like = x.operands
            if isinstance(value, float_types):
                ctx = x.context
                if value == 0:
                    return ctx.constant(0, like)
                return ctx.constant(1 if value > 0 else -1, like)

        if x.kind == "sign":
            return x

    def square(self, expr):
        (x,) = expr.operands
        if x.kind == "constant":
            value, like = x.operands
            if isinstance(value, number_types):
                return self._eval(like, "square", value)

    def symbol(self, expr):
        pass

    def hypot(self, expr):
        pass


def rewrite(expr):
    """Return rewritten expression, otherwise return None."""

    rewriter = Rewriter()

    last_result = None
    result = expr
    while True:
        result = rewriter(result)
        if result is not None:
            last_result = result
        else:
            break

    return last_result


def __rewrite_modifier__(expr):
    result = rewrite(expr)
    return expr if result is None else result


class RewriteContext:
    """Caches rewrite results."""

    def __init__(self):
        self.cache = {}

    def __contains__(self, expr):
        return expr.key in self.cache

    def __call__(self, original, new=None):
        if new is not None:
            if original.key in self.cache:
                cached = self.cache[original.key]
                assert new is cached
            else:
                self.cache[original.key] = new
                force_ref = original.props.get("force_ref")
                if force_ref is not None:
                    new.props.update(force_ref=force_ref)
                ref_name = original.props.get("reference_name")
                if isinstance(ref_name, str):
                    new.props.update(reference_name=ref_name)
            return new
        else:
            return self.cache[original.key]


class Substitute:

    def __init__(self, matches, replacements):
        self.matches = matches
        self.replacements = replacements

    @classmethod
    def fromdict(cls, dct):
        matches, replacements = [], []
        for match, replacement in dct.items():
            matches.append(match)
            replacements.append(replacement)
        return cls(matches, replacements)

    def __rewrite_modifier__(self, expr):
        for match, replacement in zip(self.matches, self.replacements):
            is_a_match = False
            if isinstance(match, str):
                if expr.kind in {"symbol", "constant"} and isinstance(expr.operands[0], str):
                    is_a_match = match == expr.operands[0]
            elif isinstance(match, _expr.Expr):
                is_a_match = match.key == expr.key
            elif callable(match):
                is_a_match = match(expr)
            else:
                raise NotImplementedError(f"{type(match)=}")

            if is_a_match:
                if isinstance(replacement, _expr.Expr):
                    return replacement
                elif isinstance(replacement, value_types) or replacement in _expr.known_constant_names:
                    return expr.context.constant(replacement, expr)
                elif isinstance(replacement, str):
                    typ = expr.get_type()
                    assert typ is not None, expr
                    return expr.context.symbol(replacement, typ=typ)
                else:
                    raise NotImplementedError(f"{type(replacement)=}")
        return expr
