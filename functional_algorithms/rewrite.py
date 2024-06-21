from . import expr as _expr


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


class Rewriter:

    patterns = {
        "lt(constant(0, _M0_), multiply(divide(sqrt(constant(largest, _M0_)), constant(8, _M0_)), constant(1000000000000.0, _M0_)))": (
            "ctx.constant(True, ctx.symbol(None, 'boolean'))",
            "constant(True, _M1_)",
        ),
        "eq(maximum(constant(1, _M0_), abs(_M0_)), minimum(constant(1, _M0_), abs(_M0_)))": (
            "ctx.eq(ctx.constant(1, _M0_), abs(_M0_))",
            "eq(constant(1, _M0_), abs(_M0_))",
        ),
        "select(logical_and(ge(abs(_M0_), multiply(divide(sqrt(constant(largest, _M0_)), constant(8, _M0_)), constant(1e-06, _M0_))), logical_not(eq(abs(_M0_), constant(posinf, _M0_)))), divide(constant(0, _M0_), abs(_M0_)), constant(0, _M0_))": (
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

    def abs(self, expr):
        (x,) = expr.operands

        if x.kind == "abs":
            return x

        if x.kind == "constant":
            value, like = x.operands
            if isinstance(value, (int, float)):
                return x.context.constant(abs(value), like)

    def apply(self, expr):
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

    def _binary_op(self, expr, op):
        x, y = expr.operands

        if x.kind == "constant" and y.kind == "constant":
            xvalue, xlike = x.operands
            yvalue, ylike = y.operands

            if isinstance(xvalue, (int, float)) and isinstance(yvalue, (int, float)):
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
                if isinstance(value, (int, float)) and value == 0:
                    return y_

    def subtract(self, expr):
        result = self._binary_op(expr, lambda x, y: x - y)

        if result is not None:
            return result

        x, y = expr.operands
        for x_, y_, s in [(x, y, -1), (y, x, 1)]:
            if x_.kind == "constant":
                value, like = x_.operands
                if isinstance(value, (int, float)) and value == 0:
                    return -y_ if s == -1 else y_

    def multiply(self, expr):
        result = self._binary_op(expr, lambda x, y: x * y)

        if result is not None:
            return result

        x, y = expr.operands
        for x_, y_ in [(x, y), (y, x)]:
            if x_.kind == "constant":
                value, like = x_.operands
                if isinstance(value, (int, float)) and value == 1:
                    return y_

    def minimum(self, expr):
        return self._binary_op(expr, lambda x, y: min(x, y))

    def maximum(self, expr):
        return self._binary_op(expr, lambda x, y: max(x, y))

    def divide(self, expr):
        x, y = expr.operands
        if y.kind == "constant":
            value, like = y.operands
            if isinstance(value, (int, float)) and value == 1:
                return x

    def complex(self, expr):
        pass

    def constant(self, expr):
        pass

    def log(self, expr):
        (x,) = expr.operands
        if x.kind == "constant":
            value, like = x.operands
            if isinstance(value, (int, float)) and value == 1:
                return x.context.constant(0, like)

    def log1p(self, expr):
        (x,) = expr.operands
        if x.kind == "constant":
            value, like = x.operands
            if isinstance(value, (int, float)) and value == 0:
                return x

    def logical_and(self, expr):
        x, y = expr.operands
        ctx = x.context
        for x_, y_ in [(x, y), (y, x)]:
            if x_.kind == "constant":
                value, like = x_.operands
                if isinstance(value, bool):
                    return y_ if value else ctx.constant(False, ctx.symbol(None, "boolean"))

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
            if isinstance(value, (int, float)):
                return x.context.constant(-value, like)

        if x.kind == "negative":
            return x.operands[0]

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

    def _compare(self, expr, relop):
        x, y = expr.operands

        if x.kind == "constant" and y.kind == "constant":
            xvalue, xlike = x.operands
            yvalue, ylike = y.operands

            if isinstance(xvalue, (int, float)) and isinstance(yvalue, (int, float)):

                r = relop(xvalue, yvalue)
                ctx = expr.context
                return ctx.constant(r, ctx.symbol(None, "boolean"))

            self._todo(expr)

    def ge(self, expr):
        return self._compare(expr, lambda x, y: x >= y)

    def gt(self, expr):
        return self._compare(expr, lambda x, y: x > y)

    def le(self, expr):
        return self._compare(expr, lambda x, y: x <= y)

    def lt(self, expr):
        return self._compare(expr, lambda x, y: x < y)

    def eq(self, expr):
        return self._compare(expr, lambda x, y: x == y)

    def ne(self, expr):
        return self._compare(expr, lambda x, y: x != y)

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
            if isinstance(value, (int, float)):
                if value == 0 or value == 1:
                    return x

    def sign(self, expr):
        (x,) = expr.operands
        if x.kind == "constant":
            value, like = x.operands
            if isinstance(value, (int, float)):
                ctx = x.context
                if value == 0:
                    return ctx.constant(0, like)
                return ctx.constant(1 if value > 0 else -1, like)

        if x.kind == "sign":
            return x

    def square(self, expr):
        (x,) = expr.operands
        if x.kind == "constant":
            self._todo(expr)

    def symbol(self, expr):
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
