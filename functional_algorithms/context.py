import inspect
import sys
import types
import typing
import warnings
from collections import defaultdict
from .utils import UNSPECIFIED
from .expr import Expr, make_constant, make_symbol, make_apply


class Context:

    def __init__(self, paths=[]):
        """Parameters
        ----------
        paths : list
          Module-like objects providing implementations to functions
          not defined by the context instance.
        """
        self._expressions = {}
        self._stack_name = ""
        self._stack_call_count = defaultdict(int)
        self._ref_values = {}
        self._paths = paths

    def _register_expression(self, expr):
        # Expressions are singletons.  Notice that Expr does not
        # implement __hash__/__eq__ on purpose: __eq__ is used to
        # construct equality expressions.
        prev = self._expressions.get(expr._serialized)
        if prev is None:
            prev = self._expressions[expr._serialized] = expr
        return prev

    def _update_expression_ref(self, expr, ref):
        expr.props.update(ref=ref)
        self._ref_values[ref] = expr

    def _update_refs(self, depth=1):
        prefix = self._stack_name + "_" + str(self._stack_call_count[self._stack_name]) + "_" if self._stack_name else ""
        frame = sys._getframe(depth)
        for name, obj in frame.f_locals.items():
            if isinstance(obj, Expr):
                ref = obj.props.get("ref", UNSPECIFIED)
                if ref is UNSPECIFIED:
                    ref = name
                if isinstance(ref, str):
                    if ref in self._ref_values:
                        if self._ref_values[ref] is obj:
                            # obj must be defined in a caller
                            continue
                        ref = prefix + ref
                    # sanity check that existing ref is unused
                    assert ref not in self._ref_values, (ref, prefix, str(self._ref_values[ref]), str(obj))
                    self._update_expression_ref(obj, ref)

    def __call__(self, expr):
        """Post-process tracing of a function that defines a functional algorithm.

        The pros-processing involves:
        - assigning variable names to sub-expression references.
        """
        self._update_refs(depth=2)
        return expr

    def trace(self, func, *args, props=UNSPECIFIED):
        """Trace a Python function to a functional graph.

        Use `*args` to override annotations or argument names of the
        function.

        It is assumed that the function implements an algorithm
        definition, that is, its first argument must be a Context
        instance that defines available operations that can be used in
        the function.
        """
        sig = inspect.signature(func)
        default_typ = UNSPECIFIED
        new_args = []
        for i, (name, param) in enumerate(sig.parameters.items()):
            if i == 0:
                if name != "ctx":
                    warnings.warn(f"The first argument of {func.__name__} is expected to have a name `ctx` but got {name}")
                continue
            assert param.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}, param.kind
            if i <= len(args):
                # overwrite argument name and type
                a = args[i - 1]
                if isinstance(a, str):
                    if ":" in a:
                        name, annot = a.split(":", 1)
                        name = name.strip()
                        param = param.replace(name=name if name else param.name, annotation=annot.strip())
                    else:
                        param = param.replace(name=a.strip())
                elif isinstance(a, type):
                    param = param.replace(annotation=a.__name__)
                else:
                    raise NotImplementedError((a, type(a)))
            if param.annotation is inspect.Parameter.empty:
                typ = default_typ
            else:
                typ = param.annotation
                if isinstance(typ, types.UnionType):
                    typ = typing.get_args(typ)[0]
                assert isinstance(typ, (type, str)), typ
            if default_typ is UNSPECIFIED:
                default_typ = typ
            a = self.symbol(param.name, typ, ref=param.name)
            new_args.append(a)
        args = tuple(new_args)
        name = self.symbol(func.__name__, ref=func.__name__)
        return make_apply(self, name, args, func(self, *args), props)
        new_args = []
        for a in args:
            if isinstance(a, str):
                if ":" in a:
                    name, typ = a.split(":")
                    name = name.strip()
                    typ = typ.strip()
                else:
                    name = a
                    typ = UNSPECIFIED
                a = self.symbol(name, typ, ref=name)
            assert isinstance(a, Expr) and a.kind == "symbol", a
            assert a.props.get("ref") == a.operands[0], a
            new_args.append(a)
        args = tuple(new_args)
        name = self.symbol(func.__name__, ref=func.__name__)
        return make_apply(self, name, args, func(self, *args), props)

    def symbol(self, name, typ=UNSPECIFIED, ref=UNSPECIFIED):
        if typ is UNSPECIFIED:
            typ = "float"
        props = dict(ref=ref)
        return make_symbol(self, name, typ, props)

    def constant(self, value, like_expr, ref=UNSPECIFIED):
        props = dict(ref=ref)
        return make_constant(self, value, like_expr, props)

    def apply(self, name, args, result, ref=UNSPECIFIED):
        props = dict(ref=ref)
        return make_apply(self, name, args, result, props)

    # for element-wise functions, use Python array API standard and
    # Python operator for naming conventions, logical operator names
    # use also CamelCase

    def abs(self, x, ref=UNSPECIFIED):
        return Expr(self, "abs", (x,), dict(ref=ref))

    def negative(self, x, ref=UNSPECIFIED):
        return Expr(self, "negative", (x,), dict(ref=ref))

    neg = negative

    def positive(self, x, ref=UNSPECIFIED):
        return Expr(self, "positive", (x,), dict(ref=ref))

    pos = positive

    def add(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "add", (x, y), dict(ref=ref))

    def subtract(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "subtract", (x, y), dict(ref=ref))

    sub = subtract

    def multiply(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "multiply", (x, y), dict(ref=ref))

    mul = multiply

    def divide(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "divide", (x, y), dict(ref=ref))

    div = divide

    def reminder(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "reminder", (x, y), dict(ref=ref))

    def floor_divide(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "floor_divide", (x, y), dict(ref=ref))

    def pow(self, x, y, ref=UNSPECIFIED):
        if isinstance(y, float) and y == 0.5:
            return self.sqrt(x, ref=ref)
        if isinstance(y, int) and y == 2:
            return self.square(x, ref=ref)
        return Expr(self, "pow", (x, y), dict(ref=ref))

    def logical_and(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "logical_and", (x, y), dict(ref=ref))

    And = logical_and

    def logical_or(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "logical_or", (x, y), dict(ref=ref))

    Or = logical_or

    def logical_xor(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "logical_xor", (x, y), dict(ref=ref))

    Xor = logical_xor

    def logical_not(self, x, ref=UNSPECIFIED):
        return Expr(self, "logical_not", (x,), dict(ref=ref))

    Not = logical_not

    def invert(self, ref=UNSPECIFIED):
        return Expr(self, "bitwise_invert", (self,), dict(ref=ref))

    def bitwise_and(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "bitwise_and", (x, y), dict(ref=ref))

    def bitwise_or(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "bitwise_or", (x, y), dict(ref=ref))

    def bitwise_xor(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "bitwise_xor", (x, y), dict(ref=ref))

    def bitwise_left_shift(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "bitwise_left_shift", (x, y), dict(ref=ref))

    def bitwise_right_shift(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "bitwise_right_shift", (x, y), dict(ref=ref))

    def lt(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "lt", (x, y), dict(ref=ref))

    less = lt

    def le(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "le", (x, y), dict(ref=ref))

    less_equal = le

    def gt(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "gt", (x, y), dict(ref=ref))

    greater = gt

    def ge(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "ge", (x, y), dict(ref=ref))

    greater_equal = ge

    def eq(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "eq", (x, y), dict(ref=ref))

    equal = eq

    def ne(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "ne", (x, y), dict(ref=ref))

    not_equal = ne

    def maximum(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "maximum", (x, y), dict(ref=ref))

    max = maximum

    def minimum(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "minimum", (x, y), dict(ref=ref))

    min = minimum

    def acos(self, x, ref=UNSPECIFIED):
        return Expr(self, "acos", (x,), dict(ref=ref))

    def acosh(self, x, ref=UNSPECIFIED):
        return Expr(self, "acosh", (x,), dict(ref=ref))

    def asin(self, x, ref=UNSPECIFIED):
        return Expr(self, "asin", (x,), dict(ref=ref))

    def asinh(self, x, ref=UNSPECIFIED):
        return Expr(self, "asinh", (x,), dict(ref=ref))

    def atan(self, x, ref=UNSPECIFIED):
        return Expr(self, "atan", (x,), dict(ref=ref))

    def atanh(self, x, ref=UNSPECIFIED):
        return Expr(self, "atanh", (x,), dict(ref=ref))

    def atan2(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "atan2", (x, y), dict(ref=ref))

    def cos(self, x, ref=UNSPECIFIED):
        return Expr(self, "cos", (x,), dict(ref=ref))

    def cosh(self, x, ref=UNSPECIFIED):
        return Expr(self, "cosh", (x,), dict(ref=ref))

    def sin(self, x, ref=UNSPECIFIED):
        return Expr(self, "sin", (x,), dict(ref=ref))

    def sinh(self, x, ref=UNSPECIFIED):
        return Expr(self, "sinh", (x,), dict(ref=ref))

    def tan(self, x, ref=UNSPECIFIED):
        return Expr(self, "tan", (x,), dict(ref=ref))

    def tanh(self, x, ref=UNSPECIFIED):
        return Expr(self, "tanh", (x,), dict(ref=ref))

    def exp(self, x, ref=UNSPECIFIED):
        return Expr(self, "exp", (x,), dict(ref=ref))

    def expm1(self, x, ref=UNSPECIFIED):
        return Expr(self, "expm1", (x,), dict(ref=ref))

    def log(self, x, ref=UNSPECIFIED):
        return Expr(self, "log", (x,), dict(ref=ref))

    def log1p(self, x, ref=UNSPECIFIED):
        return Expr(self, "log1p", (x,), dict(ref=ref))

    def log2(self, x, ref=UNSPECIFIED):
        return Expr(self, "log2", (x,), dict(ref=ref))

    def log10(self, x, ref=UNSPECIFIED):
        return Expr(self, "log10", (x,), dict(ref=ref))

    def ceil(self, x, ref=UNSPECIFIED):
        return Expr(self, "ceil", (x,), dict(ref=ref))

    def floor(self, x, ref=UNSPECIFIED):
        return Expr(self, "floor", (x,), dict(ref=ref))

    def copysign(self, x, ref=UNSPECIFIED):
        return Expr(self, "copysign", (x,), dict(ref=ref))

    def round(self, x, ref=UNSPECIFIED):
        return Expr(self, "round", (x,), dict(ref=ref))

    def sign(self, x, ref=UNSPECIFIED):
        return Expr(self, "sign", (x,), dict(ref=ref))

    def trunc(self, x, ref=UNSPECIFIED):
        return Expr(self, "trunc", (x,), dict(ref=ref))

    def conj(self, x, ref=UNSPECIFIED):
        return Expr(self, "conj", (x,), dict(ref=ref))

    def real(self, x, ref=UNSPECIFIED):
        return Expr(self, "real", (x,), dict(ref=ref))

    def imag(self, x, ref=UNSPECIFIED):
        return Expr(self, "imag", (x,), dict(ref=ref))

    def complex(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "complex", (x, y), dict(ref=ref))

    def hypot(self, x, y, ref=UNSPECIFIED):
        return Expr(self, "hypot", (x, y), dict(ref=ref))

    def square(self, x, ref=UNSPECIFIED):
        return Expr(self, "square", (x,), dict(ref=ref))

    def sqrt(self, x, ref=UNSPECIFIED):
        return Expr(self, "sqrt", (x,), dict(ref=ref))

    def select(self, cond, x, y, ref=UNSPECIFIED):
        return Expr(self, "select", (cond, x, y), dict(ref=ref))
