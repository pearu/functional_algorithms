import inspect
import sys
import types
import typing
import warnings
from collections import defaultdict
from .utils import UNSPECIFIED, boolean_types, float_types, complex_types
from .expr import Expr, make_constant, make_symbol, make_apply, known_expression_kinds
from .typesystem import Type


class Context:

    def __init__(self, paths=[], enable_alt=None, default_constant_type=None, parameters=None):
        """Parameters
        ----------
        paths : list
          Module-like objects providing implementations to functions
          not defined by the context instance.
        enable_alt : bool
          When True, enable alternative context for constant
          expression values. Enable when target printer supports
          constant target to reduce evaluating constant expressions in
          runtime.
        default_constant_type : {str, Type}
          Default type for constant expressions. Use together with
          enable_alt=True.
        parameters : dict
          Parameters dictionary that can be used to parameterize
          algorithms.
        """
        self._expressions = {}
        self._stack_name = ""
        self._stack_call_count = defaultdict(int)
        self._ref_values = {}  # TODO: move to printer context
        self._paths = paths
        self._alt = None
        self._enable_alt = enable_alt
        self._default_constant_type = default_constant_type
        self._default_like = None
        self.parameters = parameters or {}
        if "using" not in self.parameters:
            self.parameters["using"] = set()

        for k in self.parameters:
            if k.startswith("use_upcast_") or k.startswith("use_downcast_") or k.startswith("use_native_"):
                kind = k.split("_", 2)[-1]
                if kind not in known_expression_kinds:
                    warnings.warn(f"Context parameter with unknown kind: {k}")

    @property
    def alt(self):
        """Return alternate context that can be used to tracing constant
        expressions that may use different target printer.
        """
        if self._alt is None and self._enable_alt:
            self._alt = Context(paths=self._paths, enable_alt=False, default_constant_type=self._default_constant_type)
        return self._alt

    @property
    def default_like(self):
        if self._default_like is None:
            if self._default_constant_type is not None:
                typ = self._default_constant_type
                if not isinstance(typ, Type):
                    typ = Type.fromobject(self, typ)
                self._default_like = self.symbol(None, typ)
        return self._default_like

    def _register_expression(self, expr):
        # Expressions are singletons.  Notice that Expr does not
        # implement __hash__/__eq__ on purpose: __eq__ is used to
        # construct equality expressions. Also, __bool__ raises an
        # exception.
        prev = self._expressions.get(expr._serialized)
        if prev is None:
            prev = self._expressions[expr._serialized] = expr
            # origin is either an empty string (top stack) or a
            # function name modified with an identifier unique to the
            # instance of calling the function. This allows
            # differentiation of function local variables and caller
            # variables.
            prev.props["origin"] = self._stack_name

            # sanity check for detecting reference conflicts
            ref = prev.props.get("ref", UNSPECIFIED)
            # ref is not defined until Expr.ref property is called by
            # the target printer:
            assert ref is UNSPECIFIED
            if isinstance(ref, str) and ref in self._ref_values:
                assert self._ref_values[ref] is prev
        else:
            if expr.kind == "constant" and type(expr.operands[0]) is not type(prev.operands[0]):
                print(expr, type(expr.operands[0]))
                print(prev, type(prev.operands[0]))
                raise RuntimeError("attempt to re-register equivalent expression")
        return prev

    def _register_reference(self, expr, ref_name):
        # To-be used only from expr.make_ref.
        #
        # Expressions can have reference names that target printers
        # may use to assign expression to a variable that name is the
        # specified reference name, this variable will be used in
        # other expressions as a replacement of the given expression.
        assert isinstance(ref_name, str)
        other = self._ref_values.get(ref_name)
        if other is None:
            # a new reference
            pass
        elif other is expr:
            # reference name is already registered
            assert expr.props["ref"] == ref_name  # sanity check
            return ref_name
        else:
            # reference name is already used.  Make reference name
            # unique, first trying to use the origin as a prefix, and
            # then by adding a counter as a suffix:
            ref_name = expr.props["origin"] + ref_name
            other = self._ref_values.get(ref_name)
            if other is expr:
                assert expr.props["ref"] == ref_name  # sanity check
                return expr
            counter = 0
            # cannot use double-underscore (`__`) because such names
            # appear reserved in stablehlo or llvm, see
            # functional_algorithms#68
            ref_name_ = f"_{ref_name}_{counter}_"
            while other is not None:
                other = self._ref_values.get(ref_name_)
                if other is expr:
                    assert expr.props["ref"] == ref_name_  # sanity check
                    return expr
                elif other is not None:
                    counter += 1
                    ref_name_ = f"_{ref_name}_{counter}_"
            ref_name = ref_name_

        # register reference name:
        self._ref_values[ref_name] = expr
        expr.props.update(ref=ref_name)
        return ref_name

    def __call__(self, expr):
        """Post-process tracing of a function that defines a functional
        algorithm.

        The post-processing involves:
        - use variable names as references to new expressions

        It is assumed that the Context call method is called from a
        return statement of the algorithm definition.
        """
        frame = sys._getframe(1)
        for name, obj in frame.f_locals.items():
            if isinstance(obj, Expr) and obj.props["origin"] == self._stack_name:
                ref_name = obj.props.get("reference_name", UNSPECIFIED)
                if ref_name is UNSPECIFIED:
                    obj.reference(ref_name=name, force=obj.props.get("force", None))
                    if self.alt is not None and obj.kind == "constant" and isinstance(obj.operands[0], Expr):
                        obj.operands[0].reference(ref_name=name + "_", force=obj.props.get("force", None))
        return expr

    def trace(self, func, *args):
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
            a = self.symbol(param.name, typ).reference(ref_name=param.name)
            new_args.append(a)
        args = tuple(new_args)
        name = self.symbol(func.__name__).reference(ref_name=func.__name__)
        return make_apply(self, name, args, func(self, *args))

    def __rewrite_modifier__(self, expr):
        """Used to rewrite expressions based on context parameters."""
        ctx = expr.context
        if ctx.parameters.get(f"use_upcast_{expr.kind}", False):
            ctx.parameters["using"].add(f"upcast {expr.kind}")
            operands = tuple([ctx.upcast(operand) for operand in expr.operands])
            return self.downcast(Expr(ctx, expr.kind, operands))
        if ctx.parameters.get(f"use_downcast_{expr.kind}", False):
            ctx.parameters["using"].add(f"downcast {expr.kind}")
            operands = tuple([ctx.downcast(operand) for operand in expr.operands])
            return self.upcast(Expr(ctx, expr.kind, operands))
        return expr

    def symbol(self, name, typ=UNSPECIFIED):
        if typ is UNSPECIFIED:
            like = self.default_like
            if like is not None:
                typ = like.operands[1]
            else:
                typ = "float"
        return make_symbol(self, name, typ)

    def constant(self, value, like_expr=UNSPECIFIED):
        if like_expr is UNSPECIFIED:
            if isinstance(value, boolean_types):
                like_expr = self.symbol("_boolean_value", "boolean")
            elif self._default_constant_type is not None:
                like_expr = self.symbol("_value", self._default_constant_type)
            elif isinstance(value, float_types):
                like_expr = self.symbol("_float_value", type(value))
            elif isinstance(value, complex_types):
                like_expr = self.symbol("_complex_value", type(value))
            else:
                like_expr = self.default_like
        return make_constant(self, value, like_expr)

    def call(self, func, args):
        """Apply callable to arguments and return its result.

        Callable may apply Context.__call__ to its return value to
        update references. For that, an unique stack name is generated
        that callable may use for prefixing reference names.
        """
        save_stack_name = self._stack_name
        self._stack_call_count[func.__name__] += 1
        self._stack_name = f"_{func.__name__}_{self._stack_call_count[func.__name__]}_"
        try:
            result = func(self, *args)
        finally:
            self._stack_name = save_stack_name
        return result

    def apply(self, name, args, result):
        return make_apply(self, name, args, result)

    # for element-wise functions, use Python array API standard and
    # Python operator for naming conventions, logical operator names
    # use also CamelCase

    def absolute(self, x):
        return Expr(self, "absolute", (x,))

    abs = absolute

    def negative(self, x):
        return Expr(self, "negative", (x,))

    neg = negative

    def positive(self, x):
        return Expr(self, "positive", (x,))

    pos = positive

    def add(self, x, y):
        return Expr(self, "add", (x, y))

    def subtract(self, x, y):
        return Expr(self, "subtract", (x, y))

    sub = subtract

    def multiply(self, x, y):
        return Expr(self, "multiply", (x, y))

    mul = multiply

    def divide(self, x, y):
        return Expr(self, "divide", (x, y))

    div = divide

    def remainder(self, x, y):
        return Expr(self, "remainder", (x, y))

    def floor_divide(self, x, y):
        return Expr(self, "floor_divide", (x, y))

    def pow(self, x, y):
        if isinstance(y, float) and y == 0.5:
            return self.sqrt(x)
        if isinstance(y, int) and y == 2:
            return self.square(x)
        return Expr(self, "pow", (x, y))

    def logical_and(self, x, y):
        return Expr(self, "logical_and", (x, y))

    And = logical_and

    def logical_or(self, x, y):
        return Expr(self, "logical_or", (x, y))

    Or = logical_or

    def logical_xor(self, x, y):
        return Expr(self, "logical_xor", (x, y))

    Xor = logical_xor

    def logical_not(self, x):
        return Expr(self, "logical_not", (x,))

    Not = logical_not

    def bitwise_invert(self):
        return Expr(self, "bitwise_invert", (self,))

    invert = bitwise_invert

    def bitwise_and(self, x, y):
        return Expr(self, "bitwise_and", (x, y))

    def bitwise_or(self, x, y):
        return Expr(self, "bitwise_or", (x, y))

    def bitwise_xor(self, x, y):
        return Expr(self, "bitwise_xor", (x, y))

    def bitwise_left_shift(self, x, y):
        return Expr(self, "bitwise_left_shift", (x, y))

    def bitwise_right_shift(self, x, y):
        return Expr(self, "bitwise_right_shift", (x, y))

    def lt(self, x, y):
        return Expr(self, "lt", (x, y))

    less = lt

    def le(self, x, y):
        return Expr(self, "le", (x, y))

    less_equal = le

    def gt(self, x, y):
        return Expr(self, "gt", (x, y))

    greater = gt

    def ge(self, x, y):
        return Expr(self, "ge", (x, y))

    greater_equal = ge

    def eq(self, x, y):
        return Expr(self, "eq", (x, y))

    equal = eq

    def ne(self, x, y):
        return Expr(self, "ne", (x, y))

    not_equal = ne

    def maximum(self, x, y):
        return Expr(self, "maximum", (x, y))

    max = maximum

    def minimum(self, x, y):
        return Expr(self, "minimum", (x, y))

    min = minimum

    def asin_acos_kernel(self, x):
        return Expr(self, "asin_acos_kernel", (x,))

    def acos(self, x):
        return Expr(self, "acos", (x,))

    def acosh(self, x):
        return Expr(self, "acosh", (x,))

    def asin(self, x):
        return Expr(self, "asin", (x,))

    def asinh(self, x):
        return Expr(self, "asinh", (x,))

    def atan(self, x):
        return Expr(self, "atan", (x,))

    def atanh(self, x):
        return Expr(self, "atanh", (x,))

    def atan2(self, x, y):
        return Expr(self, "atan2", (x, y))

    def cos(self, x):
        return Expr(self, "cos", (x,))

    def cosh(self, x):
        return Expr(self, "cosh", (x,))

    def sin(self, x):
        return Expr(self, "sin", (x,))

    def sinh(self, x):
        return Expr(self, "sinh", (x,))

    def tan(self, x):
        return Expr(self, "tan", (x,))

    def tanh(self, x):
        return Expr(self, "tanh", (x,))

    def exp(self, x):
        return Expr(self, "exp", (x,))

    def expm1(self, x):
        return Expr(self, "expm1", (x,))

    def log(self, x):
        return Expr(self, "log", (x,))

    def log1p(self, x):
        return Expr(self, "log1p", (x,))

    def log2(self, x):
        return Expr(self, "log2", (x,))

    def log10(self, x):
        return Expr(self, "log10", (x,))

    def ceil(self, x):
        return Expr(self, "ceil", (x,))

    def floor(self, x):
        return Expr(self, "floor", (x,))

    def copysign(self, x, y):
        return Expr(self, "copysign", (x, y))

    def round(self, x):
        return Expr(self, "round", (x,))

    def sign(self, x):
        return Expr(self, "sign", (x,))

    def truncate(self, x):
        return Expr(self, "truncate", (x,))

    def conjugate(self, x):
        return Expr(self, "conjugate", (x,))

    conj = conjugate

    def real(self, x):
        return Expr(self, "real", (x,))

    def imag(self, x):
        return Expr(self, "imag", (x,))

    def complex(self, x, y):
        return Expr(self, "complex", (x, y))

    def hypot(self, x, y):
        return Expr(self, "hypot", (x, y))

    def square(self, x):
        return Expr(self, "square", (x,))

    def sqrt(self, x):
        return Expr(self, "sqrt", (x,))

    def select(self, cond, x, y):
        return Expr(self, "select", (cond, x, y))

    def upcast(self, x):
        return Expr(self, "upcast", (x,))

    def downcast(self, x):
        return Expr(self, "downcast", (x,))

    def is_finite(self, x):
        return Expr(self, "is_finite", (x,))
