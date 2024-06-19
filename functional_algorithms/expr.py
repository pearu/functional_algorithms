import warnings
from .utils import UNSPECIFIED
from . import algorithms
from .typesystem import Type
from .rewrite import rewrite


def normalize_like(expr):
    while True:
        if expr.kind in {"constant", "select"}:
            expr = expr.operands[1]
        elif expr.kind in {
            "negative",
            "positive",
            "add",
            "subtract",
            "multiply",
            "divide",
            "maximum",
            "minimum",
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atan2",
            "atanh",
            "cos",
            "cosh",
            "sin",
            "sinh",
            "tan",
            "tanh",
            "exp",
            "expm1",
            "log",
            "log1p",
            "log2",
            "log10",
            "conj",
            "hypot",
            "sqrt",
            "square",
        }:
            expr = expr.operands[0]
        elif expr.kind == "abs" and not expr.operands[0].is_complex:
            expr = expr.operands[0]
        elif expr.kind == "real" and expr.operands[0].kind == "complex":
            expr = expr.operands[0].operands[0]
        elif expr.kind == "imag" and expr.operands[0].kind == "complex":
            expr = expr.operands[0].operands[1]
        else:
            break
    return expr


def make_constant(context, value, like_expr):
    if not isinstance(like_expr, Expr):
        raise TypeError(f"Constant like expression must be Expr instance, got {type(like_expr)}")
    # The type of value is defined by the type of like expression
    # which some targets use implicitly to define the constant type.
    if isinstance(value, str):
        value = {"+inf": "posinf", "inf": "posinf", "pinf": "posinf", "-inf": "neginf", "ninf": "neginf"}.get(value, value)
    return Expr(context, "constant", (value, normalize_like(like_expr)))


def make_symbol(context, name, typ, _tmp_counter=[0]):
    if name is None:
        name = f"_tmp{_tmp_counter[0]}"
        _tmp_counter[0] += 1
    # All symbols must have a type.
    typ = Type.fromobject(context, typ)
    return Expr(context, "symbol", (name, typ))


def make_apply(context, name, args, result):
    return Expr(context, "apply", (name, *args, result))


def normalize(context, operands):
    """Convert numbers to constant expressions"""
    exprs = [operand for operand in operands if isinstance(operand, Expr)]
    if len(exprs) == 0:
        ref_operand = context.default_like
        if ref_operand is None:
            raise ValueError("cannot normalize operands with no reference operand or context default constant type")
    else:
        ref_operand = exprs[0]

    new_operands = []
    for operand in operands:
        if isinstance(operand, (int, float, complex, str)):
            operand = make_constant(context, operand, ref_operand)
        new_operands.append(operand)
    return tuple(new_operands)


def make_ref(expr):
    ref = expr.props.get("ref", UNSPECIFIED)
    if ref is not UNSPECIFIED:
        return ref
    if expr.kind == "symbol":
        return f"{expr.kind}_{expr.operands[0]}"
    if expr.kind == "constant":
        if isinstance(expr.operands[0], Expr):
            return f"{expr.kind}_{make_ref(expr.operands[0])}"
        return f"{expr.kind}_{expr.operands[0]}"
    lst = [expr.kind] + list(map(make_ref, expr.operands))
    return "_".join(lst)


class Printer:

    def tostring(self, expr, tab=""):
        if not isinstance(expr, Expr):
            return str(expr)

        if expr.kind == "symbol":
            return f"{tab}{expr.operands[0]}"
        if expr.kind == "apply":
            name = expr.operands[0]
            args = expr.operands[1:-1]
            body = expr.operands[-1]

            sname = self.tostring(name)
            lst = []
            for a in args:
                lst.append(f"{a.operands[0]}: {a.operands[1]}")
            sargs = ", ".join(lst)

            lines = []
            lines.append(f"{tab}(def {sname}, ({sargs}),")
            for line in self.tostring(body, tab=tab + "  ").splitlines():
                sline = line.lstrip()
                if sline.startswith(")"):
                    lines[-1] += sline
                else:
                    lines.append(line)
            lines[-1] += ")"
            return "\n".join(lines)

        all_symbols = not [operand for operand in expr.operands if isinstance(operand, Expr) and operand.kind != "symbol"]
        if all_symbols:
            return f'{tab}({expr.kind} {", ".join(map(self.tostring, expr.operands))})'

        lines = []
        if expr.kind == "constant":
            lines.append(f"{tab}({expr.kind} {expr.operands[0]},")
            operands = expr.operands[1:]
        else:
            lines.append(f"{tab}({expr.kind}")
            operands = expr.operands
        operand_lines = []
        for operand in operands:
            op_lines = self.tostring(operand, tab=tab + "  ").splitlines()
            operand_lines.extend(op_lines)
            operand_lines[-1] += ","
        if operand_lines:
            operand_lines[-1] = operand_lines[-1][:-1]  # deletes last comma
        if len(operand_lines) == 1:
            lines[-1] += " " + operand_lines[0].lstrip() + ")"
        else:
            lines.extend(operand_lines)
            lines.append(f"{tab})")
        return "\n".join(lines)


class Expr:

    def __new__(cls, context, kind, operands):
        obj = object.__new__(cls)

        if kind == "symbol":
            assert len(operands) == 2 and isinstance(operands[0], str) and isinstance(operands[1], Type), operands
        elif kind == "constant":
            assert len(operands) == 2
            assert isinstance(operands[0], (int, float, bool, complex, str, Expr))
            if isinstance(operands[0], Expr):
                assert operands[0].context is context.alt, operands
            if context.alt is not None and not isinstance(operands[0], Expr):
                operands = [context.alt.constant(operands[0]), operands[1]]
        else:
            if kind == "select":
                operands = operands[:1] + normalize(context, operands[1:])
            else:
                operands = normalize(context, operands)

            if context.alt is not None:
                constant_operands = []
                constant_type = None
                constant_like = None
                for o in operands:
                    if isinstance(o, Expr) and o.kind == "constant":
                        constant_operands.append(o.operands[0])
                        if constant_like is None:
                            constant_like = o.operands[1]
                            constant_type = constant_like.get_type()
                        else:
                            t = o.operands[1].get_type()
                            if not constant_type.is_same(t):
                                warnings.warn(f"{kind}: unexpected operand type `{t}`, expected `{constant_type}`")
                if len(constant_operands) == len(operands) and constant_like is not None:
                    operands = (Expr(context.alt, kind, tuple(constant_operands)), constant_like)
                    kind = "constant"

            assert False not in [isinstance(operand, Expr) for operand in operands], operands

        obj.context = context
        obj.kind = kind
        obj.operands = tuple(operands)
        # expressions are singletons within the given context and are
        # uniquely identified by its serialized string value. However,
        # expression props are mutable. Therefore. mutations (e.g. ref
        # updates) have global effect within the given context.
        obj._serialized = obj._serialize()

        # props is a dictionary that contains ref, force_ref,
        # other. In general, its content could be anything.

        # When ref is specified in props, it is used as a variable
        # name referencing the expression. When an expression with
        # given ref is used as an operand in some other expression,
        # then printing the expression will replace the operand with
        # its reference value and `<ref> = <expr>` will be added to
        # assignments list (see targets.<target>.Printer).
        #
        # If ref is UNSPECIFIED, context._update_refs will assign ref
        # value to the variable name in locals() that object is
        # identical to the expression object.
        #
        # If ref is None, no assignment of ref value will be
        # attempted.
        obj.props = dict()
        return context._register_expression(obj)

    def _serialize(self):
        if self.kind == "symbol":
            return f"{self.operands[0]}:{self.operands[1]}"
        if self.kind == "constant":
            return f"{self.operands[0]}:type({self.operands[1]._serialized})"
        return f'{self.kind}({",".join(operand._serialized for operand in self.operands)})'

    def _replace(self, other):
        if other is self:
            # nothing to replace
            return self
        # replace self by other
        ref = self.props.get("ref", UNSPECIFIED)
        force_ref = self.props.get("force_ref", None)
        if ref in self.context._ref_values:
            self.context._ref_values[ref] = other
        if isinstance(ref, str):
            other.props.update(ref=ref)
        if force_ref is not None:
            other.props.update(force_ref=force_ref)
        # self cannot be reference anymore:
        self.props.update(ref=None)
        # but self can track to other reference if needed:
        self.props.update(other=other)
        return other

    def implement_missing(self, target):
        if self.kind == "symbol":
            return self
        elif self.kind == "constant":
            like = self.operands[1].implement_missing(target)
            value = self.operands[0]
            if isinstance(value, Expr):
                value = value.implement_missing(target.constant_target)
            if value is self.operands[0] and like is self.operands[1]:
                return self
            result = make_constant(self.context, value, like)
        elif self.kind == "apply":
            body = self.operands[-1].implement_missing(target)
            if body is self.operands[-1]:
                return self
            result = make_apply(self.context, self.operands[0], self.operands[1:-1], body)
        elif target.kind_to_target.get(self.kind, NotImplemented) is NotImplemented:
            func = NotImplemented
            for m in self.context._paths:
                func = getattr(m, self.kind, NotImplemented)
                if func is not NotImplemented:
                    break
            if func is NotImplemented:
                paths = ":".join([m.__name__ for m in self.context._paths])
                raise NotImplementedError(f'{self.kind} for {target.__name__.split(".")[-1]} target [paths={paths}]')

            result = self.context.call(func, self.operands).implement_missing(target)
        else:
            operands = tuple([operand.implement_missing(target) for operand in self.operands])
            for o1, o2 in zip(operands, self.operands):
                if o1 is not o2:
                    break
            else:
                return self
            result = Expr(self.context, self.kind, operands)
        return self._replace(result)

    def simplify(self):
        if self.kind in {"symbol", "constant"}:
            return self

        # result = rewrite(self)

        # if result is not None:
        #    return self._replace(result.simplify())
        result = self

        if self.kind == "apply":
            body = self.operands[-1].simplify()
            if body is not self.operands[-1]:
                result = make_apply(self.context, self.operands[0], self.operands[1:-1], body)
        else:
            operands = tuple([operand.simplify() for operand in self.operands])

            for o1, o2 in zip(operands, self.operands):
                if o1 is not o2:
                    result = Expr(self.context, self.kind, operands)
                    break

        result_ = rewrite(result)
        if result_ is not None:
            result = result_  # call simplify?

        return self._replace(result)

    def tostring(self, target, tab="", need_ref=None, debug=0):
        if need_ref is None:

            def compute_need_ref(expr: Expr, need_ref: dict) -> None:

                ref = expr.ref

                if ref is None and "other" in expr.props:
                    compute_need_ref(expr.props["other"], need_ref)
                elif ref in need_ref:
                    # expression with ref is used more than once, hence we'll
                    # mark it as needed
                    need_ref[ref] = True
                else:
                    assert ref is not None
                    # the first usage of expression with ref does not require
                    # using ref, unless forced.
                    need_ref[ref] = expr.props.get("force_ref", False)

                    for operand in expr.operands:
                        if isinstance(operand, Expr):
                            compute_need_ref(operand, need_ref)

            need_ref = dict()
            compute_need_ref(self, need_ref)

        return target.Printer(need_ref, debug=debug).tostring(self, tab=tab)

    @property
    def ref(self):
        """Return existing reference name or generate a new one.

        Used by target printers for expressions that need referencing.
        """
        ref = self.props.get("ref", UNSPECIFIED)
        if ref is None and "other" in self.props:
            return self.props["other"].ref  # + '__other'
        if ref in {None, UNSPECIFIED}:
            ref = make_ref(self)
            # assert ref is not None, self.props
            assert ref not in self.context._ref_values, ref
        assert ref is not UNSPECIFIED
        return ref

    def reference(self, ref_name=UNSPECIFIED, force=True):
        """Manage referencing an expression. Returns self.

        Referencing an expression means that in target printer, an
        expression value is saved in a variable and its name is used
        in other expressions to reference the expression value.

        By default, expressions that are used in other expressions
        more than once, are always referenced. Expressions that are
        used only once, are inlined when force=False. When force=True,
        such expressions will be referenced. This can be useful when
        debugging (to print out the values of subexpressions) or for
        improving the readability of the target printer output.

        When an expression is referenced, it must have a reference
        name. The reference name can either be user-specified
        explicitly (using ref_name argument in `.reference(...)`),
        implicitly (assigning expression to a variable and using
        Context.__call__ in function return statement), or
        auto-generated (calling `.ref` property that uses `make_ref`
        function).

        In all cases, one must be careful for not using the same
        reference name for different expressions. To detect reference
        name conflicts, Context instance holds a mapping
        `._ref_values` that stores pairs `(<ref_name>, <expression>)`
        that is global within the given context.
        """
        if force is not None:
            self.props.update(force_ref=force)
        if ref_name is UNSPECIFIED:
            return self
        if ref_name in self.context._ref_values:
            other = self.context._ref_values[ref_name]
            if other is self:
                assert self.props["ref"] == ref_name, (self.props["ref"], ref_name)
            else:
                if ref_name.startswith(self.context._stack_name):
                    raise RuntimeError(f"reference name {ref_name} is already taken")
                ref_name = self.context._stack_name + ref_name
                assert ref_name not in self.context._ref_values
                self.props.update(ref=ref_name)
                self.context._ref_values[ref_name] = self
        else:
            self.props.update(ref=ref_name)
            self.context._ref_values[ref_name] = self
        return self

    def __str__(self):
        return Printer().tostring(self)

    def __repr__(self):
        return f"{type(self).__name__}({self.kind}, {self.operands}, {self.props})"

    def __abs__(self):
        return self.context.abs(self)

    def __neg__(self):
        return self.context.negative(self)

    def __pos__(self):
        return self.context.pos(self)

    def __invert__(self):
        return self.context.invert(self)

    def __add__(self, other):
        return self.context.add(self, other)

    def __radd__(self, other):
        return self.context.add(other, self)

    def __sub__(self, other):
        return self.context.subtract(self, other)

    def __rsub__(self, other):
        return self.context.subtract(other, self)

    def __mul__(self, other):
        return self.context.multiply(self, other)

    def __rmul__(self, other):
        return self.context.multiply(other, self)

    def __truediv__(self, other):
        return self.context.divide(self, other)

    def __rtruediv__(self, other):
        return self.context.divide(other, self)

    def __floordiv__(self, other):
        return self.context.floor_divide(self, other)

    def __rfloordiv__(self, other):
        return self.context.floor_divide(other, self)

    def __pow__(self, other):
        return self.context.pow(self, other)

    def __rpow__(self, other):
        return self.context.pow(other, self)

    def __mod__(self, other):
        return self.context.reminder(self, other)

    def __rmod__(self, other):
        return self.context.reminder(other, self)

    def __and__(self, other):
        return self.context.bitwise_and(self, other)

    def __rand__(self, other):
        return self.context.bitwise_and(other, self)

    def __or__(self, other):
        return self.context.bitwise_or(self, other)

    def __ror__(self, other):
        return self.context.bitwise_or(other, self)

    def __xor__(self, other):
        return self.context.bitwise_xor(self, other)

    def __rxor__(self, other):
        return self.context.bitwise_xor(other, self)

    def __lshift__(self, other):
        return self.context.left_shift(self, other)

    def __rlshift__(self, other):
        return self.context.left_shift(other, self, ref=None)

    def __rshift__(self, other):
        return self.context.right_shift(self, other)

    def __rrshift__(self, other):
        return self.context.right_shift(other, self, ref=None)

    def __round__(self, ndigits=None):
        return self.context.round(self, ndigits)

    def __trunc__(self):
        return self.context.trunc(self)

    def __floor__(self):
        return self.context.floor(self)

    def __ceil__(self):
        return self.context.ceil(self)

    @property
    def real(self):
        return self.context.real(self)

    @property
    def imag(self):
        return self.context.imag(self)

    def __lt__(self, other):
        return self.context.lt(self, other)

    def __le__(self, other):
        return self.context.le(self, other)

    def __gt__(self, other):
        return self.context.gt(self, other)

    def __ge__(self, other):
        return self.context.ge(self, other)

    def __eq__(self, other):
        return self.context.eq(self, other)  # TODO: why it is not effective?

    def __ne__(self, other):
        return self.context.ne(self, other)

    @property
    def is_posinf(self):
        return self == self.context.constant("posinf", self)

    @property
    def is_neginf(self):
        return self == self.context.constant("neginf", self)

    @property
    def is_complex(self):
        if self.kind in {"symbol", "constant", "select"}:
            return self.operands[1].is_complex
        elif self.kind in {
            "lt",
            "le",
            "gt",
            "ge",
            "eq",
            "ne",
            "real",
            "imag",
            "abs",
            "logical_and",
            "logical_or",
            "logical_xor",
            "logical_not",
            "bitwise_invert",
            "bitwise_and",
            "bitwise_or",
            "bitwise_xor",
            "bitwise_left_shift",
            "bitwise_right_shift",
            "ceil",
            "floor",
            "hypot",
            "maximum",
            "minimum",
            "floor_divide",
            "remainder",
        }:
            return False
        elif self.kind in {"complex", "conj"}:
            return True
        elif self.kind in {"add", "subtract", "divide", "multiply", "pow"}:
            return self.operands[0].is_complex or self.operands[1].is_complex
        elif self.kind in {
            "positive",
            "negative",
            "sqrt",
            "square",
            "asin",
            "acos",
            "atan",
            "asinh",
            "acosh",
            "atanh",
            "sinh",
            "cosh",
            "tanh",
            "sin",
            "cos",
            "tan",
            "log",
            "log1p",
            "log2",
            "log10",
            "exp",
            "expm1",
        }:
            return self.operands[0].is_complex
        elif self.kind == "apply":
            return self.operands[-1].is_complex
        else:
            raise NotImplementedError(f"Expr.is_complex for {self.kind}")

    def get_type(self):
        if self.kind == "symbol":
            return self.operands[1]
        if self.kind == "constant":
            return self.operands[1].get_type()
        if self.kind in {"lt", "le", "gt", "ge", "eq", "ne", "logical_and", "logical_or", "logical_xor"}:
            return Type.fromobject(self.context, "boolean")
        elif self.kind in {
            "positive",
            "negative",
            "sqrt",
            "square",
            "asin",
            "acos",
            "atan",
            "asinh",
            "acosh",
            "atanh",
            "sinh",
            "cosh",
            "tanh",
            "sin",
            "cos",
            "tan",
            "log",
            "log1p",
            "log2",
            "log10",
            "exp",
            "expm1",
            "ceil",
            "floor",
            "logical_not",
            "sign",
        }:
            return self.operands[0].get_type()
        elif self.kind in {
            "add",
            "subtract",
            "divide",
            "multiply",
            "pow",
            "maximum",
            "minimum",
            "hypot",
            "remainder",
            "atan2",
        }:
            return self.operands[0].get_type().max(self.operands[1].get_type())
        elif self.kind in {"abs", "real", "imag"}:
            t = self.operands[0].get_type()
            return t.complex_part if t.is_complex else t
        elif self.kind == "select":
            return self.operands[1].get_type().max(self.operands[2].get_type())
        elif self.kind == "complex":
            t = self.operands[0].get_type().max(self.operands[1].get_type())
            bits = t.bits * 2 if t.bits is not None else None
            return Type(self.context, "complex", bits)
        raise NotImplementedError((self.kind, str(self)))
