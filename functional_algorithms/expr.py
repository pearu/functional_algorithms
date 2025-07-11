import numpy
import math
import struct
import warnings
from .utils import UNSPECIFIED, warn_once, value_types, float_types, integer_types, number_types
from .typesystem import Type
from .rewrite import RewriteContext


known_expression_kinds = set(
    """
symbol, constant, apply, select, list, item, len,
negative, positive, add, subtract, multiply, divide,
minimum, maximum,
asin, acos, atan, asinh, acosh, atanh, asin_acos_kernel, atan2,
sin, cos, tan, sinh, cosh, tanh,
log, log1p, log2, log10,
exp, expm1, sqrt, square, pow, exp2,
complex, conjugate, real, imag, absolute, hypot,
lt, gt, le, ge, eq, ne,
logical_and, logical_or, logical_xor, logical_not,
bitwise_invert, bitwise_and, bitwise_or, bitwise_xor, bitwise_left_shift, bitwise_right_shift,
ceil, floor, floor_divide, remainder, round, truncate,
copysign, sign, nextafter,
upcast, downcast,
is_finite, is_inf, is_posinf, is_neginf, is_nan, is_negzero
""".replace(
        " ", ""
    )
    .replace("\n", "")
    .split(",")
)

known_constant_names = set(
    """
eps, posinf, neginf, smallest, largest, smallest_subnormal, pi, undefined, nan
""".replace(
        " ", ""
    )
    .replace("\n", "")
    .split(",")
)


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
            "exp2",
            "expm1",
            "log",
            "log1p",
            "log2",
            "log10",
            "conj",
            "hypot",
            "sqrt",
            "square",
            "asin_acos_kernel",
        }:
            expr = expr.operands[0]
        elif expr.kind == "absolute" and not expr.operands[0].is_complex:
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
        if value not in known_constant_names:
            warn_once(f"creating constant from unknown constant name: {value}")
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


def make_list(context, items):
    return Expr(context, "list", tuple(items))


def make_index(context, index):
    if isinstance(index, integer_types):
        return context.constant(index, type(index))
    elif isinstance(index, Expr):
        # TODO: check if index type is integer
        return index
    else:
        raise TypeError(f"index must be integer or integer expression, got {type(index)}")


def make_len(context, container):
    return Expr(context, "len", (container,))


def make_item(context, container, index):
    return Expr(context, "item", (container, make_index(context, index)))


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


def toidentifier(value):
    if isinstance(value, bool):
        return str(value)
    elif isinstance(value, (int, numpy.integer)):
        if value < 0:
            return "neg" + str(-value)
        return str(value)
    elif isinstance(value, float):
        try:
            intvalue = int(value)
        except OverflowError:
            intvalue = None
        if value == intvalue and intvalue.bit_length() <= 64:
            return "f" + toidentifier(intvalue)
        if math.isinf(value):
            return "posinf" if value > 0 else "neginf"
        return "f" + hex(struct.unpack("<Q", struct.pack("<d", value))[0])[1:]
    elif isinstance(value, complex):
        return "c" + toidentifier(value.real) + toidentifier(value.imag)
    elif isinstance(value, str):
        assert value.isidentifier(), value
        return value
    elif isinstance(value, numpy.floating):
        try:
            intvalue = int(value)
        except OverflowError:
            intvalue = None
        if value == intvalue and intvalue.bit_length() <= value.dtype.itemsize * 8:
            return value.dtype.kind + toidentifier(intvalue)
        if numpy.isposinf(value):
            return "posinf"
        if numpy.isneginf(value):
            return "neginf"
        if numpy.isnan(value):
            return "nan"
        return value.dtype.kind + "0x" + "".join(map(hex, value.tobytes()[::-1])).replace("0x", "")
    elif isinstance(value, numpy.complexfloating):
        return value.dtype.kind + toidentifier(value.real) + toidentifier(value.imag)
    else:
        raise NotImplementedError(type(value))


def make_ref(expr):
    """Return a reference name to an expression. If the expression does
    not have reference name, then generate one.
    """
    ref = expr.props.get("ref", UNSPECIFIED)
    ref_name = expr.props.get("reference_name", None)
    if isinstance(ref, str):
        # existing reference name
        return ref
    elif isinstance(ref_name, str):
        ref = ref_name  # context._register_reference will ensure uniqueness
    elif ref is UNSPECIFIED:
        # generate a reference name
        if expr.kind == "symbol":
            ref = f"{expr.kind}_{expr.operands[0]}"
        elif expr.kind == "constant":
            if isinstance(expr.operands[0], Expr):
                ref = f"{expr.kind}_{make_ref(expr.operands[0])}"
            else:
                ref = f"{expr.kind}_{toidentifier(expr.operands[0])}"
            assert len(ref) < 50, type(expr.operands[0])
        elif expr.kind == "absolute":
            # using abs for BC
            ref = f"abs_{make_ref(expr.operands[0])}"
        else:
            all_operands_have_ref_name = not [
                0 for o in expr.operands if not isinstance(expr.operands[0].props.get("reference_name"), str)
            ]
            if all_operands_have_ref_name:
                # for readability
                lst = [expr.kind] + list(map(make_ref, expr.operands))
                ref = "_".join(lst)
            else:
                ref = f"{expr.kind}_{expr.intkey}"
    elif ref is None:
        # referencing the expression has been disabled
        assert 0  # unreachable
    else:
        assert 0  # unreachable

    # When orig_ref_name is None, referencing the expression has been
    # disabled. The expression reference name is generated anyway
    # because it is used as a part of a parent expression, however,
    # we'll skip registering such names.
    if ref_name is None:
        return ref

    return expr.context._register_reference(expr, ref)


def _cache_in_props(mth):

    def wrap(self):
        name = mth.__name__
        r = self.props.get(name, UNSPECIFIED)
        if r is UNSPECIFIED:
            self.props[name] = r = mth(self)
        return r

    return wrap


class Printer:

    def tostring(self, expr, tab=""):
        # TODO: use SSA-like output for large expression trees with
        # many overlapping branches
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
                if a.kind == "symbol":
                    lst.append(f"{a.operands[0]}: {a.operands[1]}")
                elif a.kind == "list":
                    at = ", ".join([self.tostring(a_.operands[1]) for a_ in a.operands])
                    lst.append(f"{a.ref}: list[{at}]")
                else:
                    # argument must be a symbol or a list of symbols
                    assert 0, a.kind  # unreachable

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
        if kind not in known_expression_kinds:
            warnings.warn(f"Constructing an expression with unknown kind: {kind}", stacklevel=2)

        obj = object.__new__(cls)

        if kind == "symbol":
            assert len(operands) == 2 and isinstance(operands[0], str) and isinstance(operands[1], Type), operands
        elif kind == "constant":
            assert len(operands) == 2
            assert isinstance(operands[0], value_types + (str, Expr)), type(operands[0])
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

            assert False not in [isinstance(operand, Expr) for operand in operands], [type(o) for o in operands]

            if kind == "item":
                assert len(operands) == 2
                assert operands[0].kind == "list"
                assert operands[1].get_type().is_integer

        obj.context = context
        obj.kind = kind
        obj.operands = tuple(operands)
        # expressions are singletons within the given context and are
        # uniquely identified by its serialized string value. However,
        # expression props are mutable. Therefore. mutations (e.g. ref
        # updates) have global effect within the given context.
        obj._compute_serialized()

        # __serialize_id is an unique int id only within the given
        # context. __serialize_id will be initialized in
        # context._register_expression
        obj._set_serialized_id(None)

        # props is a dictionary that contains reference_name and
        # force_ref.  In general, its content could be anything. For
        # instance, the results of `_is_foo` method calls are cached
        # in props.

        # When ref is specified in props, it is used as a variable
        # name referencing the expression. When an expression with
        # given ref is used as an operand in some other expression,
        # then printing the expression will replace the operand with
        # its reference value and `<ref> = <expr>` will be added to
        # assignments list. See alse targets.<target>.Printer and a
        # doc-string of the `reference` method.
        #
        # If ref is UNSPECIFIED, context._update_refs will assign ref
        # value to the variable name in locals() that object is
        # identical to the expression object.
        #
        # If ref is None, no assignment of ref value will be
        # attempted.
        obj.props = dict()

        return context._register_expression(obj)

    def _compute_serialized(self):
        if self.kind == "symbol":
            r = (self.kind, *self.operands)
        elif self.kind == "constant":
            value, like = self.operands
            r = (
                "z_" + self.kind,  # prefix `z_` ensures that constants are sorted as largest kinds
                value.key if isinstance(value, Expr) else (value, type(value).__name__),
                like.key,
            )
        else:
            # We don't use `operand.key` as its size is unbounded and
            # will lead to large overhead in computing the hash value
            # of the expression. However, since intkey is unique only
            # in the given runtime, the key will have the same
            # property of intkey being unique only within the given
            # runtime and context.
            r = (self.kind, *(operand._two_level_intkey for operand in self.operands))
        self.__serialized = r

    def _set_serialized_id(self, i):
        self.__serialize_id = i

    def implement_missing(self, target):
        warn_once("Calling `Expr.implement_missing(target)` is deprecated. Use `Expr.rewrite(target)` instead.", stacklevel=2)
        return self.rewrite(target)

    def simplify(self):
        warn_once("Calling `Expr.simplify()` is deprecated. Use `Expr.rewrite(rewrite)` instead.", stacklevel=2)
        from . import rewrite

        return self.rewrite(rewrite)

    def rewrite(self, modifier, *modifiers, deep_first=None, _rewrite_context=None):
        """Rewrite expression using modifier callable.

        Parameters
        ----------
        modifier: {callable, object}
          A callable that either returns a new expression or its
          input. If the object has `__rewrite_modifier__` attribute,
          then it will be used as the modifier callable.
        modifiers: tuple
          Extra modifiers to be applied to the result.
        deep_first: bool
          If True, the modifier is first applied to expression
          operands and then to the expression itself.
          If False, the modifier is first applied to expression and
          only when the result is the same expression, the modifier is
          applied to expression operands.

        """
        if modifiers:
            result = self.rewrite(modifier, deep_first=deep_first)
            for modifier_ in modifiers:
                result = result.rewrite(modifier_, deep_first=deep_first)
            return result

        rewrite_context = RewriteContext() if _rewrite_context is None else _rewrite_context

        if self in rewrite_context:
            return rewrite_context(self)

        if hasattr(modifier, "__rewrite_modifier__"):
            modifier = modifier.__rewrite_modifier__

        if deep_first is None:
            deep_first = True

        rewrite_kwargs = dict(deep_first=deep_first, _rewrite_context=rewrite_context)

        result = self if deep_first else modifier(self)

        if result is not self:
            pass
        elif self.kind == "symbol":
            result = self
        elif self.kind == "constant":
            like = self.operands[1].rewrite(modifier, **rewrite_kwargs)
            value = self.operands[0]
            if isinstance(value, Expr):
                # alternative context uses its own rewrite context:
                value = value.rewrite(modifier, deep_first=deep_first)
            if value is self.operands[0] and like is self.operands[1]:
                result = self
            else:
                result = make_constant(self.context, value, like)
        elif self.kind == "apply":
            body = self.operands[-1].rewrite(modifier, **rewrite_kwargs)
            if body is not self.operands[-1]:
                result = make_apply(self.context, self.operands[0], self.operands[1:-1], body)
            else:
                result = self
        else:
            operands = tuple([operand.rewrite(modifier, **rewrite_kwargs) for operand in self.operands])
            for o1, o2 in zip(operands, self.operands):
                if o1 is not o2:
                    result = Expr(self.context, self.kind, operands)
                    break
            else:
                result = self

        if deep_first:
            result = modifier(result)

        return rewrite_context(self, result)

    def tostring(self, target, tab="", need_ref=None, debug=0, **printer_parameters):
        if need_ref is None:

            def compute_need_ref(expr: Expr, need_ref: dict) -> None:

                ref = expr.ref
                assert ref is not None  # sanity check

                if ref in need_ref:
                    # expression with ref is used more than once, hence we'll
                    # mark it as needed
                    need_ref[ref] = True
                else:
                    # the first usage of expression with ref does not require
                    # using ref, unless forced.
                    need_ref[ref] = expr.props.get("force_ref", False)

                    for operand in expr.operands:
                        if isinstance(operand, Expr):
                            compute_need_ref(operand, need_ref)

            need_ref = dict()
            compute_need_ref(self, need_ref)

        return target.Printer(need_ref, debug=debug, **printer_parameters).tostring(self, tab=tab)

    @property
    def key(self):
        """Return a string key unique to this expression instance and that can
        be used as a dictionary key.

        Note: the value of key is equivalent to the high-level
        structure of the expression but its finer details are
        equivalent to operands intkey values.

        Warning: the key is unique only within the given run-time.

        Warning: the key is unique only within the given context.
        """
        # Sometimes, returning id(self) would be also a valid key, but
        # in algorithms (e.g. op_collect with commutative=True) that
        # use keys sorting, this would introduce non-deterministic
        # results.
        return self.__serialized

    @property
    def intkey(self):
        """Return a integer valued key that is unique to this expression
        instance and that can be used as a dictionary key. Different
        from the .key property, the .intkey property should not be
        used in algorithms that relay on sorting of expression kinds.

        Note: the value of intkey is equivalent to the construction
        time of the expression. Hence, similar to id, intkey has no
        relation to the content of the expression but different from
        id, intkey carries a time-stamp of expression construction
        moment.

        Warning: the intkey is unique only within the given run-time.

        Warning: the intkey is unique only within the given context.
        """
        return self.__serialize_id

    @property
    def _two_level_intkey(self):
        # used in computing the key of an expression
        if self.kind in {"symbol", "constant"}:
            return (self.kind, self.intkey)
        return (self.kind, *(op.intkey for op in self.operands))

    @property
    def ref(self):
        """Return existing reference name or generate a new one.

        Used by target printers for expressions that need referencing.
        """
        return make_ref(self)

    def reference(self, ref_name=UNSPECIFIED, force=True):
        """Manage referencing an expression. Returns self.

        Referencing an expression means that in target printer, an
        expression value is saved in a variable and its name is used
        in other expressions to reference the expression value.

        By default, expressions that are used in other expressions
        more than once, are always referenced. Expressions that are
        used only once, are inlined when force=False. When force=True,
        such expressions will be always referenced. This can be useful
        when debugging (to print out the values of subexpressions) or
        for improving the readability of the target printer output.

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
            # What this means? force_ref==True?
            return self
        assert isinstance(ref_name, str), ref_name
        self.props.update(reference_name=ref_name)
        return self

    def __bool__(self):
        # Hint: for Expr object comparison, use `x is y` or `x.key == y.key`, but not `x == y`.
        raise RuntimeError(f"The truth value of an Expr instance `{self}` is undefined")

    def __str__(self):
        return Printer().tostring(self)

    def __repr__(self):
        if self.kind in {"symbol", "constant"}:
            operands = self.operands
        else:
            operands = tuple(f"{o.kind}:{o.intkey}" for o in self.operands)
        # return f"{type(self).__name__}({self.kind}, {operands}, {self.props})"
        return f"{type(self).__name__}({self.kind}, {operands})"

    def __index__(self):
        if self.kind == "constant" and self.get_type().is_integer:
            return self.operands[0]
        raise TypeError(f"cannot convert to Python int")

    def __abs__(self):
        return self.context.absolute(self)

    def __neg__(self):
        return self.context.negative(self)

    def __pos__(self):
        return self.context.pos(self)

    def __invert__(self):
        if self._is_boolean:
            return self.context.logical_not(self)
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
        return self.context.remainder(self, other)

    def __rmod__(self, other):
        return self.context.remainder(other, self)

    def __and__(self, other):
        if self._is_boolean and other._is_boolean:
            return self.context.logical_and(self, other)
        return self.context.bitwise_and(self, other)

    def __rand__(self, other):
        return self.context.bitwise_and(other, self)

    def __or__(self, other):
        if self._is_boolean and other._is_boolean:
            return self.context.logical_or(self, other)
        return self.context.bitwise_or(self, other)

    def __ror__(self, other):
        return self.context.bitwise_or(other, self)

    def __xor__(self, other):
        if self._is_boolean and other._is_boolean:
            return self.context.logical_xor(self, other)
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

    def conj(self):
        return self.context.conjugate(self)

    def __lt__(self, other):
        return self.context.lt(self, other)

    def __le__(self, other):
        return self.context.le(self, other)

    def __gt__(self, other):
        return self.context.gt(self, other)

    def __ge__(self, other):
        return self.context.ge(self, other)

    def __eq__(self, other):
        # Hint: for Expr object comparison, use `x is y` or `x.key == y.key`, but not `x == y`.
        return self.context.eq(self, other)

    def __ne__(self, other):
        return self.context.ne(self, other)

    def __getitem__(self, index):
        return self.context.item(self, index)

    def __len__(self):
        if self.kind == "list":
            return len(self.operands)
        return self.context.len(self)

    @property
    def is_posinf(self):
        return self == self.context.constant("posinf", self)

    @property
    def is_neginf(self):
        return self == self.context.constant("neginf", self)

    @property
    def is_inf(self):
        return self.context.logical_or(self.is_posinf, self.is_neginf)

    def _is(self, *props):
        result = True
        for prop in props:
            r = getattr(self, f"_is_{prop}")
            if r is None:
                return
            result = result and r
            if not result:
                break
        return result

    @property
    @_cache_in_props
    def _is_nonzero(self):
        r = self._is_zero
        if r is not None:
            return not r

    @property
    @_cache_in_props
    def _is_zero(self):
        if self.kind == "constant":
            value, like = self.operands
            if isinstance(value, Expr):
                return value._is_zero
            elif isinstance(value, number_types):
                return bool(value == 0)
            elif isinstance(value, str):
                if value in {"undefined", "nan"}:
                    return
                elif value in known_constant_names:
                    return False
        elif self.kind == "sqrt":
            if self.operands[0]._is_negative:
                return
            return self.operands[0]._is_zero
        elif self.kind in {"square", "absolute"}:
            return self.operands[0]._is_zero
        elif self._is_positive or self._is_negative:
            return False

    @property
    @_cache_in_props
    def _is_one(self):
        if self.kind == "constant":
            value, like = self.operands
            if isinstance(value, Expr):
                return value._is_one
            elif isinstance(value, number_types):
                return bool(value == 1)
            elif isinstance(value, str):
                if value in {"undefined", "nan"}:
                    return
                elif value in known_constant_names:
                    return False
        elif self.kind == "sqrt":
            if self.operands[0]._is_negative:
                return
            return self.operands[0]._is_one
        elif self.kind in {"square", "absolute"}:
            return self.operands[0]._is_one
        elif self._is_nonpositive:
            return False

    @property
    @_cache_in_props
    def _is_finite(self):
        if self.kind == "constant":
            value, like = self.operands
            if isinstance(value, str):
                if value in {"posinf", "neginf"}:
                    return False
                elif value in {"undefined", "nan"}:
                    return
                elif value in known_constant_names:
                    return True
            elif isinstance(value, float_types):
                if isinstance(value, numpy.floating):
                    return numpy.isfinite(value)
                return math.isfinite(value)
            elif isinstance(value, integer_types):
                return True
        elif self.kind == "divide":
            x, y = self.operands
            if y.kind == "constant":
                if isinstance(y.operands[0], Expr):
                    return y._is_nonzero and x._is_finite
                elif y._is_zero:
                    return False
                elif y.operands[0] in {"posinf", "neginf"}:
                    return x._is_finite
            return x._is_finite and y._is_finite
        elif self.kind == "sqrt":
            if self._is_nonnegative:
                return True
        elif self.kind in {"add", "subtract", "positive", "negative", "absolute", "multiply", "square"}:
            for x in self.operands:
                r = x._is_finite
                if r is None:
                    return
                if not r:
                    break
            else:
                return True
            return False

    @property
    @_cache_in_props
    def _is_nonnegative(self):
        assert not self.is_complex
        if self.kind == "constant":
            value, like = self.operands
            if isinstance(value, float_types + integer_types):
                return value >= 0
            elif isinstance(value, str):
                if value in {"neginf"}:
                    return False
                elif value in {"undefined", "nan"}:
                    return
                elif value in known_constant_names:
                    return True
            elif isinstance(value, Expr):
                return value._is_nonnegative
        elif self.kind == "add":
            x, y = self.operands
            if x._is_nonnegative and y._is_nonnegative:
                return True
            if x._is_negative and y._is_negative:
                return False
        elif self.kind == "subtract":
            x, y = self.operands
            if x._is_nonnegative and y._is_nonpositive:
                return True
            if x._is_negative and y._is_positive:
                return False

        elif self.kind in {"multiply", "divide"}:
            x, y = self.operands
            if (
                (x._is_nonnegative and y._is_nonnegative)
                or (x._is_nonpositive and y._is_nonpositive)
                or (x is y and x.kind != "constant")
            ):
                return True
            if (x._is_negative and y._is_positive) or (x._is_positive and y._is_negative):
                return False
        elif self.kind == "negative":
            return self.operands[0]._is_nonpositive
        elif self.kind == "positive":
            return self.operands[0]._is_nonnegative
        elif self.kind == "sqrt" and self.operands[0]._is_nonnegative:
            return True
        elif self.kind in {"absolute", "square"}:
            return True

    @property
    @_cache_in_props
    def _is_nonpositive(self):
        assert not self.is_complex
        if self.kind == "constant":
            value, like = self.operands
            if isinstance(value, float_types + integer_types):
                return value <= 0
            elif isinstance(value, str):
                if value in {"neginf"}:
                    return True
                elif value in {"undefined", "nan"}:
                    return
                elif value in known_constant_names:
                    return False
            elif isinstance(value, Expr):
                return value._is_nonpositive
        elif self.kind == "add":
            x, y = self.operands
            if x._is_nonpositive and y._is_nonpositive:
                return True
            if x._is_positive and y._is_positive:
                return False
        elif self.kind == "subtract":
            x, y = self.operands
            if x._is_nonpositive and y._is_nonnegative:
                return True
            if x._is_positive and y._is_negative:
                return False
        elif self.kind in {"multiply", "divide"}:
            x, y = self.operands
            if (x._is_nonpositive and y._is_positive) or (x._is_nonnegative and y._is_negative):
                return True
            if (x._is_negative and y._is_negative) or (x._is_positive and y._is_positive):
                return False
        elif self.kind == "negative":
            return self.operands[0]._is_nonnegative
        elif self.kind == "positive":
            return self.operands[0]._is_nonpositive
        elif self.kind in {"sqrt", "square", "absolute"} and self.operands[0]._is_positive:
            return False
        elif self.kind in {"square", "absolute"} and self.operands[0]._is_negative:
            return False

    @property
    @_cache_in_props
    def _is_positive(self):
        r = self._is_nonpositive
        if r is not None:
            return not r

    @property
    @_cache_in_props
    def _is_negative(self):
        r = self._is_nonnegative
        if r is not None:
            return not r

    @property
    @_cache_in_props
    def _is_boolean(self):
        if self.kind in {"lt", "le", "gt", "ge", "eq", "ne", "logical_and", "logical_or", "logical_xor", "logical_not"}:
            return True
        elif self.kind == "symbol":
            return self.operands[1].is_boolean
        elif self.kind in {"constant", "select"}:
            return self.operands[1]._is_boolean
        return False

    @property
    @_cache_in_props
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
            "absolute",
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
        elif self.kind in {"complex", "conjugate"}:
            return True
        elif self.kind in {"add", "subtract", "divide", "multiply", "pow"}:
            return self.operands[0].is_complex or self.operands[1].is_complex
        elif self.kind == "list":
            return False
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
            "exp2",
        }:
            return self.operands[0].is_complex
        elif self.kind == "apply":
            return self.operands[-1].is_complex
        elif self.kind == "item":
            container, index = self.operands
            return container.operands[index].is_complex
        else:
            raise NotImplementedError(f"{type(self).__name__}.is_complex not implemented for {self.kind}")

    @_cache_in_props
    def get_type(self):
        if self.kind == "symbol":
            return self.operands[1]
        elif self.kind == "constant":
            return self.operands[1].get_type()
        elif self.kind in {"lt", "le", "gt", "ge", "eq", "ne", "logical_and", "logical_or", "logical_xor", "is_finite"}:
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
            "exp2",
            "expm1",
            "ceil",
            "floor",
            "logical_not",
            "sign",
            "copysign",
            "conjugate",
            "asin_acos_kernel",
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
        elif self.kind in {"absolute", "real", "imag"}:
            t = self.operands[0].get_type()
            return t.complex_part if t.is_complex else t
        elif self.kind == "select":
            return self.operands[1].get_type().max(self.operands[2].get_type())
        elif self.kind == "complex":
            t = self.operands[0].get_type().max(self.operands[1].get_type())
            bits = t.bits * 2 if t.bits is not None else None
            return Type(self.context, "complex", bits)
        elif self.kind == "upcast":
            t = self.operands[0].get_type()
            return Type(self.context, t.kind, t.bits * 2 if t.bits is not None else None)
        elif self.kind == "downcast":
            t = self.operands[0].get_type()
            return Type(self.context, t.kind, t.bits // 2 if t.bits is not None else None)
        elif self.kind == "list":
            return Type(self.context, self.kind, tuple(item.get_type() for item in self.operands))
        elif self.kind == "item":
            ct = self.operands[0].get_type()
            if ct.kind == "list":
                return ct.param[0]
            assert 0, ct.kind  # unreachable
        raise NotImplementedError(f"{type(self).__name__}.get_type not implemented for {self.kind}")


def assert_equal(result, expected):
    assert isinstance(result, type(expected)), (type(result), type(expected))
    if isinstance(expected, Expr):
        assert result is expected, (result.key, expected.key)
    elif isinstance(expected, tuple):
        assert len(result) == len(expected), (len(result), len(expected))
        for result_item, expected_item in zip(result, expected):
            assert_equal(result_item, expected_item)
    else:
        assert result == expected
