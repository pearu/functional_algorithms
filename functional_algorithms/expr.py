
from .utils import UNSPECIFIED
from . import algorithms
from .typesystem import Type

def normalize_like(expr):
    while True:
        if expr.kind in {'constant', 'select'}:
            expr = expr.operands[1]
        elif expr.kind in {
                'negative', 'positive', 'add', 'subtract', 'multiply', 'divide',
                'maximum', 'minimum', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh',
                'cos', 'cosh', 'sin', 'sinh', 'tan', 'tanh', 'exp', 'expm1', 'log', 'log1p', 'log2',
                'log10', 'conj', 'hypot', 'sqrt', 'square',
        }:
            expr = expr.operands[0]
        elif expr.kind == 'abs' and not expr.operands[0].is_complex:
            expr = expr.operands[0]
        else:
            break
    return expr

def make_constant(context, value, like_expr, props=UNSPECIFIED):
    # The type of value is defined by the type of like expression
    # which some targets use implicitly to define the constant type.
    if isinstance(value, str):
        value = {'+inf': 'posinf', 'inf': 'posinf', 'pinf': 'posinf',
                 '-inf': 'neginf', 'ninf': 'neginf'}.get(value, value)
    return Expr(context, 'constant', (value, normalize_like(like_expr)), props)


def make_symbol(context, name, typ, props=UNSPECIFIED):
    # All symbols must have a type.
    typ = Type.fromobject(context, typ)
    return Expr(context, 'symbol', (name, typ), props)

def make_apply(context, name, args, result, props=UNSPECIFIED):
    return Expr(context, 'apply', (name, *args, result), props)


def normalize(operands):
    """Convert numbers to constant expressions
    """
    ref_operand = [operand for operand in operands if isinstance(operand, Expr)][0]
    new_operands = []
    for operand in operands:
        if isinstance(operand, (int, float, str)):
            operand = make_constant(ref_operand.context, operand, ref_operand)
        new_operands.append(operand)
    return tuple(new_operands)


def make_ref(expr):
    ref = expr.props.get('ref', UNSPECIFIED)
    if ref is not UNSPECIFIED:
        return ref
    if expr.kind == 'symbol':
        return f'{expr.kind}_{expr.operands[0]}'
    if expr.kind == 'constant':
        return f'{expr.kind}_{expr.operands[0]}'
    lst = [expr.kind] + list(map(make_ref, expr.operands))
    return '_'.join(lst)


class Expr:

    def __new__(cls, context, kind, operands, props):
        obj = object.__new__(cls)
        obj.context = context
        obj.kind = kind
        if kind not in {'symbol', 'constant'}:
            if kind == 'select':
                operands = operands[:1] + normalize(operands[1:])
            else:
                operands = normalize(operands)
        obj.operands = operands
        # expressions are singletons within the given context and are
        # uniquely identified by its serialized string value. However,
        # expression props are mutable. Therefore. mutations (e.g. ref
        # updates) have global effect within the given context.
        obj._serialized = obj._serialize()

        # When specified, props is a dictionary that contains ref but
        # otherwise its content could be anything
        if props is UNSPECIFIED:
            props = dict()

        # When ref is specified in props, it is used as a variable
        # name referencing the expression. When an expression with
        # given ref is used as an operand in some other expression,
        # then printing the expression will replace the operand with
        # its reference value and `<ref> = <expr>` will be added to
        # assignments list (see targets.<target>.Printer).
        #
        # If ref is UNSPECIFIED, context.update_refs will assign ref
        # value to the variable name in locals() that object is
        # identical to the expression object.
        #
        # If ref is None, no assignment of ref value will be
        # attempted.
        ref = props.get('ref', UNSPECIFIED)
        assert isinstance(ref, str) or ref in {None, UNSPECIFIED}, props

        obj.props = props
        return context.register_expression(obj)

    def props_(self, **props):
        self.props.update(props)
        if 'ref' in props:
            self.context.update_expression_ref(self, props['ref'])
        return self
    
    def _serialize(self):
        if self.kind == 'symbol':
            return f'{self.operands[0]}:{self.operands[1]}'
        if self.kind == 'constant':
            return f'{self.operands[0]}:type({self.operands[1]._serialized})'
        return f'{self.kind}({",".join(operand._serialized for operand in self.operands)})'

    def _compute_need_ref(self, need_ref: dict) -> None:
        ref = self.ref

        if ref is None:
            need_ref[ref] = False
        elif ref not in need_ref:
            # the first usage of expression with ref does not require
            # using ref, unless forced.
            flag = self.props.get('force_ref', False)
            need_ref[ref] = self.props.get('force_ref', False)
        else:
            # expression with ref is used more than once, hence we'll
            # mark it as needed
            need_ref[ref] = True
            return

        for operand in self.operands:
            if isinstance(operand, Expr):
                operand._compute_need_ref(need_ref)

    def implement_missing(self, target):
        if self.kind == 'symbol':
            return self
        props = self.props.copy()
        if self.kind == 'constant':
            like = self.operands[1].implement_missing(target)
            if like is self.operands[1]:
                return self
            value = self.operands[0]
            return make_constant(self.context, value, like, props)

        if self.kind == 'apply':
            body = self.operands[-1].implement_missing(target)
            if body is self.operands[-1]:
                return self
            return make_apply(self.context, self.operands[0], self.operands[1:-1], body, props)

        if target.kind_to_target.get(self.kind, NotImplemented) is NotImplemented:
            func = NotImplemented
            for m in self.context._paths:
                func = getattr(m, self.kind, NotImplemented)
                if func is not NotImplemented:
                    break
            if func is NotImplemented:
                paths = ':'.join([m.__name__ for m in self.context._paths])
                raise NotImplementedError(f'{self.kind} for {target.__name__.split(".")[-1]} target [paths={paths}]')

            # func may call context.update_refs and to avoid ref value
            # conflicts in the func and its caller, func update_ref
            # will use stack name as a prefix to its variables.
            save_stack_name = self.context._stack_name
            self.context._stack_name= self.context._stack_name + '_' + func.__name__
            self.context._stack_call_count[self.context._stack_name] += 1

            assert '.' not in self.context._stack_name, self.context._stack_name

            try:
                expr = func(self.context, *self.operands).props_(**props)
            finally:
                #self.context._ref_prefix = prev_ref_prefix
                self.context._stack_name = save_stack_name

            result = expr.implement_missing(target)
            if isinstance(self.props.get('ref', UNSPECIFIED), str):
                self.context.update_expression_ref(result, self.props['ref'])
            return result

        operands = tuple([operand.implement_missing(target) for operand in self.operands])
        for o1, o2 in zip(operands, self.operands):
            if o1 is not o2:
                break
        else:
            return self
        return Expr(self.context, self.kind, operands, props)

    def simplify(self):
        if self.kind in {'symbol', 'constant'}:
            return self
        props = self.props.copy()
        if self.kind == 'apply':
            body = self.operands[-1].simplify()
            if body is self.operands[-1]:
                return self
            return make_apply(self.context, self.operands[0], self.operands[1:-1], body, props)            

        if self.kind == 'abs' and self.operands[0].kind == 'abs':
            return self.operands[0]

        operands = tuple([operand.simplify() for operand in self.operands])
        for o1, o2 in zip(operands, self.operands):
            if o1 is not o2:
                break
        else:
            return self
        return Expr(self.context, self.kind, operands, props)

    def tostring(self, target, tab='', need_ref=None, debug=0):
        if need_ref is None:
            need_ref = dict()
            self._compute_need_ref(need_ref)
        return target.Printer(need_ref, debug=debug).tostring(self, tab=tab)

    @property
    def ref(self):
        if 'ref' in self.props:
            ref = self.props.get('ref')
            if ref is UNSPECIFIED:
                ref = None
        else:
            ref = None
        if ref is None:
            ref = make_ref(self)
            assert ref not in self.context._ref_values, ref
        return ref

    def is_same(self, other):  # NOT USED, remove?
        if self is other:
            return True
        if self.kind == other.kind and len(self.operands) == len(other.operands):
            for x, y in zip(self.operands, other.operands):
                if isinstance(x, Expr):
                    if not x.is_same(y):
                        break
                elif x != y:
                    break
            else:
                return True
        return False

    def tostr(self):
        if self.kind == 'symbol':
            return self.operands[0]
        if self.kind == 'constant':
            return str(self.operands[0])
        if self.ref:
            return self.ref
        return f'{self.kind}(' + ', '.join([operand.tostr() for operand in self.operands]) + ')'

    def __str__(self):
        if self.kind == 'symbol':
            return f'{self.operands[0]}: {self.operands[1]}'
        if self.kind == 'constant':
            return str(self.operands[0])
        if self.kind == 'apply':
            name = str(self.operands[0])
            args = ', '.join(map(str, self.operands[1:-1]))
            result = str(self.operands[-1])
            return f'{self.kind}({name}, ({args}), {result})'
        args = ', '.join(map(str, self.operands))
        return f'{self.kind}({args})'

    def __repr__(self):
        return f'{type(self).__name__}({self.kind}, {self.operands}, {self.props})'
    
    def __abs__(self): return self.context.abs(self)
    def __neg__(self): return self.context.negative(self)
    def __pos__(self): return self.context.pos(self)
    def __invert__(self): return self.context.invert(self)

    def __add__(self, other): return self.context.add(self, other)
    def __radd__(self, other): return self.context.add(other, self)
    def __sub__(self, other): return self.context.subtract(self, other)
    def __rsub__(self, other): return self.context.subtract(other, self)
    def __mul__(self, other): return self.context.multiply(self, other)
    def __rmul__(self, other): return self.context.multiply(other, self)
    def __truediv__(self, other): return self.context.divide(self, other)
    def __rtruediv__(self, other): return self.context.divide(other, self)
    def __floordiv__(self, other): return self.context.floor_divide(self, other)
    def __rfloordiv__(self, other): return self.context.floor_divide(other, self)
    def __pow__(self, other): return self.context.pow(self, other)
    def __rpow__(self, other): return self.context.pow(other, self)
    def __mod__(self, other): return self.context.reminder(self, other)
    def __rmod__(self, other): return self.context.reminder(other, self)
    def __and__(self, other): return self.context.bitwise_and(self, other)
    def __rand__(self, other): return self.context.bitwise_and(other, self)
    def __or__(self, other): return self.context.bitwise_or(self, other)
    def __ror__(self, other): return self.context.bitwise_or(other, self)
    def __xor__(self, other): return self.context.bitwise_xor(self, other)
    def __rxor__(self, other): return self.context.bitwise_xor(other, self)

    def __lshift__(self, other): return self.context.left_shift(self, other)
    def __rlshift__(self, other): return self.context.left_shift(other, self, ref=None)
    def __rshift__(self, other): return self.context.right_shift(self, other)
    def __rrshift__(self, other): return self.context.right_shift(other, self, ref=None)

    def __round__(self, ndigits=None): return self.context.round(self, ndigits)
    def __trunc__(self): return self.context.trunc(self)
    def __floor__(self): return self.context.floor(self)
    def __ceil__(self): return self.context.ceil(self)
    
    @property
    def real(self): return self.context.real(self)
    @property
    def imag(self): return self.context.imag(self)

    def __lt__(self, other): return self.context.lt(self, other)
    def __le__(self, other): return self.context.le(self, other)
    def __gt__(self, other): return self.context.gt(self, other)
    def __ge__(self, other): return self.context.ge(self, other)
    def __eq__(self, other): return self.context.eq(self, other)  # TODO: why it is not effective?
    def __ne__(self, other): return self.context.ne(self, other)

    @property
    def is_posinf(self): return self == self.context.constant('posinf', self)
    @property
    def is_neginf(self): return self == self.context.constant('neginf', self)

    @property
    def is_complex(self):
        if self.kind in {'symbol', 'constant', 'select'}:
            return self.operands[1].is_complex
        elif self.kind in {
                'lt', 'le', 'gt', 'ge', 'eq', 'ne', 'real', 'imag', 'abs',
                'logical_and', 'logical_or', 'logical_xor', 'logical_not',
                'bitwise_invert', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_left_shift',
                'bitwise_right_shift', 'ceil', 'floor', 'hypot', 'maximum', 'minimum',
                'floor_divide', 'remainder'
        }:
            return False
        elif self.kind in {'complex', 'conj'}:
            return True
        elif self.kind in {'add', 'subtract', 'divide', 'multiply', 'pow'}:
            return self.operands[0].is_complex or self.operands[1].is_complex
        elif self.kind in {'positive', 'negative', 'sqrt', 'square', 'asin', 'acos', 'atan',
                           'asinh', 'acosh', 'atanh', 'sinh', 'cosh', 'tanh', 'sin', 'cos', 'tan',
                           'log', 'log1p', 'log2', 'log10', 'exp', 'expm1'}:
            return self.operands[0].is_complex
        elif self.kind == 'apply':
            return self.operands[-1].is_complex
        else:
            raise NotImplementedError(f'Expr.is_complex for {self.kind}')

    def get_type(self):
        if self.kind == 'symbol':
            return self.operands[1]
        if self.kind == 'constant':
            return self.operands[1].get_type()
        if self.kind in {'lt', 'le', 'gt', 'ge', 'eq', 'ne', 'logical_and', 'logical_or', 'logical_xor'}:
            return Type.fromobject(self.context, 'boolean')
        elif self.kind in {'positive', 'negative', 'sqrt', 'square', 'asin', 'acos', 'atan',
                           'asinh', 'acosh', 'atanh', 'sinh', 'cosh', 'tanh', 'sin', 'cos', 'tan',
                           'log', 'log1p', 'log2', 'log10', 'exp', 'expm1', 'ceil', 'floor',
                           'logical_not'}:
            return self.operands[0].get_type()
        elif self.kind in {'add', 'subtract', 'divide', 'multiply', 'pow', 'maximum', 'minimum',
                           'hypot', 'remainder', 'atan2'}:
            return self.operands[0].get_type().max(self.operands[1].get_type())
        elif self.kind in {'abs', 'real', 'imag'}:
            t = self.operands[0].get_type()
            return t.complex_part if t.is_complex else t
        elif self.kind == 'select':
            return self.operands[1].get_type().max(self.operands[2].get_type())
        elif self.kind == 'complex':
            t = self.operands[0].get_type().max(self.operands[1].get_type())
            bits = t.bits * 2 if t.bits is not None else None
            return Type(self.context, 'complex', bits)
        raise NotImplementedError(self.kind)
