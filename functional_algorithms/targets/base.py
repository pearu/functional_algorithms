import warnings

from ..expr import Expr


def modifier_base(target, expr):
    if expr.kind in {"symbol", "constant", "apply"}:
        pass
    elif target.kind_to_target.get(expr.kind, NotImplemented) is NotImplemented:
        ctx = expr.context
        if ctx.parameters.get(f"use_native_{expr.kind}", False):
            raise NotImplementedError(f'{target.__name__.split(".")[-1]} target does not define native {expr.kind}')

        func = NotImplemented
        for m in ctx._paths:
            func = getattr(m, expr.kind, NotImplemented)
            if func is not NotImplemented:
                break
        if func is NotImplemented:
            paths = ":".join([m.__name__ for m in ctx._paths])
            raise NotImplementedError(f'{expr.kind} for {target.__name__.split(".")[-1]} target [paths={paths}]')

        result = ctx.call(func, expr.operands)
        if expr.key != result.key:
            # We will call rewrite again because when func result is a
            # new expression then it needs to be rewritten as
            # well. However, we don't know what was the user-specifed
            # deep_first value. FIX ME: call rewrite with
            # user-specified deep_first value.
            return result.rewrite(target, deep_first=True)
    return expr


class PrinterBase:

    force_cast_arguments = False
    constant_target = None

    def __init__(self, need_ref, debug=1):
        self.need_ref = need_ref
        self.defined_refs = set()
        self.assignments = []
        if self.constant_target is not None:
            self.constant_printer = self.constant_target.Printer(need_ref, debug=debug)
            self.constant_printer.assignments = self.assignments
            self.constant_printer.defined_refs = self.defined_refs
        else:
            self.constant_printer = self
        self.debug = debug

    def get_type(self, expr):
        typ = expr.get_type()
        if typ.kind == "type":
            typ = str(typ.param)
        else:
            typ = str(typ)
            typ = self.type_to_target[typ]
        return typ

    def make_assignment(self, typ, var, value):
        raise NotImplementedError(type(self))

    def make_constant(self, like, value):
        raise NotImplementedError(type(self))

    def make_apply(self, expr, name, tab=""):
        raise NotImplementedError(type(self))

    def show_value(self, var):
        return NotImplemented

    def check_dtype(self, var, dtype):
        return NotImplemented

    def tostring(self, expr, tab=""):
        if expr.ref in self.defined_refs:
            assert self.need_ref.get(expr.ref), expr.ref
            return expr.ref

        if expr.kind == "apply":
            # add arguments to defined refs
            for a in expr.operands[1:-1]:
                self.defined_refs.add(a.ref)

                if self.force_cast_arguments:
                    self.assignments.append(self.make_assignment(None, a.ref, self.make_constant(a, a)))

                if self.debug >= 2:
                    stmt = self.show_value(a.ref)
                    if isinstance(stmt, str):
                        self.assignments.append(stmt)
            name = expr.props.get("name", expr.operands[0])
            if not isinstance(name, str):
                assert name.kind == "symbol", name
                name = name.operands[0]
            result = self.make_apply(expr, name, tab=tab)

        elif expr.kind == "symbol":
            result = str(expr.operands[0])

        elif expr.kind == "constant":
            value, like = expr.operands
            if isinstance(value, Expr):
                result = self.make_constant(like, self.constant_printer.tostring(value))
            elif isinstance(value, str):
                target_value = self.constant_to_target.get(value, NotImplemented)
                if target_value is not NotImplemented:
                    typ = self.get_type(expr)
                    target_value = target_value.format(type=typ)
                    result = self.make_constant(like, target_value)
                else:
                    warnings.warn(
                        f"{type(self).__module__}.{type(self).__name__}.constant_to_target does not implement {value}"
                    )
                    result = self.make_constant(like, value)
            else:
                result = self.make_constant(like, value)

        else:
            # Generic printer for operations
            tmpl = self.kind_to_target.get(expr.kind, NotImplemented)
            if tmpl is NotImplemented:
                raise NotImplementedError(
                    f"{type(self).__module__}.{type(self).__name__}.kind_to_target does not implement `{expr.kind}`"
                )
            if callable(tmpl):
                result = tmpl(self, expr)
            else:
                m = dict()
                for i in range(len(expr.operands)):
                    m["typeof_0"] = self.get_type(expr.operands[i])
                result = tmpl.format(*[self.tostring(operand) for operand in expr.operands], **m)

        assert expr.ref is not None

        if self.need_ref.get(expr.ref):
            assert expr.ref not in self.defined_refs, expr.ref
            self.defined_refs.add(expr.ref)
            self.assignments.append(self.make_assignment(self.get_type(expr), expr.ref, result))
            result = expr.ref

            if self.debug >= 2:
                stmt = self.show_value(expr.ref)
                if isinstance(stmt, str):
                    self.assignments.append(stmt)

            if self.debug >= 1:
                stmt = self.check_dtype(expr.ref, self.get_type(expr))
                if isinstance(stmt, str):
                    self.assignments.append(stmt)

        return result
