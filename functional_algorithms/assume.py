__all__ = ["assume", "check"]

from collections import defaultdict
from .rewrite import op_flatten, op_expand
from .utils import number_types


class Choice:
    """Holds a choice.

    A choice is a logical or-expression of relational operators lt and
    eq. Other types of relational expressions are normalized to use lt
    and eq operators only.

    For example, x >= y is expressed as a choice instance containing:
      is_lt = {y: x}
      is_eq = {x: y, y: x}

    For x != y, we'll have
      is_lt = {x: y, y: x}
      is_eq = {}

    For x > a or x < b, we'll have
      is_lt = {a: x, x: b}
      is_eq = {}

    For x > a and x <= b, we'll have two choice instances associated
    with logical and-operation:
      is_lt = {a: x}           is_lt = {x: b}
      is_eq = {}               is_eq = {x: b, b: x}

    For (x > a and x <= b) or (x > c and x <= d), we'll have two
    assumptions associated with logical or-operation. Both assumptions
    contain two choice instances as explained above.

    """

    def __init__(self, relational_expr):
        self.is_lt = dict()  # {lhs.key: rhs)
        # is_gt mirrors is_lt
        self.is_gt = dict()  # {rhs.key: lhs)
        self.is_eq = dict()  # {lhs.key: rhs, rhs.key: lhs}
        # is_ne mirrors is_lt
        self.is_ne = dict()  # {lhs.key: rhs, rhs.key: lhs}

        x, y = relational_expr.operands
        if relational_expr.kind == "lt":
            self.is_lt[x.key] = y
            self.is_gt[y.key] = x
        elif relational_expr.kind == "gt":
            self.is_gt[x.key] = y
            self.is_lt[y.key] = x
        elif relational_expr.kind == "eq":
            self.is_eq[x.key] = y
            self.is_eq[y.key] = x
        elif relational_expr.kind == "le":
            self.is_lt[x.key] = y
            self.is_gt[y.key] = x
            self.is_eq[x.key] = y
            self.is_eq[y.key] = x
        elif relational_expr.kind == "ge":
            self.is_gt[x.key] = y
            self.is_lt[y.key] = x
            self.is_eq[x.key] = y
            self.is_eq[y.key] = x
        elif relational_expr.kind == "ne":
            self.is_ne[x.key] = y
            self.is_ne[y.key] = x
        else:
            assert 0, relational_expr.kind  # unreachable

    def register(self, x):
        x.props


class Assumption:
    """Holds a list of assumptions.

    An assumption is a logical and-expression of choices.
    """

    def __init__(self):
        self.choices = []


def assume(logical_expr):
    for logical_and_expr in op_expand(
        logical_expr, commutative=True, idempotent=True, over_commutative=True, over_idempotent=True, kind="logical_and"
    ):
        # print("AAA")
        assumption = Assumption()
        for expr in op_flatten(logical_and_expr, commutative=True, idempotent=True, kind="logical_and"):
            assert expr.kind in {"lt", "gt", "le", "ge", "eq", "ne"}, expr.kind
            # print(expr.key)

            choice = Choice(expr)


def assume_relational(x, rop, y):
    n = f"_is_{rop}"
    m = f'_is_{dict(lt="gt", le="ge", gt="lt", ge="le", eq="eq", ne="ne")[rop]}'
    if n not in x.props:
        x.props[n] = []
    if m not in y.props:
        y.props[m] = []
    x.props[n].append(y)
    y.props[m].append(x)


def _assume(logical_expr):
    for expr in op_flatten(logical_expr, commutative=True, idempotent=True, kind="logical_and"):
        if expr.kind in {"lt", "le", "gt", "ge", "eq", "ne"}:
            x, y = expr.operands
            assume_relational(x, expr.kind, y)
        else:
            raise NotImplementedError(expr.kind)


def check_relational(x, rop, y):
    rel_op = dict(
        lt=lambda x, y: x < y,
        le=lambda x, y: x <= y,
        gt=lambda x, y: x > y,
        ge=lambda x, y: x >= y,
        eq=lambda x, y: x == y,
        ne=lambda x, y: x != y,
    )[rop]

    if x.kind == "constant":
        xvalue, xlike = x.operands
        if y.kind == "constant":
            yvalue, ylike = y.operands
            if isinstance(xvalue, number_types) and isinstance(yvalue, number_types):
                return rel_op(x, y)

    swap_rop = dict(lt="gt", le="ge", gt="lt", ge="le", eq="eq", ne="ne")[rop]
    for y_ in x.props.get("_is_{rop}", []):
        if y_ is y:
            return True

    # print(f"{x=}")
    # print(f"{rop, swap_rop=}")
    # print(f"{y=}")

    for x_ in y.props.get("_is_{swap_rop}", []):
        if x_ is x:
            return True


def check_logical_and(logical_and_expr):
    for expr in op_flatten(logical_and_expr, commutative=True, idempotent=True, kind="logical_and"):
        r = check(expr)
        if r is None:
            return None
        if r is False:
            return False
    return True


def check(logical_expr):
    result = False
    for expr in op_flatten(logical_expr, commutative=True, idempotent=True, kind="logical_or"):
        if expr.kind in {"lt", "le", "gt", "ge", "eq", "ne"}:
            x, y = expr.operands
            r = check_relational(x, expr.kind, y)
        elif expr.kind == "logical_and":
            r = check_logical_and(expr)
        else:
            raise NotImplementedError(expr.kind)
        if r is None:
            result = None
        if r is True:
            return True
    return result
