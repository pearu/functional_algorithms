r"""Design of assumption system
---------------------------

The assumption system is needed for branching between different
computational paths depending on the conditions postulated on inputs.

As a simple example, if `x` is non-negative, `abs(x)` can be evaluated
as `x`, and if `x` is negative, then as `-x`. The sign of `x` can be
specifed using `assume` function. Here are three possible
alternatives:

  assume(x > -3)
  assume(x > 2)
  assume(x < -1)

When no assumptions are given, or when `x > -3` is assumed, `abs(x)`
cannot be rewritten. In other cases, however, `abs(x)` can be
rewritten as `x` or `-x` when `x > 2` or `x < -1` is assumed,
respectively.

Consider a system of logical expressions that involve relational
expressions and logical operations such as conjunction, disjunction,
negation, or exclusive disjunction. When collecting all non-boolean
symbols in this system (that is, all the symbols that are present in
relational expressions), the truth value of the system (`logical_expr
== true` for all logical expressions in the system) defines a subset
in a multi-dimensional space that coordinates are the non-boolean
symbols.

For example, the following logical expression

  2 * x + y > 0  and  x <= 0

has two non-boolean symbols x and y and the given system of
inequalities defines a subset in the second quarter of x-y plane that
looks like unbounded triangle:

                      ^ y
              ........| 
              \xxxxxxx|
               \xxxxxx|
                \xxxxx|
                 \xxxx|
                  \xxx|
                   \xx|
                    \x|
                     \|
   -------------------0-------------------> x

When calling

  assume(y > -2 * x)
  assume(x <= 0)

then it means that all possible values to symbols x, y are those
point pairs that are in the subset defined by the system of
inequalities defined above.

We define the following tools for the assumption system:

- `is_subset_of(logical_expr, other_logical_expr)` that returns true
  when a subset A (defined logical_expr) is a subset of B (defined by
  other_logical_expr), false when there exists A subset that is not a
  subset of B, or result_logical_expr that defines a set R such that
    B = intersection(A, R)

  For example,

     is_subset_of((y > -2 * x) & (x <= 0), y >= 0) -> true
     is_subset_of((y > -2 * x) & (x <= 0), y <= 0) -> false
     is_subset_of((y > -2 * x) & (x <= 0), y == 1) -> y == 1

- `assume(logical_expr)` that defines restrictions to non-boolean
  subexpressions used in the specified logical expression. The
  restrictions are saved in `<Expr instance>.props['_assumptions']`
  dictionary.

- `assume(nonlogical_expr)` that generates restrictions to
  subexpressions used in the specified non-logical expression.

- `check(logical_expr)`
"""

__all__ = ["assume", "check"]

from collections import defaultdict
from . import rewrite
from .utils import number_types
from .targets import symbolic


def _rewrite(mth):
    def wrapper(self, *args):
        r = mth(self, *args)
        if r is not None:
            r = r.rewrite(rewrite)
        return r

    return wrapper


@_rewrite
def is_subset_of(logical_expr, other):
    """Check if logical_expr is a subset of other logical expression.

    Definition:

       logical_expr(x)   is a subset of   other logical_expr(x)

    when

       other logical_expr(x) is true for all x such that logical_expr(x) is true

    where x is a set of all non-boolean symbols present in logical
    expressions.

    The logical expression may contain logical `or`, logical `and`, or
    relational operations.

    The `is_subset_of` function either returns true, false, or a
    result logical expression such that

      is_subset_of(logical_expr & result_logical_expr, other_logical_expr)

    would return true.
    """

    boolean = logical_expr.context.constant
    false = boolean(False)
    true = boolean(True)

    if other.kind == "logical_and":
        operands = []
        for o in rewrite.op_flatten(other, commutative=True, idempotent=True):
            r = is_subset_of(logical_expr, o)
            if r is false:
                return false
            if r is true:
                continue
            operands.append(o if r is None else r)
        if operands:
            return rewrite.op_unflatten(operands, "logical_and")
        return true

    elif other.kind == "logical_or":
        operands = []
        for o in rewrite.op_flatten(other, commutative=True, idempotent=True):
            r = is_subset_of(logical_expr, o)
            if r is true:
                return true
            if r is false:
                continue
            operands.append(o if r is None else r)
        if operands:
            return rewrite.op_unflatten(operands, "logical_or")
        return false

    elif logical_expr.kind == "logical_and":
        input_operands = list(rewrite.op_flatten(logical_expr, commutative=True, idempotent=True))
        operands = []
        for expr in input_operands:
            r = is_subset_of(expr, other)
            if r is true:
                return true
            if r is false:
                return false
            assert r is not None
            if [None for inp in input_operands if normalize(inp).key == normalize(r).key]:
                continue
            operands.append(r)
        if operands:
            return rewrite.op_unflatten(operands, "logical_or")
        return true

    elif logical_expr.kind == "logical_or":
        operands = []
        for expr in rewrite.op_flatten(logical_expr, commutative=True, idempotent=True):
            r = is_subset_of(expr, other)
            if r is true:
                continue
            if r is false:
                return false
            assert r is not None
            operands.append(r)
        if operands:
            return rewrite.op_unflatten(operands, "logical_and")
        return false

    elif logical_expr.kind in {"lt", "gt", "le", "ge", "eq", "ne"} and logical_expr.kind in {
        "lt",
        "gt",
        "le",
        "ge",
        "eq",
        "ne",
    }:
        r = relational_is_subset_of(logical_expr, other)

        return r

    raise NotImplementedError((logical_expr.kind, other.kind))


def normalize(expr):
    if expr.kind == "ge":
        return expr.context.le(*(expr.operands[::-1]))
    elif expr.kind == "gt":
        return expr.context.lt(*(expr.operands[::-1]))
    elif expr.kind == "eq" and expr.operands[0]._is_constant:
        return expr.context.eq(*(expr.operands[::-1]))
    elif expr.kind == "ne" and expr.operands[0]._is_constant:
        return expr.context.ne(*(expr.operands[::-1]))
    return expr


def relational_is_subset_of(rel_expr, other):
    ctx = rel_expr.context
    boolean = ctx.constant
    false = boolean(False)
    true = boolean(True)

    r1 = normalize(rel_expr)
    r2 = normalize(other)

    rop1, (lhs1, rhs1) = r1.kind, r1.operands
    rop2, (lhs2, rhs2) = r2.kind, r2.operands

    def ispair(x, y):
        if x._is_constant:
            return False
        return x is y

    def make_largest(x):
        return x.context.constant("largest", x)

    def is_eq_posinf(x):
        return x == x.context.constant("posinf", x)

    def is_eq_neginf(x):
        return x == x.context.constant("neginf", x)

    def is_ne_posinf(x):
        return x != x.context.constant("posinf", x)

    def is_ne_neginf(x):
        return x != x.context.constant("neginf", x)

    def is_ne_inf(x):
        return (x != x.context.constant("neginf", x)) & (x != x.context.constant("posinf", x))

    def is_lt_largest(x):
        return x < x.context.constant("largest", x)

    if rop1 == "lt":
        #    x < a      is a subset of      x < b             if (a <= b) and b != -inf
        #    x < a      is a subset of      b < x             if b < x and b < a and b != -inf
        #    a < x      is a subset of      x < b             if x < b and a < b and b != inf
        #    a < x      is a subset of      b < x             if (b <= a) and b != inf

        #    x < a      is a subset of      x <= b            if a <= b or b == -inf
        #    x < a      is a subset of      b <= x            if (b <= x and b < a) or (b == -inf)
        #    a < x      is a subset of      x <= b            if (x <= b and b > a) or (b == inf)
        #    a < x      is a subset of      b <= x            if b <= a or b == inf

        #    x < a      is a subset of      x == b            if (b == x and a > b) and (abs(b) != inf)
        #    x < a      is a subset of      b == x            if (b == x and a > b)
        #    a < x      is a subset of      x == b            if (b == x and a < b) and (abs(b) != inf)
        #    a < x      is a subset of      b == x            if (b == x and a < b)

        #    x < a      is a subset of      x != b            if (b >= a) and (b is not x) or (b == -inf and -a < largest)
        #    x < a      is a subset of      b != x            if (b >= a) and (b is not x) or (b == -inf and -a < largest)
        #    a < x      is a subset of      x != b            if (b <= a) and (b is not x) or (b == inf and a < largest)
        #    a < x      is a subset of      b != x            if (b <= a) and (b is not x) or (b == inf and a < largest)
        if rop2 == "lt":
            if lhs2 is rhs2:
                return false
            if ispair(lhs1, lhs2):
                return (rhs1 <= rhs2) & is_ne_neginf(rhs2)
            if ispair(lhs1, rhs2):
                return (lhs2 < rhs2) & (lhs2 < rhs1) & is_ne_neginf(lhs2)
            if ispair(rhs1, lhs2):
                return (lhs2 < rhs2) & (lhs1 < rhs2) & is_ne_posinf(rhs2)
            if ispair(rhs1, rhs2):
                return (lhs2 <= lhs1) & is_ne_posinf(lhs2)
        elif rop2 == "le":
            if lhs2 is rhs2:
                return true
            if ispair(lhs1, lhs2):
                return (rhs1 <= rhs2) | is_eq_neginf(rhs2)
            if ispair(lhs1, rhs2):
                return ((lhs2 <= rhs2) & (lhs2 < rhs1)) | is_eq_neginf(lhs2)
            if ispair(rhs1, lhs2):
                return ((lhs2 <= rhs2) & (lhs1 < rhs2)) | is_eq_posinf(rhs2)
            if ispair(rhs1, rhs2):
                return (lhs2 <= lhs1) | is_eq_posinf(lhs2)
        elif rop2 == "eq":
            if lhs2 is rhs2:
                return true
            if ispair(lhs1, lhs2):
                return ((lhs2 == rhs2) & (rhs1 > rhs2)) & is_ne_inf(rhs2)
            if ispair(lhs1, rhs2):
                return (lhs2 == rhs2) & (rhs1 > lhs2)
            if ispair(rhs1, lhs2):
                return ((lhs2 == rhs2) & (lhs1 < rhs2)) & is_ne_inf(rhs2)
            if ispair(rhs1, rhs2):
                return (lhs2 == rhs2) & (lhs1 < lhs2)
        elif rop2 == "ne":
            if lhs2 is rhs2:
                return false
            if ispair(lhs1, lhs2):
                return (rhs2 >= rhs1) | (is_eq_neginf(rhs2) & is_lt_largest(-rhs1))
            if ispair(lhs1, rhs2):
                return (lhs2 >= rhs1) | (is_eq_neginf(lhs2) & is_lt_largest(-rhs1))
            if ispair(rhs1, lhs2):
                return (rhs2 <= lhs1) | (is_eq_posinf(rhs2) & is_lt_largest(lhs1))
            if ispair(rhs1, rhs2):
                return (lhs2 <= lhs1) | (is_eq_posinf(lhs2) & is_lt_largest(lhs1))
        else:
            assert 0  # unreachable
    elif rop1 == "le":
        #    x <= a      is a subset of      x < b             if a < b
        #    x <= a      is a subset of      b < x             if b < x and b < a and b != -inf
        #    a <= x      is a subset of      x < b             if x < b and a < b and b != inf
        #    a <= x      is a subset of      b < x             if b < a

        #    x <= a      is a subset of      x <= b            if a <= b
        #    x <= a      is a subset of      b <= x            if (x >= b and b < a) or (x == b and a == b) or (b == -inf)
        #    a <= x      is a subset of      x <= b            if (x <= b and a < b) or (x == b and a == b) or (b == inf)
        #    a <= x      is a subset of      b <= x            if b <= a

        #    x <= a      is a subset of      x == b            if (a == b and a == -inf) or (x == b and b <= a)
        #    x <= a      is a subset of      b == x            if (a == b and a == -inf)
        #    a <= x      is a subset of      x == b            if (a == b and a == inf) or (x == b and b >= a)
        #    a <= x      is a subset of      b == x            if (a == b and a == inf) or (x == b and b >= a)

        #    x <= a      is a subset of      x != b            if (b > a) and (b is not x)
        #    x <= a      is a subset of      b != x            if (b > a) and (b is not x)
        #    a <= x      is a subset of      x != b            if (b < a) and (b is not x)
        #    a <= x      is a subset of      b != x            if (b < a) and (b is not x)
        if rop2 == "lt":
            if lhs2 is rhs2:
                return false
            if ispair(lhs1, lhs2):
                return rhs1 < rhs2
            if ispair(lhs1, rhs2):
                return (lhs2 < rhs2) & (lhs2 < rhs1) & is_ne_neginf(lhs2)
            if ispair(rhs1, lhs2):
                return (lhs2 < rhs2) & (rhs2 > lhs1) & is_ne_posinf(rhs2)
            if ispair(rhs1, rhs2):
                return lhs2 < lhs1
        elif rop2 == "le":
            if lhs2 is rhs2:
                return true
            if ispair(lhs1, lhs2):
                return rhs1 <= rhs2
            if ispair(lhs1, rhs2):
                return ((lhs2 <= rhs2) & (lhs2 < rhs1)) | ((lhs2 == rhs2) & (lhs2 == rhs1)) | is_eq_neginf(lhs2)
            if ispair(rhs1, lhs2):
                return ((lhs2 <= rhs2) & (lhs1 < rhs2)) | ((lhs2 == rhs2) & (lhs1 == rhs2)) | is_eq_posinf(rhs2)
            if ispair(rhs1, rhs2):
                return lhs2 <= lhs1
        elif rop2 == "eq":
            if lhs2 is rhs2:
                return true
            if ispair(lhs1, lhs2):
                return ((rhs1 == rhs2) & is_eq_neginf(rhs1)) | ((lhs2 == rhs2) & (rhs2 <= rhs1))
            if ispair(lhs1, rhs2):
                return (rhs1 == lhs2) & is_eq_neginf(rhs1)
            if ispair(rhs1, lhs2):
                return ((lhs1 == rhs2) & is_eq_posinf(lhs1)) | ((lhs2 == rhs2) & (rhs2 >= lhs1))
            if ispair(rhs1, rhs2):
                return ((lhs1 == lhs2) & is_eq_posinf(lhs1)) | ((lhs2 == rhs2) & (lhs2 >= lhs1))
        elif rop2 == "ne":
            if lhs2 is rhs2:
                return false
            if ispair(lhs1, lhs2):
                return (rhs2 > rhs1) & boolean(not ispair(rhs2, lhs1))
            if ispair(lhs1, rhs2):
                return (lhs2 > rhs1) & boolean(not ispair(lhs2, lhs1))
            if ispair(rhs1, lhs2):
                return (rhs2 < lhs1) & boolean(not ispair(rhs2, rhs1))
            if ispair(rhs1, rhs2):
                return (lhs2 < lhs1) & boolean(not ispair(lhs2, rhs1))
        else:
            assert 0  # unreachable
    elif rop1 == "eq":
        #    x == a      is a subset of      x < b             if a < b
        #    a == x      is a subset of      x < b             if a < b
        #    x == a      is a subset of      b < x             if a > b
        #    a == x      is a subset of      b < x             if a > b

        #    x == a      is a subset of      x <= b            if a <= b
        #    a == x      is a subset of      x <= b            if a <= b
        #    x == a      is a subset of      b <= x            if a >= b
        #    a == x      is a subset of      b <= x            if a >= b

        #    x == a      is a subset of      x == b            if (a == b) or (x is b)
        #    a == x      is a subset of      x == b            if (a == b) or (x is b)
        #    x == a      is a subset of      b == x            if (a == b) or (x is b)
        #    a == x      is a subset of      b == x            if (a == b) or (x is b)

        #    x == a      is a subset of      x != b            if (a != b) and (b is not x)
        #    a == x      is a subset of      x != b            if (a != b) and (b is not x)
        #    x == a      is a subset of      b != x            if (a != b) and (b is not x)
        #    a == x      is a subset of      b != x            if (a != b) and (b is not x)
        if rop2 == "lt":
            if lhs2 is rhs2:
                return false
            if ispair(lhs1, lhs2):
                return rhs1 < rhs2
            if ispair(rhs1, lhs2):
                return lhs1 < rhs2
            if ispair(lhs1, rhs2):
                return rhs1 > lhs2
            if ispair(rhs1, rhs2):
                return lhs1 > lhs2
        elif rop2 == "le":
            if lhs2 is rhs2:
                return true
            if ispair(lhs1, lhs2):
                return rhs1 <= rhs2
            if ispair(rhs1, lhs2):
                return lhs1 <= rhs2
            if ispair(lhs1, rhs2):
                return rhs1 >= lhs2
            if ispair(rhs1, rhs2):
                return lhs1 >= lhs2
        elif rop2 == "eq":
            if lhs2 is rhs2:
                return true
            if ispair(lhs1, lhs2):
                return rhs1 == rhs2
            if ispair(rhs1, lhs2):
                return lhs1 == rhs2
            if ispair(lhs1, rhs2):
                return rhs1 == lhs2
            if ispair(rhs1, rhs2):
                return lhs1 == lhs2
        elif rop2 == "ne":
            if lhs2 is rhs2:
                return false
            if ispair(lhs1, lhs2):
                return rhs1 != rhs2
            if ispair(rhs1, lhs2):
                return lhs1 != rhs2
            if ispair(lhs1, rhs2):
                return rhs1 != lhs2
            if ispair(rhs1, rhs2):
                return lhs1 != lhs2
        else:
            assert 0  # unreachable
    elif rop1 == "ne":
        #    x != a      is a subset of      x < b             if (x < b and (a >= b or x != a) or (b == inf and a == inf)) and (b != -inf)
        #    a != x      is a subset of      x < b             if (x < b and (a >= b or x != a) or (b == inf and a == inf)) and (b != -inf)
        #    x != a      is a subset of      b < x             if (b < x and (a <= b or x != a) or (b == -inf and a == -inf)) and b != inf
        #    a != x      is a subset of      b < x             if (b < x and (a <= b or x != a) or (b == -inf and a == -inf)) and b != inf

        #    x != a      is a subset of      x <= b            if (((x < b and a == b) or (x <= b and a != b)) and (b != -inf)) or (b == inf)
        #    a != x      is a subset of      x <= b            if (((x < b and a == b) or (x <= b and a != b)) and (b != -inf)) or (b == inf)
        #    x != a      is a subset of      b <= x            if (((b < x and a == b) or (b <= x and a != b)) and (b != inf)) or (b == -inf)
        #    a != x      is a subset of      b <= x            if (((b < x and a == b) or (b <= x and a != b)) and (b != inf)) or (b == -inf)

        #    x != a      is a subset of      x == b            if (b is x) or (x == b and a != b)
        #    a != x      is a subset of      x == b            if (b is x) or (x == b and a != b)
        #    x != a      is a subset of      b == x            if (b is x) or (x == b and a != b)
        #    a != x      is a subset of      b == x            if (b is x) or (x == b and a != b)

        #    x != a      is a subset of      x != b            if a == b or (x != b and a != b)
        #    a != x      is a subset of      x != b            if a == b or (x != b and a != b)
        #    x != a      is a subset of      b != x            if a == b or (x != b and a != b)
        #    a != x      is a subset of      b != x            if a == b or (x != b and a != b)
        if rop2 == "lt":
            if lhs2 is rhs2:
                return false
            if ispair(lhs1, lhs2):
                return ((lhs2 < rhs2) | ((rhs1 == rhs2) & is_eq_posinf(rhs1))) & is_ne_neginf(rhs2)
            if ispair(rhs1, lhs2):
                return ((lhs2 < rhs2) | ((lhs1 == rhs2) & is_eq_posinf(lhs1))) & is_ne_neginf(rhs2)
            if ispair(lhs1, rhs2):
                return ((lhs2 < rhs2) | ((rhs1 == lhs2) & is_eq_neginf(rhs1))) & is_ne_posinf(lhs2)
            if ispair(rhs1, rhs2):
                return ((lhs2 < rhs2) | ((rhs1 == lhs2) & is_eq_neginf(lhs1))) & is_ne_posinf(lhs2)
        elif rop2 == "le":
            if lhs2 is rhs2:
                return true
            if ispair(lhs1, lhs2):
                return (
                    (((lhs2 < rhs2) & (rhs1 == rhs2)) | ((lhs2 <= rhs2) & (rhs1 != rhs2))) & is_ne_neginf(rhs2)
                ) | is_eq_posinf(rhs2)
            if ispair(rhs1, lhs2):
                return (
                    (((lhs2 < rhs2) & (lhs1 == rhs2)) | ((lhs2 <= rhs2) & (lhs1 != rhs2))) & is_ne_neginf(rhs2)
                ) | is_eq_posinf(rhs2)
            if ispair(lhs1, rhs2):
                return (
                    (((lhs2 < rhs2) & (rhs1 == lhs2)) | ((lhs2 <= rhs2) & (rhs1 != lhs2))) & is_ne_posinf(lhs2)
                ) | is_eq_neginf(lhs2)
            if ispair(rhs1, rhs2):
                return (
                    (((lhs2 < rhs2) & (lhs1 == lhs2)) | ((lhs2 <= rhs2) & (lhs1 != lhs2))) & is_ne_posinf(lhs2)
                ) | is_eq_neginf(lhs2)
        elif rop2 == "eq":
            if lhs2 is rhs2:
                return true
            if ispair(lhs1, lhs2):
                return (lhs2 == rhs2) & (rhs1 != rhs2)
            if ispair(rhs1, lhs2):
                return (lhs2 == rhs2) & (lhs1 != rhs2)
            if ispair(lhs1, rhs2):
                return (lhs2 == rhs2) & (rhs1 != lhs2)
            if ispair(rhs1, rhs2):
                return (lhs2 == rhs2) & (lhs1 != lhs2)
        elif rop2 == "ne":
            if lhs2 is rhs2:
                return false
            if ispair(lhs1, lhs2):
                return (rhs1 == rhs2) | ((lhs2 != rhs2) & (rhs1 != rhs2))
            if ispair(rhs1, lhs2):
                return (lhs1 == rhs2) | ((lhs2 != rhs2) & (lhs1 != rhs2))
            if ispair(lhs1, rhs2):
                return (rhs1 == lhs2) | ((lhs2 != rhs2) & (rhs1 != lhs2))
            if ispair(rhs1, rhs2):
                return (lhs1 == lhs2) | ((lhs2 != rhs2) & (lhs1 != lhs2))
        else:
            assert 0  # unreachable
    return rel_expr


class LogicalAndAssumption:
    """Holds a list of assumptions associated with logical and-operation

    A logical and assumption is a logical and-expression of logical
    expressions.
    """

    def __init__(self):
        self.operands = []


def logical_binary_operation(logical_expr, kind, left=None, right=None):
    ctx = logical_expr.context
    op = getattr(ctx, kind)

    def f(expr):
        return expr.rewrite(rewrite)

    if logical_expr.kind in {"logical_and", "logical_or", "logical_xor", "logical_not"}:
        logical_op = getattr(ctx, logical_expr.kind)
        return logical_op(
            *[logical_binary_operation(operand, kind, left=left, right=right) for operand in logical_expr.operands]
        )
    elif (logical_expr.kind in {"eq", "ne"} and kind in {"multiply", "divide"}) or (
        logical_expr.kind in {"eq", "ne", "lt", "le", "gt", "ge"} and kind in {"add", "subtract"}
    ):
        logical_op = getattr(ctx, logical_expr.kind)
        if left is not None:
            if right is not None:
                return logical_op(*[f(op(op(left, operand), right)) for operand in logical_expr.operands])
            else:
                return logical_op(*[f(op(left, operand)) for operand in logical_expr.operands])
        elif right is not None:
            return logical_op(*[f(op(operand, right)) for operand in logical_expr.operands])
        else:
            return logical_expr
    elif logical_expr.kind in {"lt", "le", "gt", "ge"} and kind in {"multiply", "divide"}:
        logical_op1 = getattr(ctx, logical_expr.kind)
        logical_op2 = getattr(ctx, dict(lt="gt", le="ge", gt="lt", ge="le")[logical_expr.kind])
        if left is not None:
            if right is not None:
                r1 = logical_op1(*[f(op(op(left, operand), right)) for operand in logical_expr.operands]) & f(left * right > 0)
                r2 = logical_op2(*[f(op(op(left, operand), right)) for operand in logical_expr.operands]) & f(left * right < 0)
            else:
                r1 = logical_op1(*[f(op(left, operand)) for operand in logical_expr.operands]) & f(left > 0)
                r2 = logical_op2(*[f(op(left, operand)) for operand in logical_expr.operands]) & f(left < 0)
        elif right is not None:
            r1 = logical_op1(*[f(op(operand, right)) for operand in logical_expr.operands]) & f(right > 0)
            r2 = logical_op2(*[f(op(operand, right)) for operand in logical_expr.operands]) & f(right < 0)
        else:
            return logical_expr
        false = logical_expr.context.constant(False)
        true = logical_expr.context.constant(True)
        r1 = rewrite.op_rewrite(r1, commutative=True, idempotent=True, identity=true, absorber=false, kind="logical_and")
        r2 = rewrite.op_rewrite(r2, commutative=True, idempotent=True, identity=true, absorber=false, kind="logical_and")
        r = r1 | r2
        return rewrite.op_rewrite(r, commutative=True, idempotent=True, identity=false, absorber=true, kind="logical_or")
    else:
        assert 0, (kind, logical_expr.kind)  # unreachable


class AssumptionsGenerator:

    def __init__(self, data=None):
        self.data = dict() if data is None else data

    def __rewrite_modifier__(self, expr):
        if expr.kind in {"symbol"}:
            pass
        elif expr.kind == "constant":
            value, like = expr.operands
            if isinstance(value, str):
                if value in {"smallest_subnormal", "smallest", "eps", "pi", "largest"}:
                    assume(expr > 0, _data=self.data)
        elif expr.kind == "absolute":
            (x,) = expr.operands
            if x.get_type().is_real:
                assume(expr >= 0, _data=self.data)
        elif expr.kind in {"multiply", "divide", "add", "subtract"}:
            x, y = expr.operands
            if x.key in self.data:
                assume(logical_binary_operation(self.data[x.key], expr.kind, right=y), _data=self.data)
            if y.key in self.data:
                assume(logical_binary_operation(self.data[y.key], expr.kind, left=x), _data=self.data)
        elif expr.kind in {"logical_and", "logical_or", "logical_xor", "logical_not", "gt", "lt", "ge", "le", "eq", "ne"}:
            pass
        else:
            assert 0, expr.kind  # not implemeneted
        return expr


def assume(expr, _data=None):
    """Attach assumptions to symbols and expressions given by a logical
    expression that truth-value is assumed to be true.

    Calling assume multiple time for the same symbol or expression
    merges the assumptions using logical conjugate.
    """
    if expr._is_boolean:
        data = {} if _data is None else _data
        for and_operand in rewrite.op_flatten(
            rewrite.op_collect(
                expr, commutative=True, idempotent=True, over_commutative=True, over_idempotent=True, over_kind="logical_or"
            ),
            commutative=True,
            idempotent=True,
            kind="logical_and",
        ):
            collector = rewrite.Collector()
            and_operand.rewrite(collector)
            for expr_ in collector.data.values():
                props = expr_.props
                if "_assumptions" not in props:
                    props["_assumptions"] = {}
                if and_operand.key not in props["_assumptions"]:
                    props["_assumptions"][and_operand.key] = and_operand
                    data[expr_.key] = and_operand
        return list(data.values())
    else:
        modifier = AssumptionsGenerator(_data)
        expr.rewrite(modifier)
        return list(modifier.data.values())


def check(logical_expr):
    """Check if logical_expr defines a non-empty subset.

    Definition:

       logical_expr(x)   is a non-empty subset

    when there exists x such that logical_expr(x) is true, x is a set
    of all non-boolean symbols present in the logical expression.
    """
    true = logical_expr.context.constant(True)
    false = logical_expr.context.constant(False)

    logical_expr.rewrite(AssumptionsGenerator())

    collector = rewrite.Collector()
    logical_expr.rewrite(collector)
    assumptions = {}
    for expr in collector.data.values():
        assumptions.update(expr.props.get("_assumptions", {}))

    if assumptions:
        assumption = rewrite.op_unflatten(assumptions.values(), "logical_and")
        r = is_subset_of(logical_expr, assumption)
        if r is true:
            return True
        if r is false:
            return False
    return
