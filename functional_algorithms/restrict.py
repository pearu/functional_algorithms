"""
"""

from collections import defaultdict
from . import rewrite


def restrict(expr, domain=None):
    """Yield an expression and the corresponding logical expression such that when

       fp(<logical expression>) is true

    then

       fp(<input expression>) == fp(<expression>)

    holds. fp is a floating-point number evaluator of a floating-point system.

    All such yielded pairs `<expression>, <logical expression>`
    constitute the so-called restrictions set of the input expression.

    The expression is typically a simplified version of the input expression.

    For example, `restrict(log1p(x))` yields:

      x,        abs(x) < eps / 2
      log(x),   abs(x) > 4 / eps
      log1p(x), (abs(x) >= eps / 2) & (abs(x) <= 4 / eps)

    `restrict(log(1 + x))` yields:

      log(1),     abs(x) < eps / 2
      log(x),     abs(x) > 4 / eps
      log(1 + x), (abs(x) >= eps / 2) & (abs(x) <= 4 / eps)

    because `1 + x` yields

      1,         abs(x) < eps / 2
      x,         abs(x) > 4 / eps
      1 + x      (abs(x) >= eps / 2) & (abs(x) <= 4 / eps)

    Notice that mathematically equivalent but different expressions
    may have different restriction sets that elements may or may not
    be accurate for evaluating the input expression in the
    corresponding restriction region within the given floating-point
    system.
    """
    if domain is None:
        domain = expr.context.constant(True)

    unary_cases = dict(
        log1p=[log1p_with_small, log1p_with_large],
    )

    binary_cases = dict(
        add=[add_with_left_dominant, add_with_right_dominant],
    )

    kind_op = getattr(expr.context, expr.kind)

    expressions = dict()
    restrictions = defaultdict(list)
    if expr.kind == "constant":
        yield expr, domain
        return
    elif expr.kind == "symbol":
        yield expr, domain
        return
    elif expr.kind in binary_cases:
        x, y = expr.operands
        for x_, xd in restrict(x):
            for y_, yd in restrict(y):
                d = expr.context.constant(True)
                for case in binary_cases[expr.kind]:
                    e_, d_ = case(x_, y_)
                    d = d & (~d_)
                    expressions[e_.key] = e_
                    restrictions[e_.key].append(d_ & xd & yd)
                e = kind_op(x_, y_)
                expressions[e.key] = e
                restrictions[e.key].append(d & xd & yd)
    elif expr.kind in unary_cases:
        (x,) = expr.operands
        for x_, xd in restrict(x):
            d = expr.context.constant(True)
            for case in unary_cases[expr.kind]:
                e_, d_ = case(x_)
                d = d & (~d_)
                expressions[e_.key] = e_
                restrictions[e_.key].append(d_ & xd)
            e = kind_op(x_)
            expressions[e.key] = e
            restrictions[e.key].append(d & xd)
    else:
        raise NotImplementedError(expr.kind)

    for key, expression in expressions.items():
        restriction = rewrite.op_unflatten(restrictions[key], "logical_or")
        restriction = rewrite.op_rewrite(restriction, commutative=True, idempotent=True, kind="logical_or") & domain
        restriction = rewrite.op_collect(
            restriction, commutative=True, idempotent=True, over_commutative=True, over_idempotent=True, over_kind="logical_or"
        )
        restriction = restriction.rewrite(rewrite)
        assert restriction is not None
        yield expression, restriction


def log1p_with_small(x):
    eps = x.context.constant("eps")
    return x, abs(x) < eps / 2


def log1p_with_large(x):
    x_, r_ = add_with_right_dominant(x.context.constant(1), x)
    return x.context.log(x_), r_


def add_with_left_dominant(x, y):
    eps = x.context.constant("eps")
    return x, abs(x) * eps > 4 * abs(y)


def add_with_right_dominant(x, y):
    return add_with_left_dominant(y, x)
