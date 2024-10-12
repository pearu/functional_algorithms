from functional_algorithms import targets, expr, rewrite, context
import pytest
import warnings


@pytest.fixture(scope="function", params=targets.__all__)
def target_name(request):
    return request.param


def test_target(target_name):
    target = getattr(targets, target_name)
    for kind in target.kind_to_target:
        assert kind in expr.known_expression_kinds, (target_name, kind)

    notimpl_lst = [kind for kind in expr.known_expression_kinds if kind not in target.kind_to_target]
    if notimpl_lst:
        msg = f"{target_name}.kind_to_target has unspecified keys: {', '.join(notimpl_lst)}"
        warnings.warn(msg)


def test_rewrite():
    for kind, value in rewrite.Rewriter.__dict__.items():
        if kind.startswith("_") or not callable(value):
            continue
        assert kind in expr.known_expression_kinds, (target_name, kind)

    notimpl_lst = [kind for kind in expr.known_expression_kinds if kind not in rewrite.Rewriter.__dict__]

    if notimpl_lst:
        msg = f"rewrite.Rewriter has unimplemented methods: {', '.join(notimpl_lst)}"
        warnings.warn(msg)


def test_context():
    for kind, value in context.Context.__dict__.items():
        if kind.startswith("_") or not callable(value):
            continue
        kind = value.__name__
        if kind in {"trace", "call"}:
            continue
        assert kind in expr.known_expression_kinds, (target_name, kind)

    notimpl_lst = [kind for kind in expr.known_expression_kinds if kind not in context.Context.__dict__]

    if notimpl_lst:
        msg = f"context.Context has unimplemented methods: {', '.join(notimpl_lst)}"
        warnings.warn(msg)
