from . import algorithms as faa
from . import apmath


class definition(faa.definition):
    _registry = {}


@definition("square", domain="real")
def real_square(ctx, x: list[float, ...], functional: bool = True, size: int = None):
    """Square on real input: x * x"""
    return apmath.square(ctx, x, functional=functional, size=2)


@definition("square")
def square(ctx, z: list[float | complex, ...]):
    """Square on floating-point expansion"""
    assert 0
