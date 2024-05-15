__all__ = ["Constext", "Expr", "targets"]

try:
    from ._version import __version__
except ImportError:
    import importlib.metadata

    try:
        __version__ = importlib.metadata.version("py" + __name__)
    except:
        __version__ = "N/A"


from .expr import Expr
from .context import Context
from . import targets
