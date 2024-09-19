__all__ = ["Constext", "Expr", "targets", "TextImage"]

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
from . import algorithms
from . import targets
from .textimage import TextImage
from . import rewrite
from .signatures import filter_signatures
