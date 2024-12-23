__all__ = ["Context", "Expr", "targets", "TextImage", "algorithms", "targets", "rewrite", "filter_signatures", "restrict"]

try:
    from ._version import __version__
except ImportError:
    import importlib.metadata

    try:
        __version__ = importlib.metadata.version("py" + __name__)
    except Exception:
        __version__ = "N/A"


from .expr import Expr
from .context import Context
from . import algorithms
from . import targets
from .textimage import TextImage
from . import rewrite
from .restrict import restrict
from .signatures import filter_signatures
from . import fpu
