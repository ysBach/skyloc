"""skyloc — Solar System object locator and ephemeris toolkit.

Provides classes and utilities for locating Solar System objects in
astronomical images, querying orbital databases (JPL SBDB / Horizons),
propagating orbits via n-body integration (through the optional ``kete``
backend), and working with FITS header databases and WCS transformations.
"""

from .query import *
from .ssoflux import *

try:
    from .keteutils import *
except ImportError:
    pass
from .jplhorizons import *
from .jplsbdb import *
from .ioutils import *
from .utils import *

try:
    from .core import *
except ImportError:
    pass
from .configs import *
from .logging import set_log_level

from . import spicetools
from . import plotutils

try:
    from .keteutils._kete_import import KETE_AVAILABLE
except ImportError:
    KETE_AVAILABLE = False

__all__ = ["KETE_AVAILABLE"]
