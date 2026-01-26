"""
Centralized lazy import handling for kete.

This module provides a single mechanism to import kete only when needed,
and raises informative errors if kete is not installed.
"""

import inspect
from functools import wraps

__all__ = ["kete", "KETE_AVAILABLE", "require_kete"]

KETE_AVAILABLE = False
kete = None

try:
    import kete as _kete

    kete = _kete
    KETE_AVAILABLE = True
except ImportError:
    KETE_AVAILABLE = False
    msg = (
        "kete is required but not installed."
        "\n\nInstall it with: pip install skyloc[kete]"
        "\nOr install from source: pip install git+https://github.com/Caltech-IPAC/kete.git"
        "\n\nIMPORTANT: This requires the Caltech IPAC version (v1.x) not 2+"
    )


def require_kete(func=None):
    """Decorator or function to ensure kete is available.

    Can be used as:
    1. Decorator: @require_kete
    2. Function call: require_kete() (auto-detects function name)

    Parameters
    ----------
    func : callable, optional
        Function to decorate. If None, acts as a function call and auto-detects
        the calling function name from the call stack.

    Examples
    --------
    As a decorator:

    .. code-block:: python

        @require_kete
        def my_function():
            return kete.State(...)

    As a function call (auto-detects name):

    .. code-block:: python

        def my_function():
            require_kete()  # Auto-detects "my_function"
            return kete.State(...)
    """

    if callable(func):
        if KETE_AVAILABLE:
            # Avoid extra call overhead when kete is installed.
            return func

        func_name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not KETE_AVAILABLE:
                raise ImportError(f"{func_name} requires kete, but it is not installed.\n\n{msg}")
            return func(*args, **kwargs)

        return wrapper

    if not KETE_AVAILABLE:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            func_name = frame.f_back.f_code.co_name
        else:
            func_name = None

        if func_name:
            raise ImportError(f"{func_name} requires kete, but it is not installed.\n\n{msg}")
        raise ImportError(msg)

    return None
