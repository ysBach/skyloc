"""Logging utilities for skyloc package.

Usage
-----
In each module:
    import logging
    logger = logging.getLogger(__name__)

To control logging level:
    import skyloc
    skyloc.set_log_level("INFO")   # or "DEBUG", logging.WARNING, etc.
"""
import logging
import sys

__all__ = ["set_log_level"]

_PKG_LOGGER_NAME = "skyloc"


def set_log_level(level, handler_level=None):
    """Set the logging level for the skyloc package.

    Parameters
    ----------
    level : int or str
        Logging level (e.g., logging.DEBUG, "INFO", logging.WARNING).
    handler_level : int or str, optional
        Separate level for the handler. If None, uses `level`.

    Examples
    --------
    >>> import skyloc
    >>> skyloc.set_log_level("INFO")   # Enable info messages
    >>> skyloc.set_log_level("DEBUG")  # Verbose debugging
    >>> skyloc.set_log_level(logging.WARNING)  # Back to default (quiet)

    Granular control available:

    >>> import logging
    >>> import skyloc
    >>> skyloc.set_log_level("WARNING")  # All quiet
    >>> logging.getLogger("skyloc.query").setLevel(logging.DEBUG)  # Only query verbose
    """
    logger = logging.getLogger(_PKG_LOGGER_NAME)
    logger.setLevel(level)
    if handler_level is not None:
        for h in logger.handlers:
            h.setLevel(handler_level)


def _setup_logger():
    """One-time setup of the package root logger with default handler."""
    logger = logging.getLogger(_PKG_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)  # Handler passes all; logger filters
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.WARNING)  # Default: quiet
    return logger


_setup_logger()
