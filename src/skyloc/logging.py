"""Logging utilities for skyloc package. (`help(skyloc.logging)`)

Enabling Log Output
-------------------
By default, skyloc logs are set to WARNING level (quiet). To see log messages:

    >>> import skyloc
    >>> skyloc.set_log_level("INFO")   # Basic progress info
    >>> skyloc.set_log_level("DEBUG")  # Verbose debugging

Saving Logs to File
-------------------
To also write logs to a file:

    >>> import logging
    >>> import skyloc
    >>>
    >>> file_handler = logging.FileHandler("skyloc.log")
    >>> file_handler.setFormatter(
    ...     logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    ... )
    >>> logging.getLogger("skyloc").addHandler(file_handler)
    >>> skyloc.set_log_level("INFO")
    >>>
    >>> # Now logs go to both terminal AND skyloc.log

Granular Control
----------------
Control individual module verbosity:

    >>> import logging
    >>> skyloc.set_log_level("WARNING")  # All quiet
    >>> logging.getLogger("skyloc.query").setLevel(logging.DEBUG)  # Only query verbose

Developer Note
--------------
In each module, use the standard pattern:

    import logging
    logger = logging.getLogger(__name__)
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


class _ISO8601Formatter(logging.Formatter):
    """Formatter with ISO8601 timestamp to centisecond (0.01s) precision."""

    def formatTime(self, record, datefmt=None):
        from datetime import datetime, timezone

        ct = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone()
        # ISO8601 with centiseconds: 2026-01-23T06:43:23.45+09:00
        return ct.strftime("%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs // 10):02d}" + ct.strftime("%z")


def _setup_logger():
    """One-time setup of the package root logger with default handler."""
    logger = logging.getLogger(_PKG_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)  # Handler passes all; logger filters
        formatter = _ISO8601Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.WARNING)  # Default: quiet
    return logger


_setup_logger()

