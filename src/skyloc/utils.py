# Very simple utility functions
from itertools import compress, repeat

from astropy.time import Time
import numpy as np

from .ioutils import all_world2pix_infov, infov2d

__all__ = [
    "au2km",
    "km2au",
    "kmps2aupd",
    "aupd2kmps",
    "tdb2utc",
    "utc2tdb",
    "as_iter",
    "zip_iters",
    "listmask",
    "all_world2pix_infov",
    "infov2d",
]


def au2km(x):
    """Convert AU to km."""
    return x * 1.49597870700e8


def km2au(x):
    """Convert km to AU."""
    return x / 1.49597870700e8


def kmps2aupd(x):
    """Convert km/s to AU/day."""
    return x * 0.00057754833


def aupd2kmps(x):
    """Convert AU/day to km/s."""
    return x * 1731.4568


def tdb2utc(tdb, format="jd"):
    """Convert TDB to UTC."""
    return Time(tdb, format=format, scale="tdb").utc


def utc2tdb(utc, format="jd"):
    """Convert UTC to TDB."""
    return Time(utc, format=format, scale="utc").tdb


def listmask(inlist, mask):
    """numpy-like masking as `inlist[mask]`, but for lists."""
    if mask is None:
        return inlist
    return list(compress(inlist, mask))


def as_iter(x, n):
    """Return an iterator over x. If x is scalar, repeat it n times.
    Useful for making "dummy" iterators for broadcasting.
    """
    # treat numpy scalars, Python scalars as scalar; avoid treating str/bytes as iterable
    if np.ndim(x) == 0 or isinstance(x, (str, bytes)):
        return repeat(x, n)
    # already array-like / iterable
    return iter(x)


def zip_iters(*args):
    """Zip iterators, broadcasting them to the same length.
    This makes it easier to iterate over multiple arrays of different lengths

    Usage:

    >>> for vals in zip_iters(arg1, arg2, ...):
    >>>     # do something

    """
    n = len(next(arg for arg in args if np.ndim(arg) > 0))
    return zip(*(as_iter(x, n) for x in args))


