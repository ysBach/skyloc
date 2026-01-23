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


# Conversion constants
# IAU 2012 exact definition: 1 AU = 149597870.700 km
AU_KM_IAU2012 = 1.49597870700e8  # km
# SPICE/NAIF DE-405 era value: 1 AU = 149597870.691 km
AU_KM_SPICE = 1.49597870691e8  # km
# Seconds per day (exact)
DAY_S = 86400.0


def au2km(x, use_spice=False):
    """Convert AU to km.

    Parameters
    ----------
    x : float or array-like
        Distance in AU.
    use_spice : bool, optional
        If True, use SPICE/NAIF DE-405 AU value (149597870.691 km).
        If False (default), use IAU 2012 exact value (149597870.700 km).

    Returns
    -------
    float or array-like
        Distance in km.
    """
    au_km = AU_KM_SPICE if use_spice else AU_KM_IAU2012
    return x * au_km


def km2au(x, use_spice=False):
    """Convert km to AU.

    Parameters
    ----------
    x : float or array-like
        Distance in km.
    use_spice : bool, optional
        If True, use SPICE/NAIF DE-405 AU value (149597870.691 km).
        If False (default), use IAU 2012 exact value (149597870.700 km).

    Returns
    -------
    float or array-like
        Distance in AU.
    """
    au_km = AU_KM_SPICE if use_spice else AU_KM_IAU2012
    return x / au_km


def kmps2aupd(x, use_spice=False):
    """Convert km/s to AU/day.

    Parameters
    ----------
    x : float or array-like
        Velocity in km/s.
    use_spice : bool, optional
        If True, use SPICE/NAIF DE-405 AU value (149597870.691 km).
        If False (default), use IAU 2012 exact value (149597870.700 km).

    Returns
    -------
    float or array-like
        Velocity in AU/day.
    """
    au_km = AU_KM_SPICE if use_spice else AU_KM_IAU2012
    return x * DAY_S / au_km


def aupd2kmps(x, use_spice=False):
    """Convert AU/day to km/s.

    Parameters
    ----------
    x : float or array-like
        Velocity in AU/day.
    use_spice : bool, optional
        If True, use SPICE/NAIF DE-405 AU value (149597870.691 km).
        If False (default), use IAU 2012 exact value (149597870.700 km).

    Returns
    -------
    float or array-like
        Velocity in km/s.
    """
    au_km = AU_KM_SPICE if use_spice else AU_KM_IAU2012
    return x * au_km / DAY_S


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


