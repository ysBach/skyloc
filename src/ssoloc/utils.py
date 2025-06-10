# Very simple utility functions
from astropy.time import Time


__all__ = [
    "tdb2utc",
    "utc2tdb",
]


def tdb2utc(tdb, format="jd"):
    """Convert TDB to UTC."""
    return Time(tdb, format=format, scale="tdb").utc


def utc2tdb(utc, format="jd"):
    """Convert UTC to TDB."""
    return Time(utc, format=format, scale="utc").tdb
