# Very simple utility functions
from itertools import compress

from astropy.time import Time
import numpy as np

__all__ = [
    "tdb2utc",
    "utc2tdb",
    "listmask",
    "all_world2pix_infov",
    "infov2d",
]


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


def all_world2pix_infov(wcs, *args, naxes=None, bezels=0.5, **kwargs):
    """
    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        The WCS object.

    naxes : array-like of int, optional
        The number of pixels along each axis. If not provided, it defaults
        first to `wcs._naxis`, and if it does not exist, the maximum values of
        `x` and `y`.

    bezels : int or 2-D array-like, optional
        The number of pixels to drop from each edge of the image. If an int,
        it is applied to all edges. If 2-D, it should be like ``[[left, right], [bottom, top]]``. Use negative values to accept objects slightly
        outside the image.

    """
    pixels = wcs.all_world2pix(*args, **kwargs)
    if naxes is None:
        naxes = getattr(wcs, "_naxis", None)
        for _n in np.atleast_1d(naxes):
            if _n < 1:
                raise ValueError(
                    "`wcs._naxis`, which should be the number of pixels along each axis, "
                    f"must be a positive integer or array-like. Got this: {naxes = }."
                    f" Either use proper WCS or give `naxes` explicitly."
                )
    infov = infov2d(pixels[0], pixels[1], bezels=bezels, naxes=naxes)
    return pixels, infov


def infov2d(x, y, bezels=0.5, naxes=None):
    """Create a 2-D boolean array indicating if the points are within the image
    defined by `naxes` and `bezels`.

    Parameters
    ----------
    x, y : array-like
        The x and y coordinates of the points.

    bezels : int or 2-D array-like, optional
        The number of pixels to drop from each edge of the image. If an int,
        it is applied to all edges. If 2-D, it should be like ``[[left, right], [bottom, top]]``. Use negative values to accept objects slightly
        outside the image.
        Default is 0.5 because python assigns values -0.5 to 0.5 to be the 0-th
        pixel.

    naxes : array-like of int, optional
        The number of pixels along each axis. If not provided, it defaults to
        the maximum values of `x` and `y`.

    Returns
    -------
    infov : 1-D boolean array
        A boolean array indicating if the points are within the image.
    """
    if naxes is None:
        naxes = [max(x), max(y)]
    elif isinstance(naxes, (int, float)):
        naxes = [naxes, naxes]
    elif len(naxes) != 2:
        raise ValueError(f"naxes must have 2 entries, got {len(naxes)=}.")

    if isinstance(bezels, (int, float)):
        bezels = [[bezels, bezels]] * 2

    infov = (
        (bezels[0][0] <= x)
        & (x <= naxes[0] - bezels[0][1])
        & (bezels[1][0] <= y)
        & (y <= naxes[1] - bezels[1][1])
    )

    return infov


# def all_world2pix(
#     wcs,
#     *args,
#     naxes=None,
#     bezels=0,
#     tolerance=1e-4,
#     maxiter=20,
#     adaptive=False,
#     detect_divergence=True,
#     quiet=False,
# ):
#     """world2pix and drop objects outside the image.

#     naxes : array-like of int, optional
#         The number of pixels along each axis. If not provided, it is
#         determined from the WCS's CRPIXn keywords (assuming CRPIX is at the
#         center of image).

#     bezels : 2-D array-like of int, optional
#         The number of pixels to drop from each edge of the image. If not
#         provided, it defaults to 0 (no bezels). If array-like, it should
#         be 2-D, e.g., ``[[left, right], [bottom, top]]``.


#     Image size is determined by the WCS object.

#     all other args : see `~astropy.wcs.WCS.all_world2pix`
#     """
#     naxis = wcs.wcs.naxis
#     if naxes is None:
#         # assume image is centered around CRPIX
#         naxes = [2*c for c in wcs.wcs.crpix[:naxis]]
#     elif isinstance(naxes, int):
#         naxes = [naxes]*naxis
#     elif len(naxes) != naxis:
#         raise ValueError(f"naxis must have {wcs.wcs.naxis=} entries, got {len(naxes)=}.")

#     if bezels is None:
#         bezels = [[0, 0]]*naxis
#     elif isinstance(bezels, int):
#         bezels = [[bezels, bezels]]*naxis
#     # make error catcher here....

#     pixels = wcs.all_world2pix(
#         *args,
#         bezels=bezels,
#         tolerance=tolerance,
#         maxiter=maxiter,
#         adaptive=adaptive,
#         detect_divergence=detect_divergence,
#         quiet=quiet,
#     )

#     infov = (
#         (bezels[0][0] <= pixels[0])
#         & (pixels[0] < naxes[0] - bezels[0][1])
#         & (bezels[1][0] <= pixels[1])
#         & (pixels[1] < naxes[1] - bezels[1][1])
#     )

#     return infov
