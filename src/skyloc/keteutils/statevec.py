"""Tools related to ``State`` and ``Vector`` objects from kete.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

from ._kete_import import kete, require_kete
from ._util import parse_frame

__all__ = ["make_kete_state", "ecl2eq", "eq2ecl", "eq2gal", "gal2eq"]

# ---------------------------------------------------------------------------
# Frame rotation constants (ecliptic <-> equatorial J2000)
# ---------------------------------------------------------------------------

# IAU 1976 mean obliquity of the ecliptic at J2000.0 (degrees).
# Numerically identical to what kete uses for Vector.as_equatorial.
_ECL_OBLIQUITY_DEG = 23.4392911
_COS_EPS = np.cos(np.deg2rad(_ECL_OBLIQUITY_DEG))
_SIN_EPS = np.sin(np.deg2rad(_ECL_OBLIQUITY_DEG))

# Ecliptic J2000 -> equatorial J2000: rotation around x-axis by +eps
_R_ECL2EQ = np.array(
    [[1.0, 0.0,      0.0     ],
     [0.0, _COS_EPS, -_SIN_EPS],
     [0.0, _SIN_EPS,  _COS_EPS]]
)
# Equatorial J2000 -> ecliptic J2000: transpose (pure rotation, so R^-1 = R^T)
_R_EQ2ECL = _R_ECL2EQ.T

# Equatorial J2000 -> galactic (IAU 1958 / Hipparcos J2000 matrix)
_R_EQ2GAL = np.array(
    [[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
     [ 0.4941094278755837, -0.4448296299600112,  0.7469822444972189],
     [-0.8676661490190047, -0.1980763734312015,  0.4559837761750669]]
)


def ecl2eq(xyz):
    """Rotate ecliptic J2000 Cartesian coordinates to equatorial J2000.

    Applies the IAU 1976 mean obliquity rotation (23.4392911 deg around the
    x-axis). Numerically identical to kete's ``Vector.as_equatorial``, but
    operates on plain numpy arrays: no per-element kete object construction.

    Parameters
    ----------
    xyz : array_like, shape (..., 3)
        Cartesian coordinates in the ecliptic J2000 frame (any units).

    Returns
    -------
    np.ndarray, shape (..., 3)
        Coordinates in the equatorial J2000 frame, same units as input.

    Examples
    --------
    Single vector::

        ecl2eq([1.0, 0.5, 0.3])

    Batch of N vectors::

        ecl2eq(np.random.randn(1000, 3))
    """
    xyz = np.asarray(xyz, dtype=float)
    return xyz @ _R_ECL2EQ.T


def eq2ecl(xyz):
    """Rotate equatorial J2000 Cartesian coordinates to ecliptic J2000.

    Inverse of `ecl2eq`. Numerically identical to kete's
    ``Vector.as_ecliptic``, but operates on plain numpy arrays: no per-element
    kete object construction.

    Parameters
    ----------
    xyz : array_like, shape (..., 3)
        Cartesian coordinates in the equatorial J2000 frame (any units).

    Returns
    -------
    np.ndarray, shape (..., 3)
        Coordinates in the ecliptic J2000 frame, same units as input.

    Examples
    --------
    Round-trip check::

        v = np.array([1.0, 0.5, 0.3])
        assert np.allclose(v, eq2ecl(ecl2eq(v)))
    """
    xyz = np.asarray(xyz, dtype=float)
    return xyz @ _R_EQ2ECL.T


def eq2gal(xyz):
    """Rotate equatorial J2000 Cartesian coordinates to galactic.

    Applies the IAU 1958 / Hipparcos J2000 rotation. Operates on plain numpy
    arrays. Galactic longitude wrapping to [0, 360) is left to the caller;
    this function only rotates the Cartesian vector.

    Parameters
    ----------
    xyz : array_like, shape (..., 3)
        Cartesian coordinates in the equatorial J2000 frame (any units).

    Returns
    -------
    np.ndarray, shape (..., 3)
        Coordinates in the galactic frame, same units as input.
    """
    xyz = np.asarray(xyz, dtype=float)
    return xyz @ _R_EQ2GAL.T


def gal2eq(xyz):
    """Rotate galactic Cartesian coordinates to equatorial J2000.

    Inverse of ``eq2gal``.

    Parameters
    ----------
    xyz : array_like, shape (..., 3)
        Cartesian coordinates in the galactic frame (any units).

    Returns
    -------
    np.ndarray, shape (..., 3)
        Coordinates in the equatorial J2000 frame, same units as input.
    """
    xyz = np.asarray(xyz, dtype=float)
    return xyz @ _R_EQ2GAL


def _state2list(states):
    """Normalize a single state or an iterable of states to a list."""
    if isinstance(states, (list, tuple)):
        return list(states)
    if isinstance(states, np.ndarray):
        return list(states.tolist())
    return [states]


@require_kete
def _resolve_observer_pos(observer):
    """Resolve observer into Cartesian xyz array and kete.Vector.

    Parameters
    ----------
    observer : object
        One of kete.State, kete.Vector, FOVCollection, kete FOV object,
        or array_like with shape (3,).

    Returns
    -------
    xyz : np.ndarray
        Observer position in heliocentric ecliptic J2000 coordinates (AU),
        shape (3,).
    vec : kete.Vector
        Observer position as a kete.Vector.
    """
    require_kete()
    from .fov import FOVCollection
    
    if isinstance(observer, kete.State):
        xyz = np.array([observer.pos[0], observer.pos[1], observer.pos[2]], dtype=float)
        return xyz, observer.pos

    if isinstance(observer, kete.Vector):
        xyz = np.array([observer[0], observer[1], observer[2]], dtype=float)
        return xyz, observer

    if hasattr(observer, "observer") and hasattr(observer.observer, "pos"):
        obspos = observer.observer.pos
        xyz = np.array([obspos[0], obspos[1], obspos[2]], dtype=float)
        return xyz, obspos

    if isinstance(observer, FOVCollection):
        if len(observer) == 0:
            raise ValueError("FOVCollection is empty; cannot resolve observer position.")
        if len(observer) > 1:
            logger.warning(
                "FOVCollection has %d FOVs; using observer from the first FOV.",
                len(observer),
            )
        obspos = observer[0].observer.pos
        xyz = np.array([obspos[0], obspos[1], obspos[2]], dtype=float)
        return xyz, obspos

    xyz = np.asarray(observer, dtype=float).reshape(-1)
    if xyz.shape != (3,):
        raise ValueError(
            "observer must be kete.State, kete.Vector, FOV/FOVCollection, "
            "or array_like with shape (3,)."
        )
    return xyz, kete.Vector(xyz)


@require_kete
def make_kete_state(
    state_desig,
    jd_tdb,
    pos_au,
    vel_aupd,
    center_id=399,
    pos_frame=None,
    vel_frame=None,
):
    """Convenience function to create a `kete.State` object.

    **Requires kete**: This function needs kete to be installed.
    Install with: ``pip install skyloc[kete]``

    Parameters
    ----------
    state_desig : str
        Name of the observer `State` used in the RectanlgeFOV object.

    jd_tdb : float or `kete.Time`
        The time of the state in TDB jd time, see `kete.Time`.

    pos_au, vel_aupd : array_like or `kete.Vector`
        Position and velocity of the object in au and au/day, respectively,
        with respect to the center (`center_id`).
        Can be either numeric arrays or existing `kete.Vector` objects.
        If `kete.Vector` objects are provided, their existing frames are used
        and `pos_frame`/`vel_frame` parameters are ignored for those inputs.

    pos_frame, vel_frame : `kete.Frames`, str, or None, optional
        The frame of the position and velocity. Default is None, which uses
        `kete.Frames.Ecliptic`. Only used when `pos_au`/`vel_aupd` are numeric
        arrays (not `kete.Vector` objects).
    """
    if isinstance(pos_au, kete.Vector):
        pos = pos_au
    else:
        if pos_frame is None:
            pos_frame = kete.Frames.Ecliptic
        pos = kete.Vector(pos_au, frame=parse_frame(pos_frame))

    if isinstance(vel_aupd, kete.Vector):
        vel = vel_aupd
    else:
        if vel_frame is None:
            vel_frame = kete.Frames.Ecliptic
        vel = kete.Vector(vel_aupd, frame=parse_frame(vel_frame))

    return kete.State(
        desig=state_desig,
        jd=jd_tdb,
        pos=pos,
        vel=vel,
        center_id=center_id,
    )


# FIXME: Under Development
@require_kete
def make_kete_states(
    state_desig,
    jd_tdb,
    pos_au,
    vel_aupd,
    center_id=399,
    pos_frame=None,
    vel_frame=None,
):
    """Convenience function to create a `kete.State` object(s).

    **Requires kete**: This function needs kete to be installed.
    Install with: ``pip install skyloc[kete]``

    Parameters
    ----------
    state_desig : str, array_like
        Name of the observer `State` used in the RectanlgeFOV object.
        If `jd_tdb` is array-like, this should be a string of the same length
        as `jd_tdb`.

    jd_tdb : float or `kete.Time` or array_like
        The time of the state in TDB jd time, see `kete.Time`.
        If this is array-like, this function will create a list of `kete.State`
        objects.

    pos_au, vel_aupd : array_like or `kete.Vector`
        Position and velocity of the object in au and au/day, respectively,
        with respect to the center (`center_id`).
        Can be either numeric arrays or existing `kete.Vector` objects.
        If `kete.Vector` objects are provided, their existing frames are used
        and `pos_frame`/`vel_frame` parameters are ignored for those inputs.
        If `jd_tdb` is array-like, these should also be array-like with
        the same length as `jd_tdb` along the first axis (i.e.,
        `pos_au.shape[0]` must match `jd_tdb.shape[0]`).

    center_id : int, array-like, optional
        The NAIF ID of the center of the state. Default is `399` (Earth).
        If array-like, it should have the same length as `jd_tdb`.

    pos_frame, vel_frame : `kete.Frames`, str, None, or array-like of such, optional
        The frame of the position and velocity. Default is None, which uses
        `kete.Frames.Ecliptic`. Only used when `pos_au`/`vel_aupd` are numeric
        arrays (not `kete.Vector` objects).
        If array-like, it should have the same length as `jd_tdb`.

    """
    require_kete()
    import numpy as np
    from ..utils import as_iter

    if pos_frame is None:
        pos_frame = kete.Frames.Ecliptic
    if vel_frame is None:
        vel_frame = kete.Frames.Ecliptic

    if np.size(jd_tdb) == 1:
        if isinstance(pos_au, kete.Vector):
            pos = pos_au
        else:
            pos = kete.Vector(pos_au, frame=parse_frame(pos_frame))

        if isinstance(vel_aupd, kete.Vector):
            vel = vel_aupd
        else:
            vel = kete.Vector(vel_aupd, frame=parse_frame(vel_frame))

        return kete.State(
            desig=state_desig,
            jd=jd_tdb,
            pos=pos,
            vel=vel,
            center_id=center_id,
        )

    pos_au = np.atleast_2d(pos_au)
    vel_aupd = np.atleast_2d(vel_aupd)
    nsize = jd_tdb.shape[0]
    state_desig = as_iter(state_desig, nsize)  # ensure it's iterable

    # if state_desig:

    if pos_au.shape[0] != nsize:
        raise ValueError(
            f"`pos_au` and `jd_tdb` must have the same length along the first axis. "
            f"Got {pos_au.shape[0] = } and {jd_tdb.shape[0] = }."
        )
    if vel_aupd.shape[0] != nsize:
        raise ValueError(
            f"`vel_aupd` and `jd_tdb` must have the same length along the first axis. "
            f"Got {vel_aupd.shape[0] = } and {jd_tdb.shape[0] = }."
        )

    if np.size(center_id) != 1 and np.size(center_id) != nsize:
        raise ValueError(
            f"`center_id` must be a scalar or have the same length as `jd_tdb`. "
            f"Got {np.size(center_id) = } and {jd_tdb.shape[0] = }."
        )

    if np.size(pos_frame) != 1 and np.size(pos_frame) != nsize:
        raise ValueError(
            f"`pos_frame` must be a scalar or have the same length as `jd_tdb`. "
            f"Got {np.size(pos_frame) = } and {jd_tdb.shape[0] = }."
        )

    if np.size(vel_frame) != 1 and np.size(vel_frame) != nsize:
        raise ValueError(
            f"`vel_frame` must be a scalar or have the same length as `jd_tdb`. "
            f"Got {np.size(vel_frame) = } and {jd_tdb.shape[0] = }."
        )

    center_id = as_iter(center_id, nsize)
    pos_frame = as_iter(pos_frame, nsize)
    vel_frame = as_iter(vel_frame, nsize)
    states = []

    for _jd, _pos, _vel, _cen, _pf, _vf in zip(
        jd_tdb, pos_au, vel_aupd, center_id, pos_frame, vel_frame
    ):
        if isinstance(_pos, kete.Vector):
            pos = _pos
        else:
            pos = kete.Vector(_pos, frame=_pf)

        if isinstance(_vel, kete.Vector):
            vel = _vel
        else:
            vel = kete.Vector(_vel, frame=_vf)

        states.append(
            kete.State(
                desig=state_desig,
                jd=_jd,
                pos=pos,
                vel=vel,
                center_id=_cen,
            )
        )
    return states
