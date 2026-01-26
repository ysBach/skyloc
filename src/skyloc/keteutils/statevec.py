from ._kete_import import kete, require_kete
from ._util import parse_frame

__all__ = ["make_kete_state"]


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

    pos_au, vel_aupd : array_like
        Position and velocity of the object in au and au/day, respectively,
        with respect to the center (`center_id`).

    pos_frame, vel_frame : `kete.Frames`, str, optional
        The frame of the position and velocity, by default
        `kete.Frames.Ecliptic`.
    """
    if pos_frame is None:
        pos_frame = kete.Frames.Ecliptic
    if vel_frame is None:
        vel_frame = kete.Frames.Ecliptic
    return kete.State(
        desig=state_desig,
        jd=jd_tdb,
        pos=kete.Vector(pos_au, frame=parse_frame(pos_frame)),
        vel=kete.Vector(vel_aupd, frame=parse_frame(vel_frame)),
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

    pos_au, vel_aupd : array_like
        Position and velocity of the object in au and au/day, respectively,
        with respect to the center (`center_id`).
        If `jd_tdb` is array-like, these should also be array-like with
        the same length as `jd_tdb` along the first axis (i.e.,
        `pos_au.shape[0]` must match `jd_tdb.shape[0]`).

    center_id : int, array-like, optional
        The NAIF ID of the center of the state. Default is `399` (Earth).
        If array-like, it should have the same length as `jd_tdb`.

    pos_frame, vel_frame : `kete.Frames`, array-like of such, optional
        The frame of the position and velocity, by default
        `kete.Frames.Ecliptic`.
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
        return kete.State(
            desig=state_desig,
            jd=jd_tdb,
            pos=kete.Vector(pos_au, frame=parse_frame(pos_frame)),
            vel=kete.Vector(vel_aupd, frame=parse_frame(vel_frame)),
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
        states.append(
            kete.State(
                desig=state_desig,
                jd=_jd,
                pos=kete.Vector(_pos, frame=_pf),
                vel=kete.Vector(_vel, frame=_vf),
                center_id=_cen,
            )
        )
    return states
