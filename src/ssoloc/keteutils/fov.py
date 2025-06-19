import kete
import numpy as np
from .statevec import make_kete_state

__all__ = ["make_rect_fov", "make_cone_fov", "fov_state_check"]


def make_rect_fov(
    state_desig,
    jd_tdb,
    pos_au,
    vel_aupd,
    center_ra_deg,
    center_dec_deg,
    rotation_deg,
    lon_width_deg=0.1,
    lat_width_deg=0.1,
    pos_frame=kete.Frames.Ecliptic,
    vel_frame=kete.Frames.Ecliptic,
    center_id=399,
    fov_to_sun=True,
):
    """Make kete RectangleFOV

    Parameters
    ----------
    state_desig : str
        Name of the observer `State` used in the RectanlgeFOV object.

    jd_tdb : float or `kete.Time`
        The time of the state in TDB jd time, see `kete.Time`.

    pos_au, vel_aupd : array_like
        Position and velocity of the object in au and au/day, respectively,
        with respect to the center (`center_id`).

    center_ra_deg, center_dec_deg : float
        The center of the FOV in equatorial J2000 RA/DEC in degrees.

    rotation_deg : float
        The rotation of the field of view in degrees.

    lon_width_deg, lat_width_deg : float, optional
        The width of the FOV in degrees, by default 0.1 degrees in both
        longitude and latitude.

    pos_frame, vel_frame : `kete.Frames`, optional
        The frame of the position and velocity, by default
        `kete.Frames.Ecliptic`.

    center_id : int, optional
        The center of the FOV, by default 399 (Earth geocenter; NOT Earth-Moon
        barycenter (``3``)).

    fov_to_sun : bool, optional
        If True, the field of view is centered on the Sun, by default True.
        Note that kete's fov must be centered around the Sun to avoid any
        "surprises" when checking an object is in the FOV.


    Returns
    -------
    obssta : `kete.State`
        The state of the observer.

    fov : `kete.RectangleFOV`
        The field of view.
    """
    obssta = make_kete_state(
        state_desig=state_desig,
        jd_tdb=jd_tdb,
        pos_au=pos_au,
        vel_aupd=vel_aupd,
        pos_frame=pos_frame,
        vel_frame=vel_frame,
        center_id=center_id,
    )
    if fov_to_sun:
        obssta = obssta.change_center(10)

    fov = kete.RectangleFOV(
        pointing=kete.Vector.from_ra_dec(ra=center_ra_deg, dec=center_dec_deg),
        rotation=rotation_deg,
        observer=obssta,
        lon_width=lon_width_deg,
        lat_width=lat_width_deg,
    )
    return obssta, fov


def make_cone_fov(
    state_desig,
    jd_tdb,
    pos_au,
    vel_aupd,
    center_ra_deg,
    center_dec_deg,
    radius_deg=0.1,
    pos_frame=kete.Frames.Ecliptic,
    vel_frame=kete.Frames.Ecliptic,
    center_id=399,
):
    """Make kete ConeFOV

    Parameters
    ----------
    state_desig : str
        Name of the observer `State` used in the ConeFOV object.

    jd_tdb : float or `kete.Time`
        The time of the state in TDB jd time, see `kete.Time`.

    pos_au, vel_aupd : array_like
        Position and velocity of the object in au and au/day, respectively,
        with respect to the center (`center_id`).

    center_ra_deg, center_dec_deg : float
        The center of the FOV in equatorial J2000 RA/DEC in degrees.

    rotation_deg : float
        The rotation of the field of view in degrees.

    radius_deg : float, optional
        The radius of the FOV in degrees, by default 0.1 degrees.

    pos_frame, vel_frame : `kete.Frames`, optional
        The frame of the position and velocity, by default
        `kete.Frames.Ecliptic`.

    center_id : int, optional
        The center of the FOV, by default 399 (Earth geocenter; NOT Earth-Moon
        barycenter (``3``)).

    Returns
    -------
    obssta : `kete.State`
        The state of the observer.

    fov : `kete.ConeFOV`
        The field of view.
    """
    obssta = make_kete_state(
        state_desig=state_desig,
        jd_tdb=jd_tdb,
        pos_au=pos_au,
        vel_aupd=vel_aupd,
        pos_frame=pos_frame,
        vel_frame=vel_frame,
        center_id=center_id,
    )

    fov = kete.ConeFOV(
        pointing=kete.Vector.from_ra_dec(ra=center_ra_deg, dec=center_dec_deg),
        angle=radius_deg,
        observer=obssta,
    )

    return obssta, fov


def fov_state_check(
    obj_state,
    fovs,
    dt_limit=3.0,
    include_asteroids=False,
    return_desigs_map=False,
    return_uniq_objids=False,
):
    """Convenience tool around `kete.fov_state_check`.

    Parameters
    ----------
    states : list[State]
        States which do not already have a specified FOV.

    fov : FOVList, iterable of `kete.FOV`
        A field of view from which to subselect objects which are visible.

    dt_limit : float
        Length of time in days where 2-body mechanics is a good approximation.
        Default is 3.0 [days].

    include_asteroids : bool
        Include the 5 largest asteroids during the computation.
        Default is `False`. If `True`, the function will include the 5 largest
        asteroids (1 Ceres, 2 Pallas, 4 Vesta, 10 Hygiea, and 704 Interamnia).

    return_desigs_map : bool
        If `True`, return a mapping of FOV designations to object designations.
        The keys are the FOV designations, and the values are lists of object
        designations that were visible in the corresponding FOV. If `False`,
        this mapping is not returned.

    return_uniq_objids : bool
        If `True`, return a unique list of object designations that were visible
        in the FOVs. This is useful if you want to know which objects were
        visible in any of the FOVs without needing the mapping. If `False`,
        `uniq_objids` is `None`.
        Returned only if `return_desigs_map` is `True`.
        Default is `False`.

    Returns
    -------
    res : list[kete.SimultaneousState]
        A list of `kete.SimultaneousState` objects, each containing the FOV and
        the states of the objects that are visible to the observer at the time
        of the FOV.

    fov_obj_map : dict[str, list[str]] or None
        If `return_desigs_map` is `True`, a mapping of FOV designations to
        object designations. The keys are the FOV designations
        (``list(fov_obj_map.keys())``), and the values are lists of object
        designations that were visible in the corresponding FOV. (FOVs without
        any objects are not returned by `kete.fov_state_check`.)
        Returned only if `return_desigs_map` is `True`.

    uniq_objids : list[str] or None
        If `return_desigs_map` is `True` and `return_uniq_objids` is `True`,
        a unique list of object designations that were visible in any of the
        FOVs. If `return_desigs_map` is `False`, this is `None`.
        Returned only if `return_desigs_map` is `True`.
    """
    # Convenience: convert to FOVList if needed
    if isinstance(fovs, (kete.RectangleFOV, kete.ConeFOV, kete.OmniDirectionalFOV)):
        fovs = kete.fov.FOVList((fovs,))
    elif not isinstance(fovs, kete.fov.FOVList):
        try:
            fovs = kete.fov.FOVList(tuple(fovs))
        except Exception:
            fovs = kete.fov.FOVList(list(fovs))

    res = kete.fov.fov_state_check(
        obj_state,
        fovs,
        dt_limit=dt_limit,
        include_asteroids=include_asteroids,
    )

    if not return_desigs_map:
        return res

    # Convenience: Collect the designations of the FOVs and objects
    fov_obj_map = {}

    for _r in res:
        _objids = []
        for _s in _r.states:
            _objids.append(_s.desig)
        fov_obj_map.setdefault(_r.fov.observer.desig, []).extend(_objids)

    uniq_objids = None
    if return_uniq_objids:
        # ~ 10x faster than np.unique(np.concatenate(list(fov_obj_map.values())))
        uniq_objids = set()
        for objids in fov_obj_map.values():
            uniq_objids.update(objids)
        uniq_objids = list(uniq_objids)

    return res, fov_obj_map, uniq_objids
