import kete

__all__ = ["make_rect_fov"]


def make_rect_fov(
    state_desig,
    jd_tdb,
    pos_au,
    vel_aupd,
    center_ra_deg,
    center_dec_deg,
    rotation_deg,
    pos_frame=kete.Frames.Ecliptic,
    vel_frame=kete.Frames.Ecliptic,
    center_id=399,
    fov_to_sun=True,
    lon_width_deg=14,
    lat_width_deg=4,
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

    lon_width_deg, lat_width_deg : float, optional
        The width of the FOV in degrees, by default 14 and 4 degrees, respectively.

    Returns
    -------
    obssta : `kete.State`
        The state of the observer.

    fov : `kete.RectangleFOV`
        The field of view.
    """
    obssta = kete.State(
        desig=state_desig,
        jd=jd_tdb,
        pos=kete.Vector(pos_au, frame=pos_frame),
        vel=kete.Vector(vel_aupd, frame=vel_frame),
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
