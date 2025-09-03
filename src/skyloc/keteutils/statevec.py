import kete
from .frames import parse_frame

__all__ = ["make_kete_state"]


def make_kete_state(
    state_desig,
    jd_tdb,
    pos_au,
    vel_aupd,
    center_id=399,
    pos_frame=kete.Frames.Ecliptic,
    vel_frame=kete.Frames.Ecliptic,
):
    """Convenience function to create a `kete.State` object.
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
    return kete.State(
        desig=state_desig,
        jd=jd_tdb,
        pos=kete.Vector(pos_au, frame=parse_frame(pos_frame)),
        vel=kete.Vector(vel_aupd, frame=parse_frame(vel_frame)),
        center_id=center_id,
    )
