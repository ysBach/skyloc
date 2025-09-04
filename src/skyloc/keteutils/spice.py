import kete

from ._util import (
    KETE_LOADED_SPKS,
    KETE_LOADED_SPKS_ID2NAME,
    KETE_LOADED_SPKS_NAME,
    KETE_LOADED_SPKS_NAME2ID,
    parse_frame,
)

__all__ = ["is_spk_loaded"]


def is_spk_loaded(idstr, convert_to=None):
    """Check if a SPK ID or NAME is loaded in KETE.

    Parameters
    ----------
    idstr : str
        The SPK ID or NAME to check. For convenience of Horizons usages,
        "500@<id>" is also considered `True` because in Horizons, it means the
        center of the SPKID.

    convert_to : `None`, {"id", "name"}, optional
        If `None` (default), return the original ID or NAME as is.
        To force convert them to int-like IDs (e.g., '399' instead of 'earth'),
        use "id". To force convert to NAME (e.g., 'earth' instead of '399'),
        use "name".

    Returns
    -------
    is_loaded : bool
        `True` if this SPK is loaded in kete.

    idstr : str
        The ID or NAME of the object. For "500@<id>" format, the first 4
        characters ("500@") will be stripped off.
    """
    # Handle Horizons-style "500@<id>"
    if idstr.startswith("500@"):
        idstr = idstr[5:]

    # Is this ID/NAME in the loaded sets?
    is_loaded = idstr in KETE_LOADED_SPKS or idstr in KETE_LOADED_SPKS_NAME

    # Optionally convert
    if convert_to is None:
        return is_loaded, idstr

    convert_to = convert_to.lower()
    if convert_to == "id":
        return is_loaded, KETE_LOADED_SPKS_NAME2ID.get(idstr, idstr)
    if convert_to == "name":
        return is_loaded, KETE_LOADED_SPKS_ID2NAME.get(idstr, idstr)

    raise ValueError(f"Unknown conversion type: {convert_to}")


def get_states(target, jds, center, frame=kete.Frames.Ecliptic):
    """Run multiple get_state for jd values
    Parameters
    ----------
    target : str
        The names of the target object, this can include any object name listed in
        :meth:`~kete.spice.loaded_objects`
    jds : float or list of float
        Julian time (TDB) of the desired record.
    center : str
        The center point, this defaults to being heliocentric.
    frame : str or `kete.Frame`
        Coordinate frame of the state, defaults to ecliptic.
    """
    frame = parse_frame(frame)
