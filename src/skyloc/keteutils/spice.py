"""SPICE kernel query helpers using kete's built-in kernel manager.

Wraps ``kete.spice.get_state`` for batch retrieval and provides
convenience functions to check whether a given SPK ID is loaded.
"""

import numpy as np

from ._kete_import import kete, require_kete

from ._util import get_kete_loaded_objects, parse_frame

__all__ = ["is_spk_loaded", "get_states", "get_state_arrays"]


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
    if idstr is None:
        return idstr

    if idstr.startswith("500@"):
        idstr = idstr[4:]

    elif idstr.startswith("@"):
        idstr = idstr[1:]

    # Get lazy-loaded kete objects
    cache = get_kete_loaded_objects()
    spks = cache["spks"]
    spks_name = cache["spks_name"]
    name2id = cache["name2id"]
    id2name = cache["id2name"]

    # Is this ID/NAME in the loaded sets?
    is_loaded = idstr in spks or idstr in spks_name

    if convert_to is None:
        return is_loaded, idstr

    # Optionally convert
    convert_to = convert_to.lower()
    if convert_to == "id":
        return is_loaded, name2id.get(idstr, idstr)
    if convert_to == "name":
        return is_loaded, id2name.get(idstr, idstr)

    raise ValueError(f"Unknown conversion type: {convert_to}")


def get_states(target, jds, center, frame=None):
    """Get SPICE states for a target at multiple JD times.

    **Requires kete**: This function needs kete to be installed.
    Install with: ``pip install skyloc[kete]``

    Parameters
    ----------
    target : str
        The name of the target object. This can include any object name listed in
        :meth:`~kete.spice.loaded_objects`.

    jds : float or array-like of float
        Julian time(s) (TDB) of the desired record(s).

    center : str
        The center point for the state vector (e.g., "sun" for heliocentric).

    frame : str or `kete.Frames`, optional
        Coordinate frame of the state. Default is `kete.Frames.Ecliptic`.

    Returns
    -------
    states : list of `kete.State`
        List of State objects for each requested JD.

    Notes
    -----
    This function wraps `kete.spice.get_state()` for batch retrieval.
    All times must be within the coverage of the loaded SPICE kernels.
    """
    require_kete()
    if frame is None:
        frame = kete.Frames.Ecliptic
    frame = parse_frame(frame)
    jds = np.atleast_1d(jds)

    states = []
    for jd in jds:
        state = kete.spice.get_state(target, jd, center, frame)
        states.append(state)

    return states


def get_state_arrays(target, jds, center="sun", frame=None):
    """Get SPICE states as numpy arrays for a target at multiple JD times.

    **Requires kete**: This function needs kete to be installed.
    Install with: ``pip install skyloc[kete]``

    Parameters
    ----------
    target : str
        The name of the target object. This can include any object name listed in
        :meth:`~kete.spice.loaded_objects`.

    jds : float or array-like of float
        Julian time(s) (TDB) of the desired record(s).

    center : str, optional
        The center point for the state vector. Default is "sun" (heliocentric).

    frame : str or `kete.Frames`, optional
        Coordinate frame of the state. Default is `kete.Frames.Ecliptic`.

    Returns
    -------
    pos : np.ndarray
        Position vectors in AU, shape (N, 3) where N is len(jds).

    vel : np.ndarray
        Velocity vectors in AU/day, shape (N, 3) where N is len(jds).

    Notes
    -----
    This is a convenience function for numerical operations.
    """
    require_kete()
    if frame is None:
        frame = kete.Frames.Ecliptic
    states = get_states(target, jds, center, frame)

    pos = np.array([[s.pos[0], s.pos[1], s.pos[2]] for s in states])
    vel = np.array([[s.vel[0], s.vel[1], s.vel[2]] for s in states])

    return pos, vel