"""
Simple constants and utility functions that has extremely simple dependence on
kete, and will likely be used throughout many modules.
"""

import kete

__all__ = [
    "get_kete_loaded_objects",
    "parse_frame",
]

# Cache for lazy-loaded kete SPICE objects
_KETE_CACHE = None


def get_kete_loaded_objects():
    """Get loaded kete SPICE objects with lazy initialization.

    Returns a dict with keys:
    - 'spks': list of SPK IDs as strings
    - 'spks_name': list of SPK names
    - 'name2id': dict mapping names to SPK IDs
    - 'id2name': dict mapping SPK IDs to names
    - 'asteroids': list of asteroid SPK IDs as strings
    - 'asteroids_name': list of asteroid names

    This function defers the kete.spice.loaded_objects() call until first use,
    avoiding network calls (and potential SSL errors) at import time.
    """
    global _KETE_CACHE

    if _KETE_CACHE is not None:
        return _KETE_CACHE

    spks = []
    spks_name = []
    name2id = {}
    id2name = {}
    asteroids = []
    asteroids_name = []

    for name, spkid in kete.spice.loaded_objects():
        if spkid < 0:
            continue
        elif spkid > 20_000_000:
            spkid_str = str(spkid - 20_000_000)
            asteroids.append(spkid_str)
            asteroids_name.append(name)
            id2name[spkid_str] = name
        name2id[name] = str(spkid)
        id2name[str(spkid)] = name
        spks.append(str(spkid))
        spks_name.append(name)

    _KETE_CACHE = {
        'spks': spks,
        'spks_name': spks_name,
        'name2id': name2id,
        'id2name': id2name,
        'asteroids': asteroids,
        'asteroids_name': asteroids_name,
    }
    return _KETE_CACHE


def parse_frame(frame):
    """Parse a frame input into a `kete.Frame` object.

    Parameters
    ----------
    frame : `~kete.Frames` or str
        The frame to parse. If str, currently the followings are available:
        {'Ecliptic', 'Equatorial', 'FK4', 'Galactic'}.

    """
    if isinstance(frame, kete.Frames):
        return frame
    if isinstance(frame, str):
        return getattr(kete.Frames, frame)
    if frame is None:
        return None
    raise ValueError(f"Unknown frame type: {frame}")
