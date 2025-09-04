"""
Simple constants and utility functions that has extremely simple dependence on
kete, and will likely be used throughout many modules.
"""

import kete

__all__ = [
    "KETE_LOADED_SPKS",
    "KETE_LOADED_SPKS_NAME",
    "KETE_LOADED_SPKS_NAME2ID",
    "KETE_LOADED_SPKS_ID2NAME",
    "KETE_LOADED_ASTEROIDS",
    "KETE_LOADED_ASTEROIDS_NAME",
    "parse_frame"
]


# Largest asteroids that are loaded in kete for n-body propagation.
KETE_LOADED_SPKS = []
KETE_LOADED_SPKS_NAME = []
KETE_LOADED_SPKS_NAME2ID = {}
KETE_LOADED_SPKS_ID2NAME = {}
KETE_LOADED_ASTEROIDS = []
KETE_LOADED_ASTEROIDS_NAME = []

for name, spkid in kete.spice.loaded_objects():
    if spkid < 0:
        continue
    elif spkid > 20_000_000:
        spkid = str(spkid - 20_000_000)
        KETE_LOADED_ASTEROIDS.append(spkid)
        KETE_LOADED_ASTEROIDS_NAME.append(name)
        KETE_LOADED_SPKS_ID2NAME[spkid] = name
    KETE_LOADED_SPKS_NAME2ID[name] = str(spkid)
    KETE_LOADED_SPKS_ID2NAME[str(spkid)] = name
    KETE_LOADED_SPKS.append(str(spkid))
    KETE_LOADED_SPKS_NAME.append(name)


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
