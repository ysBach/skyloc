from ._constants import KETE_ASTEROIDS_PHYSICS, KETE_PLANETS
from .fov import *
from .propagate import *
from .stderr import *
from .spice import *
from ._util import *
from .statevec import *


# Default targets for SpiceLocator
DEFAULT_SPICE_PLANETS = [
    "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"
]


def get_spice_asteroid_names():
    """Get list of asteroid names loadable via SPICE in kete.

    Returns
    -------
    list of str
        Names of the 5 large asteroids that kete uses as perturbers.
    """
    return [v["name"] for v in KETE_ASTEROIDS_PHYSICS.values()]


def get_spice_asteroid_desigs():
    """Get list of asteroid designations (number strings) loadable via SPICE.

    Returns
    -------
    list of str
        Designations like "1", "2", "4", "10", "704".
    """
    return list(KETE_ASTEROIDS_PHYSICS.keys())


def get_default_spice_targets(include_planets=True, include_asteroids=True):
    """Get default list of SPICE targets for locator.

    Parameters
    ----------
    include_planets : bool, optional
        Include planets in the target list. Default is True.

    include_asteroids : bool, optional
        Include large asteroids in the target list. Default is True.

    Returns
    -------
    list of str
        List of targets suitable for kete.fov.fov_spice_check().
        Planets use SPICE IDs (e.g., "199"); asteroids use names (e.g., "ceres").
    """
    targets = []
    if include_planets:
        # Planets: use SPICE IDs (e.g., "199" for Mercury)
        targets.extend([str(info["spkid"]) for info in KETE_PLANETS.values()])
    if include_asteroids:
        # Asteroids: use names (kete doesn't recognize SPKIDs like "20000001")
        # Note: names may cause fuzzy matching errors, which are caught in caller
        targets.extend([v["name"] for v in KETE_ASTEROIDS_PHYSICS.values()])
    return targets


def map_spice_name_to_desig(name):
    """Map a SPICE target name/ID to a designation used in SBDB orbit files.

    Parameters
    ----------
    name : str
        The SPICE target name (e.g., "vesta", "ceres", "mercury") or
        SPICE ID string (e.g., "199" for Mercury, "20000004" for Vesta).
        Note: Asteroid SPKIDs are 20000000 + asteroid_number.

    Returns
    -------
    str
        The mapped designation in SBDB-style:
        - Asteroids: number only (e.g., "4" for Vesta, "1" for Ceres)
        - Planets: lowercase name (e.g., "mercury", "jupiter")
        If no mapping is found, returns the input name unchanged.
    """
    name_str = str(name)
    lower_name = name_str.lower()

    # Check asteroids by name or SPKID
    # SBDB uses simple numbers ("1", "2", "4", ...) not SPKIDs ("20000001", ...)
    for desig, info in KETE_ASTEROIDS_PHYSICS.items():
        if info["name"] == lower_name:
            return str(desig)  # e.g., "vesta" -> "4"
        if name_str == str(info["spkid"]):
            return str(desig)  # e.g., "20000004" -> "4"

    # Check planets by SPICE ID or name
    for pname, pinfo in KETE_PLANETS.items():
        if name_str == str(pinfo["spkid"]):
            return pname  # e.g., "199" -> "mercury"
        if lower_name == pname:
            return pname

    # Default: return as-is (preserving case for non-SPICE objects)
    return name


def is_spice_resolvable(desig):
    """Check if a designation can be resolved via SPICE kernels.

    Parameters
    ----------
    desig : str
        Object designation (name or number string).

    Returns
    -------
    bool
        True if the object is a planet or one of the 5 large asteroids.

    str or None
        The SPICE-compatible name if resolvable, else None.
    """
    desig_lower = str(desig).lower()

    # Check planets
    if desig_lower in KETE_PLANETS:
        return True, desig_lower

    # Check large asteroids by name
    for d, info in KETE_ASTEROIDS_PHYSICS.items():
        if desig_lower == info["name"]:
            return True, info["name"]
        if str(desig) == d:
            return True, info["name"]

    return False, None
