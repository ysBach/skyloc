from .fov import *
from .propagate import *

import kete


# Largest asteroids that are loaded in kete for n-body propagation.
KETE_LOADED_ASTEROIDS = []
KETE_LOADED_ASTEROIDS_NAME = []
for name, spkid in kete.spice.loaded_objects():
    if spkid > 20_000_000:
        KETE_LOADED_ASTEROIDS.append(str(spkid - 20_000_000))
        KETE_LOADED_ASTEROIDS_NAME.append(name)
