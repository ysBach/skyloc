from .fov import *
from .propagate import *
from .stderr import *
from .spice import *
from .frames import *
from .statevec import *

# SBDB, as of 2025 Jun
KETE_ASTEROIDS_PHYSICS = {
    "1": {
        "name": "ceres",
        "spkid": 20_000_001,
        "H": 3.34,
        "G": 0.12,
        "diameter": 939.4,
        "p_V": 0.090,
    },
    "2": {
        "name": "pallas",
        "spkid": 20_000_002,
        "H": 4.11,
        "G": 0.11,
        "diameter": 513.0,
        "p_V": 0.155,
    },
    "4": {
        "name": "vesta",
        "spkid": 20_000_004,
        "H": 3.25,
        "G": 0.32,
        "diameter": 522.77,
        "p_V": 0.4228,
    },
    "10": {
        "name": "hygiea",
        "spkid": 20_000_010,
        "H": 5.65,
        "G": 0.15,
        "diameter": 407.12,
        "p_V": 0.0717,
    },
    "704": {
        "name": "interamnia",
        "spkid": 20_000_704,
        "H": 6.34,
        "G": -0.02,
        "diameter": 306.313,
        "p_V": 0.078,
    },
}
