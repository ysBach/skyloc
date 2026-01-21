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

# Default planets for SPICE ephemerides (NAIF IDs and names)
# These are typically available in standard SPICE kernels
KETE_PLANETS = {
    "mercury": {"spkid": 199, "barycenter": 1},
    "venus": {"spkid": 299, "barycenter": 2},
    "earth": {"spkid": 399, "barycenter": 3},
    "mars": {"spkid": 499, "barycenter": 4},
    "jupiter": {"spkid": 599, "barycenter": 5},
    "saturn": {"spkid": 699, "barycenter": 6},
    "uranus": {"spkid": 799, "barycenter": 7},
    "neptune": {"spkid": 899, "barycenter": 8},
    "pluto": {"spkid": 999, "barycenter": 9},
}
