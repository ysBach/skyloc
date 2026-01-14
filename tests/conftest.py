"""Pytest configuration and fixtures for skyloc tests."""
import numpy as np
import pandas as pd
import pytest
import kete


# ==============================================================================
# Fixtures: Sample orbital data
# ==============================================================================
@pytest.fixture
def sample_orb_asteroid():
    """Create a sample orbit DataFrame for a single asteroid (15 Eunomia).

    Eunomia is a well-studied S-type asteroid with stable orbital elements.
    It is NOT one of the "large asteroids" in kete (Ceres, Pallas, Interamnia,
    Hygiea, Vesta), so it behaves like a general SSO.

    Elements from JPL SBDB (epoch JD 2461000.5) with full precision:
    https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=15&full-prec=true

    Returns a DataFrame with orbital elements in kete-style column names.
    """
    return pd.DataFrame({
        "desig": ["15 Eunomia"],
        "ecc": [0.187781330746169],
        "incl": [11.76143490937041],
        "peri_dist": [2.14603288941038],  # q from JPL
        "peri_arg": [98.50719966771784],
        "lon_node": [292.8812264964728],
        "peri_time": [2460504.935745898130],
        "epoch": [2461000.5],
        "H": [5.43],
        "G": [0.23],
        "M1": [np.nan],
        "M2": [np.nan],
        "K1": [np.nan],
        "K2": [np.nan],
        "PC": [np.nan],
        "A1": [0.0],
        "A2": [0.0],
        "A3": [0.0],
        "DT": [0.0],
    })


@pytest.fixture
def sample_orb_comet():
    """Create a sample orbit DataFrame for a single comet (Halley).

    Returns a DataFrame with orbital elements and comet magnitude params.
    """
    return pd.DataFrame({
        "desig": ["1P/Halley"],
        "ecc": [0.96714],
        "incl": [162.26],
        "peri_dist": [0.586],
        "peri_arg": [111.33],
        "lon_node": [58.42],
        "peri_time": [2446467.4],
        "epoch": [2449400.5],
        "H": [np.nan],
        "G": [np.nan],
        "M1": [5.5],
        "M2": [13.8],
        "K1": [4.0],
        "K2": [10.0],
        "PC": [0.03],
        "A1": [0.01],
        "A2": [0.005],
        "A3": [0.0],
        "DT": [0.0],
    })


@pytest.fixture
def sample_orb_multi(sample_orb_asteroid, sample_orb_comet):
    """Create a sample orbit DataFrame with multiple objects."""
    return pd.concat([sample_orb_asteroid, sample_orb_comet], ignore_index=True)


# ==============================================================================
# Fixtures: Sample FOV
# ==============================================================================
@pytest.fixture
def sample_jd_tdb():
    """A sample Julian Date (TDB) for testing: 2025-01-01 00:00:00 TDB."""
    return 2460676.5


@pytest.fixture
def sample_fov(sample_jd_tdb):
    """Create a sample ConeFOV centered on a position in space."""
    # Observer state (approximate Earth position/velocity at epoch)
    observer = kete.State(
        desig="test_obs",
        jd=sample_jd_tdb,
        pos=kete.Vector([0.99, 0.02, 0.01], frame=kete.Frames.Ecliptic),
        vel=kete.Vector([-0.002, 0.017, 0.0], frame=kete.Frames.Ecliptic),
        center_id=10,
    )
    # Pointing direction (ecliptic lon=180, lat=0)
    pointing = kete.Vector.from_lat_lon(0.0, 180.0)

    return kete.fov.ConeFOV(observer=observer, pointing=pointing, angle=1.0)


@pytest.fixture
def sample_fovlist(sample_jd_tdb):
    """Create a sample FOVList with multiple FOVs at different times."""
    fovs = []
    for i, jd_offset in enumerate([0, 1, 2]):
        jd = sample_jd_tdb + jd_offset
        observer = kete.State(
            desig=f"test_obs_{i}",
            jd=jd,
            pos=kete.Vector([0.99 + 0.01 * i, 0.02, 0.01], frame=kete.Frames.Ecliptic),
            vel=kete.Vector([-0.002, 0.017, 0.0], frame=kete.Frames.Ecliptic),
            center_id=10,
        )
        pointing = kete.Vector.from_lat_lon(0.0, 180.0 + i * 10)
        fovs.append(kete.fov.ConeFOV(observer=observer, pointing=pointing, angle=1.0))
    return kete.fov.FOVList(fovs)


# ==============================================================================
# Fixtures: Tolerances
# ==============================================================================
@pytest.fixture
def atol_deg():
    """Absolute tolerance for angles in degrees (arcsec level)."""
    return 1 / 3600  # 1 arcsec


@pytest.fixture
def rtol():
    """Default relative tolerance."""
    return 1e-6


# ==============================================================================
# Markers
# ==============================================================================
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "network: marks tests requiring network access")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
