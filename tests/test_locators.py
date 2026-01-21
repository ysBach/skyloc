"""Tests for SpiceLocator and StarLocator classes.

Tests verify SPICE-based ephemerides and static FOV checks.
"""
import numpy as np
import pandas as pd
import pytest
import kete

from skyloc.core import SpiceLocator, StarLocator
from skyloc.keteutils import (
    get_default_spice_targets,
    get_spice_asteroid_names,
    get_spice_asteroid_desigs,
    is_spice_resolvable,
    KETE_PLANETS,
    KETE_ASTEROIDS_PHYSICS,
)
from skyloc.keteutils.fov import FOVCollection


# ==============================================================================
# Tests: SPICE Configuration
# ==============================================================================
class TestSpiceConfig:
    """Test SPICE target configuration functions."""

    def test_get_default_spice_targets_all(self):
        """Default targets include planets and asteroids."""
        targets = get_default_spice_targets(
            include_planets=True, include_asteroids=True
        )
        assert "499" in targets  # Mars
        assert "599" in targets  # Jupiter
        assert "ceres" in targets  # Ceres
        assert "vesta" in targets  # Vesta
        assert len(targets) == 14  # 9 planets + 5 asteroids

    def test_get_default_spice_targets_planets_only(self):
        """Can get planets only."""
        targets = get_default_spice_targets(
            include_planets=True, include_asteroids=False
        )
        assert "599" in targets  # Jupiter
        assert "ceres" not in targets
        assert len(targets) == 9

    def test_get_default_spice_targets_asteroids_only(self):
        """Can get asteroids only."""
        targets = get_default_spice_targets(
            include_planets=False, include_asteroids=True
        )
        assert "599" not in targets
        assert "ceres" in targets
        assert len(targets) == 5

    def test_get_spice_asteroid_names(self):
        """Asteroid names match known values."""
        names = get_spice_asteroid_names()
        assert "ceres" in names
        assert "pallas" in names
        assert "vesta" in names
        assert "hygiea" in names
        assert "interamnia" in names
        assert len(names) == 5

    def test_get_spice_asteroid_desigs(self):
        """Asteroid designations match known values."""
        desigs = get_spice_asteroid_desigs()
        assert "1" in desigs  # Ceres
        assert "2" in desigs  # Pallas
        assert "4" in desigs  # Vesta
        assert "10" in desigs  # Hygiea
        assert "704" in desigs  # Interamnia

    def test_is_spice_resolvable_planet(self):
        """Planet names are resolvable."""
        is_res, name = is_spice_resolvable("jupiter")
        assert is_res is True
        assert name == "jupiter"

    def test_is_spice_resolvable_asteroid_by_name(self):
        """Asteroid names are resolvable."""
        is_res, name = is_spice_resolvable("ceres")
        assert is_res is True
        assert name == "ceres"

    def test_is_spice_resolvable_asteroid_by_number(self):
        """Asteroid numbers are resolvable."""
        is_res, name = is_spice_resolvable("1")
        assert is_res is True
        assert name == "ceres"

    def test_is_spice_resolvable_unknown(self):
        """Unknown objects return False."""
        is_res, name = is_spice_resolvable("unknown_object")
        assert is_res is False
        assert name is None

    def test_kete_planets_structure(self):
        """KETE_PLANETS has correct structure."""
        assert "earth" in KETE_PLANETS
        assert KETE_PLANETS["earth"]["spkid"] == 399
        assert KETE_PLANETS["jupiter"]["spkid"] == 599

    def test_kete_asteroids_physics_structure(self):
        """KETE_ASTEROIDS_PHYSICS has correct structure."""
        assert "1" in KETE_ASTEROIDS_PHYSICS
        assert KETE_ASTEROIDS_PHYSICS["1"]["name"] == "ceres"
        assert KETE_ASTEROIDS_PHYSICS["1"]["spkid"] == 20_000_001


# ==============================================================================
# Tests: SpiceLocator
# ==============================================================================
class TestSpiceLocator:
    """Test SpiceLocator class."""

    def test_init_default_targets(self, sample_fovlist):
        """SpiceLocator initializes with default targets."""
        sl = SpiceLocator(sample_fovlist)
        assert len(sl.targets) == 14  # 9 planets + 5 asteroids
        assert "599" in sl.targets
        assert "ceres" in sl.targets

    def test_init_custom_targets(self, sample_fovlist):
        """SpiceLocator accepts custom targets."""
        sl = SpiceLocator(sample_fovlist, targets=["mars", "jupiter"])
        assert sl.targets == ["mars", "jupiter"]

    def test_init_planets_only(self, sample_fovlist):
        """SpiceLocator can be planets only."""
        sl = SpiceLocator(
            sample_fovlist, include_planets=True, include_asteroids=False
        )
        assert "599" in sl.targets
        assert "ceres" not in sl.targets

    def test_init_asteroids_only(self, sample_fovlist):
        """SpiceLocator can be asteroids only."""
        sl = SpiceLocator(
            sample_fovlist, include_planets=False, include_asteroids=True
        )
        assert "599" not in sl.targets
        assert "ceres" in sl.targets

    def test_fovc_set(self, sample_fovlist):
        """FOVCollection is properly set."""
        sl = SpiceLocator(sample_fovlist)
        assert isinstance(sl.fovc, FOVCollection)
        assert len(sl.fovc) == 3

    def test_loaded_objects(self, sample_fovlist):
        """loaded_objects attribute is populated."""
        sl = SpiceLocator(sample_fovlist)
        assert sl.loaded is not None
        # Should have at least planets loaded
        loaded_names = [name for name, _ in sl.loaded]
        assert "Earth" in loaded_names or "earth" in loaded_names

    @pytest.mark.integration
    def test_fov_spice_check_runs(self, sample_fov_earth_2024):
        """fov_spice_check executes without error."""
        sl = SpiceLocator([sample_fov_earth_2024], targets=["599"])
        sl.fov_spice_check()
        # Results may or may not find Jupiter depending on pointing
        assert sl.fov_check_fov2objs is not None
        assert sl.fov_check_objids is not None


# ==============================================================================
# Tests: StarLocator
# ==============================================================================
class TestStarLocator:
    """Test StarLocator class."""

    @pytest.fixture
    def sample_stars(self):
        """Sample star catalog with known positions."""
        return pd.DataFrame({
            "desig": ["Vega", "Sirius", "Arcturus"],
            "ra": [279.2347, 101.2875, 213.9153],
            "dec": [38.7837, -16.7161, 19.1824],
            "vmag": [0.03, -1.46, -0.05],
        })

    def test_init_with_sources(self, sample_fovlist, sample_stars):
        """StarLocator initializes with sources."""
        sl = StarLocator(sample_fovlist, sources=sample_stars)
        assert sl.sources is not None
        assert len(sl.sources) == 3

    def test_init_without_sources(self, sample_fovlist):
        """StarLocator initializes without sources."""
        sl = StarLocator(sample_fovlist)
        assert sl.sources is None

    def test_set_sources(self, sample_fovlist, sample_stars):
        """Can set sources after init."""
        sl = StarLocator(sample_fovlist)
        sl.set_sources(sample_stars)
        assert sl.sources is not None
        assert len(sl.sources) == 3

    def test_set_sources_auto_desig(self, sample_fovlist):
        """Auto-generates designations if not provided."""
        sources = pd.DataFrame({
            "ra": [0.0, 90.0, 180.0],
            "dec": [0.0, 45.0, -45.0],
        })
        sl = StarLocator(sample_fovlist)
        sl.set_sources(sources)
        assert "desig" in sl.sources.columns
        assert sl.sources["desig"].iloc[0] == "star_0"

    def test_set_sources_missing_columns(self, sample_fovlist):
        """Raises error if required columns missing."""
        sources = pd.DataFrame({"ra": [0.0]})  # Missing dec
        sl = StarLocator(sample_fovlist)
        with pytest.raises(ValueError, match="Missing required columns"):
            sl.set_sources(sources)

    def test_set_sources_wrong_type(self, sample_fovlist):
        """Raises error if sources not DataFrame."""
        sl = StarLocator(sample_fovlist)
        with pytest.raises(TypeError):
            sl.set_sources([(0, 0), (90, 45)])

    def test_fov_static_check_no_sources(self, sample_fovlist):
        """fov_static_check raises error without sources."""
        sl = StarLocator(sample_fovlist)
        with pytest.raises(ValueError, match="No sources set"):
            sl.fov_static_check()

    @pytest.mark.integration
    def test_fov_static_check_runs(self, sample_fov_earth_2024, sample_stars):
        """fov_static_check executes without error."""
        sl = StarLocator([sample_fov_earth_2024], sources=sample_stars)
        sl.fov_static_check()
        assert sl.fov_check_fov2objs is not None
        assert sl.sources_infov_mask is not None

    @pytest.mark.integration
    def test_get_visible_sources(self, sample_fov_earth_2024, sample_stars):
        """get_visible_sources returns DataFrame."""
        sl = StarLocator([sample_fov_earth_2024], sources=sample_stars)
        sl.fov_static_check()
        visible = sl.get_visible_sources()
        assert isinstance(visible, pd.DataFrame)

    def test_get_visible_sources_no_check(self, sample_fovlist, sample_stars):
        """get_visible_sources raises error without check."""
        sl = StarLocator(sample_fovlist, sources=sample_stars)
        with pytest.raises(ValueError, match="Run fov_static_check"):
            sl.get_visible_sources()


# ==============================================================================
# Tests: keteutils.spice module
# ==============================================================================
class TestSpiceUtils:
    """Test spice utility functions."""

    def test_get_states_single_jd(self):
        """get_states works with single JD."""
        from skyloc.keteutils.spice import get_states

        jd = 2460676.5  # 2025-01-01
        states = get_states("earth", jd, center="sun")
        assert len(states) == 1
        assert states[0].desig.lower() == "earth"

    def test_get_states_multiple_jds(self):
        """get_states works with multiple JDs."""
        from skyloc.keteutils.spice import get_states

        jds = [2460676.5, 2460677.5, 2460678.5]
        states = get_states("earth", jds, center="sun")
        assert len(states) == 3

    def test_get_state_arrays(self):
        """get_state_arrays returns numpy arrays."""
        from skyloc.keteutils.spice import get_state_arrays

        jds = [2460676.5, 2460677.5]
        pos, vel = get_state_arrays("earth", jds, center="sun")
        assert pos.shape == (2, 3)
        assert vel.shape == (2, 3)

    def test_is_spk_loaded_earth(self):
        """Earth should be loaded by default."""
        from skyloc.keteutils.spice import is_spk_loaded

        is_loaded, name = is_spk_loaded("earth")
        assert is_loaded is True

    def test_is_spk_loaded_horizons_format(self):
        """Handles Horizons "500@" format."""
        from skyloc.keteutils.spice import is_spk_loaded

        is_loaded, name = is_spk_loaded("500@399")
        assert name == "399"
