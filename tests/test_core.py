"""Tests for skyloc.core module - SSOLocator and related functions.

Tests verify specific behaviors with known values for 15 Eunomia at
JD 2460676.5 (2025-01-01).

Reference state vectors from kete propagation:
- Position: [0.0948511064, 2.2950080713, 0.2039276033] AU
- Velocity: [-0.0116237533, 0.0023363467, -0.0020398849] AU/day
"""
import numpy as np
import pytest
import pandas as pd
import kete
from skyloc.core import SSOLocator, calc_ephems
from skyloc.keteutils.fov import FOVCollection


class TestSSOLocatorInit:
    """Test SSOLocator initialization."""

    def test_basic_init(self, sample_orb_asteroid, sample_fovlist):
        """Basic initialization works."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        assert loc is not None
        assert len(loc.orb) == 1

    def test_orb_validation_error_message(self, sample_fovlist):
        """Missing required columns should raise error with specific message."""
        bad_orb = pd.DataFrame({"desig": ["test"], "ecc": [0.1]})
        with pytest.raises(ValueError, match="missing required columns"):
            SSOLocator(fovs=sample_fovlist, orb=bad_orb)

    def test_fovc_property_type(self, sample_orb_asteroid, sample_fovlist):
        """fovc property returns FOVCollection."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        assert isinstance(loc.fovc, FOVCollection)

    def test_fovc_length(self, sample_orb_asteroid, sample_fovlist):
        """fovc should have correct number of FOVs."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        assert len(loc.fovc) == 3

    def test_with_none_fovs(self, sample_orb_asteroid):
        """Can initialize with None fovs."""
        loc = SSOLocator(fovs=None, orb=sample_orb_asteroid)
        assert loc.fovc is None

    def test_orb_copy(self, sample_orb_asteroid, sample_fovlist):
        """copy_orb=True should create a copy."""
        original_ecc = sample_orb_asteroid.loc[0, "ecc"]
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid, copy_orb=True)
        sample_orb_asteroid.loc[0, "ecc"] = 0.999
        assert loc.orb.loc[0, "ecc"] == original_ecc
        sample_orb_asteroid.loc[0, "ecc"] = original_ecc


class TestSSOLocatorNonGravs:
    """Test SSOLocator non-gravitational model handling."""

    def test_non_gravs_true_creates_models(self, sample_orb_comet, sample_fovlist):
        """non_gravs=True creates models for comets."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_comet, non_gravs=True)
        assert len(loc.non_gravs) == 1
        assert loc.non_gravs[0] is not None

    def test_non_gravs_false_all_none(self, sample_orb_comet, sample_fovlist):
        """non_gravs=False creates None for all."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_comet, non_gravs=False)
        assert all(ng is None for ng in loc.non_gravs)

    def test_non_gravs_list_length(self, sample_orb_multi, sample_fovlist):
        """non_gravs list should match orb length."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_multi, non_gravs=True)
        assert len(loc.non_gravs) == len(sample_orb_multi)


class TestSSOLocatorStates:
    """Test SSOLocator state conversion."""

    def test_states_from_orb_length(self, sample_orb_asteroid, sample_fovlist):
        """states_from_orb returns correct number of states."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        states = loc.states_from_orb
        assert len(states) == 1

    def test_states_from_orb_desig(self, sample_orb_asteroid, sample_fovlist):
        """states_from_orb has correct designation."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        states = loc.states_from_orb
        assert states[0].desig == "15 Eunomia"


@pytest.mark.slow
@pytest.mark.integration
class TestSSOLocatorPropagation:
    """Test SSOLocator propagation methods.

    Reference: 15 Eunomia at JD 2460676.5 (2025-01-01)
    Position: [0.0948511064, 2.2950080713, 0.2039276033] AU
    Velocity: [-0.0116237533, 0.0023363467, -0.0020398849] AU/day
    """

    def test_propagate_n_body_sets_jd0(self, sample_orb_asteroid, sample_fovlist, sample_jd_tdb):
        """Propagation sets jd0 correctly."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        loc.propagate_n_body(jd0=sample_jd_tdb, suppress_errors=True)
        assert np.isclose(loc.jd0, sample_jd_tdb, atol=1e-10)

    def test_propagate_n_body_state_count(self, sample_orb_asteroid, sample_fovlist, sample_jd_tdb):
        """Propagation creates correct number of states."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        loc.propagate_n_body(jd0=sample_jd_tdb, suppress_errors=True)
        assert loc.states_propagated_jd0 is not None
        assert len(loc.states_propagated_jd0) == 1

    def test_propagate_with_mean_jd(self, sample_orb_asteroid, sample_fovlist, sample_jd_tdb):
        """Propagation with callable jd0 (np.mean) uses mean of FOV JDs."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        loc.propagate_n_body(jd0=np.mean, suppress_errors=True)
        expected_jd = sample_jd_tdb + 1.0
        assert np.isclose(loc.jd0, expected_jd, atol=1e-10)

    def test_propagated_state_desig(self, sample_orb_asteroid, sample_fovlist, sample_jd_tdb):
        """Propagated state has correct designation."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        loc.propagate_n_body(jd0=sample_jd_tdb, suppress_errors=True)
        assert loc.states_propagated_jd0[0].desig == "15 Eunomia"

    def test_propagated_state_jd(self, sample_orb_asteroid, sample_fovlist, sample_jd_tdb):
        """Propagated state has correct JD."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        loc.propagate_n_body(jd0=sample_jd_tdb, suppress_errors=True)
        assert np.isclose(loc.states_propagated_jd0[0].jd, sample_jd_tdb, atol=1e-10)

    def test_propagated_position_values(self, sample_orb_asteroid, sample_fovlist, sample_jd_tdb):
        """Propagated position should match reference values."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        loc.propagate_n_body(jd0=sample_jd_tdb, suppress_errors=True)
        state = loc.states_propagated_jd0[0]
        pos = np.array([state.pos[0], state.pos[1], state.pos[2]])
        # Reference: [0.0948511064, 2.2950080713, 0.2039276033] AU
        expected = np.array([0.0948511064, 2.2950080713, 0.2039276033])
        np.testing.assert_allclose(pos, expected, atol=1e-6)

    def test_propagated_velocity_values(self, sample_orb_asteroid, sample_fovlist, sample_jd_tdb):
        """Propagated velocity should match reference values."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        loc.propagate_n_body(jd0=sample_jd_tdb, suppress_errors=True)
        state = loc.states_propagated_jd0[0]
        vel = np.array([state.vel[0], state.vel[1], state.vel[2]])
        # Reference: [-0.0116237533, 0.0023363467, -0.0020398849] AU/day
        expected = np.array([-0.0116237533, 0.0023363467, -0.0020398849])
        np.testing.assert_allclose(vel, expected, atol=1e-8)

    def test_propagated_distance_from_sun(self, sample_orb_asteroid, sample_fovlist, sample_jd_tdb):
        """Heliocentric distance should match reference value."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        loc.propagate_n_body(jd0=sample_jd_tdb, suppress_errors=True)
        state = loc.states_propagated_jd0[0]
        r_hel = np.sqrt(state.pos[0]**2 + state.pos[1]**2 + state.pos[2]**2)
        # Reference: 2.306002 AU
        assert np.isclose(r_hel, 2.306002, atol=1e-5), f"r_hel={r_hel:.6f} AU"

    def test_propagated_velocity_magnitude(self, sample_orb_asteroid, sample_fovlist, sample_jd_tdb):
        """Orbital velocity magnitude should match reference value."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        loc.propagate_n_body(jd0=sample_jd_tdb, suppress_errors=True)
        state = loc.states_propagated_jd0[0]
        v_mag = np.sqrt(state.vel[0]**2 + state.vel[1]**2 + state.vel[2]**2)
        # Reference: sqrt(0.0116^2 + 0.0023^2 + 0.0020^2) ≈ 0.01193 AU/day
        expected_v = np.sqrt(0.0116237533**2 + 0.0023363467**2 + 0.0020398849**2)
        assert np.isclose(v_mag, expected_v, atol=1e-8), f"v={v_mag:.8f} AU/day"
