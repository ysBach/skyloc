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


class TestSSOLocatorMultiObject:
    """Test SSOLocator handling of multiple objects (Eunomia + Eros)."""

    def test_multi_object_pipeline(self, sample_orb_asteroid, sample_orb_eros, sample_fov_earth_2024):
        """End-to-end test for 2 objects: propagation -> fov check -> flux."""
        # Setup
        orb = pd.concat([sample_orb_asteroid, sample_orb_eros], ignore_index=True)
        loc = SSOLocator(fovs=sample_fov_earth_2024, orb=orb)

        # Propagate
        loc.propagate_n_body(jd0=sample_fov_earth_2024.observer.jd, include_asteroids=True)
        assert len(loc.states_propagated_jd0) == 2

        # FOV Check
        loc.fov_state_check(include_asteroids=False, use_spice_for_loaded=False)
        assert len(loc.fov_check_simstates) == 1
        assert len(loc.fov_check_simstates[0].states) == 2

        # Flux (Calc Ephems)
        loc.calc_ephems()
        eph = loc.eph
        assert len(eph) == 2

        # Reference Positions (Sun-centric J2000 Ecliptic)
        # Earth (from fixture/Horizons)
        pos_earth = np.array([-0.165851246, 0.969230782, -0.000054915])

        # Eunomia (Horizons)
        pos_eun = np.array([1.987815889, -1.216791974, 0.282474578])
        # Eros (Horizons)
        pos_eros = np.array([1.324107311, 0.525501167, 0.265860328])

        # Get Rows
        row_eun = eph[eph['desig'].str.contains('Eunomia')].iloc[0]
        row_eros = eph[eph['desig'].str.contains('Eros')].iloc[0]

        # --- Eunomia Validation ---
        # Magnitudes
        assert np.isclose(row_eun['vmag'], 10.45, atol=0.1)

        # Geometries
        # r_hel
        expected_r_hel_eun = np.linalg.norm(pos_eun)
        assert np.isclose(row_eun['r_hel'], expected_r_hel_eun, atol=1e-4)

        # r_obs
        expected_r_obs_eun = np.linalg.norm(pos_eun - pos_earth)
        assert np.isclose(row_eun['r_obs'], expected_r_obs_eun, atol=1e-4)

        # Phase (alpha)
        # Cos(alpha) = dot(Sun->Obj, Obs->Obj) / (|Sun->Obj| * |Obs->Obj|)
        vec_so = pos_eun
        vec_oo = pos_eun - pos_earth
        cos_alpha = np.dot(vec_so, vec_oo) / (np.linalg.norm(vec_so) * np.linalg.norm(vec_oo))
        expected_alpha_eun = np.degrees(np.arccos(cos_alpha))
        assert np.isclose(row_eun['alpha'], expected_alpha_eun, atol=0.1)

        # --- Eros Validation ---
        # Magnitudes
        assert np.isclose(row_eros['vmag'], 13.14, atol=0.2)

        # Geometries
        # r_hel
        expected_r_hel_eros = np.linalg.norm(pos_eros)
        assert np.isclose(row_eros['r_hel'], expected_r_hel_eros, atol=1e-4)

        # r_obs
        expected_r_obs_eros = np.linalg.norm(pos_eros - pos_earth)
        assert np.isclose(row_eros['r_obs'], expected_r_obs_eros, atol=1e-4)

        # Phase (alpha)
        vec_so = pos_eros
        vec_oo = pos_eros - pos_earth
        cos_alpha_eros = np.dot(vec_so, vec_oo) / (np.linalg.norm(vec_so) * np.linalg.norm(vec_oo))
        expected_alpha_eros = np.degrees(np.arccos(cos_alpha_eros))
        assert np.isclose(row_eros['alpha'], expected_alpha_eros, atol=0.1)


@pytest.mark.integration
class TestSSOLocatorSpiceReplacement:
    """Test SPICE replacement for loaded asteroids.

    When include_asteroids=True, the 5 large asteroids (Ceres, Pallas, Vesta,
    Hygiea, Interamnia) should have their n-body states replaced with SPICE
    data to avoid self-impact issues.
    """

    def test_vesta_gets_spice_state(self, sample_orb_vesta, sample_fovlist, sample_jd_tdb):
        """Vesta state should match SPICE data after propagation."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_vesta, drop_major_asteroids=False)
        loc.propagate_n_body(jd0=sample_jd_tdb, include_asteroids=True)

        # Get propagated state
        propagated = loc.states_propagated_jd0[0]

        # Get expected SPICE state
        expected = kete.spice.get_state("vesta", sample_jd_tdb)

        # Positions should match exactly (replaced by SPICE)
        np.testing.assert_allclose(
            [propagated.pos[0], propagated.pos[1], propagated.pos[2]],
            [expected.pos[0], expected.pos[1], expected.pos[2]],
            atol=1e-10
        )

    def test_vesta_not_nan(self, sample_orb_vesta, sample_fovlist, sample_jd_tdb):
        """Vesta state should never be NaN when include_asteroids=True."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_vesta, drop_major_asteroids=False)
        loc.propagate_n_body(jd0=sample_jd_tdb, include_asteroids=True)

        state = loc.states_propagated_jd0[0]
        assert not np.isnan(state.pos[0]), "Vesta position should not be NaN"
        assert not np.isnan(state.pos[1]), "Vesta position should not be NaN"
        assert not np.isnan(state.pos[2]), "Vesta position should not be NaN"

    def test_mixed_objects_spice_replacement(
        self, sample_orb_asteroid, sample_orb_vesta, sample_fovlist, sample_jd_tdb
    ):
        """Mixed objects: Vesta gets SPICE, Eunomia gets n-body."""
        orb = pd.concat([sample_orb_asteroid, sample_orb_vesta], ignore_index=True)
        loc = SSOLocator(fovs=sample_fovlist, orb=orb, drop_major_asteroids=False)
        loc.propagate_n_body(jd0=sample_jd_tdb, include_asteroids=True)

        assert len(loc.states_propagated_jd0) == 2

        # Find Vesta and Eunomia states
        vesta_state = None
        eunomia_state = None
        for s in loc.states_propagated_jd0:
            if s.desig == "vesta" or s.desig == "4":
                vesta_state = s
            elif "Eunomia" in s.desig or s.desig == "15":
                eunomia_state = s

        # Vesta should match SPICE exactly
        expected_vesta = kete.spice.get_state("vesta", sample_jd_tdb)
        np.testing.assert_allclose(
            [vesta_state.pos[0], vesta_state.pos[1], vesta_state.pos[2]],
            [expected_vesta.pos[0], expected_vesta.pos[1], expected_vesta.pos[2]],
            atol=1e-10
        )

        # Eunomia should NOT be NaN (it's a regular n-body result)
        assert not np.isnan(eunomia_state.pos[0])

    def test_spice_replacement_from_cache(
        self, sample_orb_vesta, sample_fovlist, sample_jd_tdb, tmp_path
    ):
        """SPICE replacement should work even when loading from cached file."""
        output_file = tmp_path / "test_simstates.parq"

        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_vesta, drop_major_asteroids=False)

        # First run: compute and save (with include_asteroids=True)
        loc.propagate_n_body(
            jd0=sample_jd_tdb, include_asteroids=True, output=output_file
        )
        first_pos = [
            loc.states_propagated_jd0[0].pos[0],
            loc.states_propagated_jd0[0].pos[1],
            loc.states_propagated_jd0[0].pos[2],
        ]

        # Second run: load from cache (should STILL apply SPICE replacement)
        loc2 = SSOLocator(fovs=sample_fovlist, orb=sample_orb_vesta, drop_major_asteroids=False)
        loc2.propagate_n_body(
            jd0=sample_jd_tdb, include_asteroids=True, output=output_file, overwrite=False
        )
        second_pos = [
            loc2.states_propagated_jd0[0].pos[0],
            loc2.states_propagated_jd0[0].pos[1],
            loc2.states_propagated_jd0[0].pos[2],
        ]

        # Both should match SPICE data
        expected = kete.spice.get_state("vesta", sample_jd_tdb)
        np.testing.assert_allclose(
            first_pos,
            [expected.pos[0], expected.pos[1], expected.pos[2]],
            atol=1e-10
        )
        np.testing.assert_allclose(
            second_pos,
            [expected.pos[0], expected.pos[1], expected.pos[2]],
            atol=1e-10
        )

    def test_fov_state_check_no_asteroid_spice_errors(
        self, sample_orb_vesta, sample_fov_earth_2024
    ):
        """fov_state_check should not fail with 'Multiple objects match' for asteroids.

        This tests that we don't use fov_spice_check for asteroids (which causes
        fuzzy matching issues), but instead rely on the SPICE-replaced n-body results.
        """
        import logging
        import io

        # Set up logging capture
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("skyloc.core")
        original_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        try:
            loc = SSOLocator(fovs=sample_fov_earth_2024, orb=sample_orb_vesta, drop_major_asteroids=False)
            loc.propagate_n_body(jd0=sample_fov_earth_2024.observer.jd, include_asteroids=True)
            loc.fov_state_check(include_asteroids=True, use_spice_for_loaded=True)

            # Check logs for asteroid-related errors
            log_output = log_capture.getvalue()
            # Should NOT see "Multiple objects match" for asteroids
            assert "fov_spice_check failed for ceres" not in log_output
            assert "fov_spice_check failed for vesta" not in log_output
            assert "fov_spice_check failed for pallas" not in log_output
            assert "fov_spice_check failed for hygiea" not in log_output
            assert "fov_spice_check failed for interamnia" not in log_output
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

    def test_full_pipeline_with_loaded_asteroid(
        self, sample_orb_vesta, sample_fov_earth_2024
    ):
        """Full pipeline test: Vesta should have valid SPICE-based ephemeris.

        Tests propagate_n_body -> fov_state_check -> calc_ephems for a loaded asteroid.
        """
        loc = SSOLocator(fovs=sample_fov_earth_2024, orb=sample_orb_vesta, drop_major_asteroids=False)

        # Step 1: Propagate (should SPICE-replace Vesta)
        jd = sample_fov_earth_2024.observer.jd
        loc.propagate_n_body(jd0=jd, include_asteroids=True)

        # Verify Vesta has SPICE position
        expected_spice = kete.spice.get_state("vesta", jd)
        propagated = loc.states_propagated_jd0[0]
        np.testing.assert_allclose(
            [propagated.pos[0], propagated.pos[1], propagated.pos[2]],
            [expected_spice.pos[0], expected_spice.pos[1], expected_spice.pos[2]],
            atol=1e-10,
            err_msg="Vesta should have SPICE position after propagation"
        )

        # Step 2: FOV state check (should work without errors)
        # Note: Vesta may or may not be in the FOV depending on geometry
        loc.fov_state_check(include_asteroids=True, use_spice_for_loaded=False)

        # Step 3: If Vesta is in FOV, calculate ephemeris
        if loc.fov_check_simstates and any(
            "4" in [s.desig for s in ss.states] or "vesta" in [s.desig.lower() for s in ss.states]
            for ss in loc.fov_check_simstates
        ):
            loc.calc_ephems()
            assert loc.eph is not None, "Ephemeris should be calculated"

            # Verify Vesta's heliocentric distance is reasonable (2-3 AU for Vesta)
            vesta_eph = loc.eph[loc.eph["desig"] == "4"]
            if len(vesta_eph) > 0:
                r_hel = vesta_eph.iloc[0]["r_hel"]
                assert 2.0 < r_hel < 3.0, f"Vesta r_hel should be 2-3 AU, got {r_hel}"


class TestCalcEphemsEdgeCases:
    """Test edge cases in calc_ephems."""

    def test_calc_ephems_spice_only_has_geometry(self, sample_orb_asteroid):
        """SPICE-only objects should have geometry data with vmag=99.0.

        This tests the case where fov_state_check finds only planets (from SPICE check)
        but the orbit file only contains asteroids. The planets should appear in
        ephemeris with geometric data but vmag=99.0 (no magnitude calculation possible).
        """
        from skyloc.core import _calc_ephem

        # Create a mock simulstates with only a planet designation (no orbit data)
        observer = kete.State(
            desig="test_obs",
            jd=2460676.5,
            pos=kete.Vector([0.99, 0.02, 0.01], frame=kete.Frames.Ecliptic),
            vel=kete.Vector([-0.002, 0.017, 0.0], frame=kete.Frames.Ecliptic),
            center_id=10,
        )
        planet_state = kete.State(
            desig="saturn barycenter",  # SPICE-only, no orbit data
            jd=2460676.5,
            pos=kete.Vector([5.0, 5.0, 0.1], frame=kete.Frames.Ecliptic),
            vel=kete.Vector([-0.001, 0.002, 0.0], frame=kete.Frames.Ecliptic),
            center_id=10,
        )
        fov = kete.fov.ConeFOV(
            observer=observer,
            pointing=kete.Vector.from_lat_lon(0, 45),
            angle=180,
        )
        simulstates = kete.SimultaneousStates([planet_state], fov=fov)

        # Should return DataFrame with geometry but vmag=99.0
        eph, _ = _calc_ephem(sample_orb_asteroid, simulstates)
        assert len(eph) == 1, "Should include SPICE-only object"
        assert eph.iloc[0]["desig"] == "saturn barycenter"
        assert eph.iloc[0]["vmag"] == 99.0, "SPICE-only object should have vmag=99.0"
        # Geometry should be valid (not NaN)
        assert not np.isnan(eph.iloc[0]["r_hel"]), "r_hel should be calculated"
        assert not np.isnan(eph.iloc[0]["r_obs"]), "r_obs should be calculated"
        assert not np.isnan(eph.iloc[0]["ra"]), "ra should be calculated"
        assert not np.isnan(eph.iloc[0]["dec"]), "dec should be calculated"
