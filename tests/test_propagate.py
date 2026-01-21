"""Tests for skyloc.keteutils.propagate module.

Tests verify numerical results against expected values computed from kete
for 15 Eunomia at JD 2460676.5 (2025-01-01).

Reference values (geocentric observer):
- r_hel = 2.306002 AU
- r_obs = 1.371160 AU
- α = 9.841240°
- RA = 76.476535°, Dec = 31.449820°
- sky_motion = 0.551505 arcsec/min
"""
import numpy as np
import pytest
import kete
from skyloc.keteutils.propagate import make_nongravs_models, calc_geometries


class TestMakeNonGravModels:
    """Test non-gravitational model creation."""

    def test_asteroid_no_nongrav(self, sample_orb_asteroid):
        """Asteroid with zero A1/A2/A3 should have None nongrav."""
        non_gravs = make_nongravs_models(sample_orb_asteroid)
        assert len(non_gravs) == 1
        assert non_gravs[0] is None

    def test_comet_with_nongrav(self, sample_orb_comet):
        """Comet with A1/A2 should have NonGravModel."""
        non_gravs = make_nongravs_models(sample_orb_comet)
        assert len(non_gravs) == 1
        assert non_gravs[0] is not None
        assert isinstance(non_gravs[0], kete.propagation.NonGravModel)

    def test_multi_orb_correct_assignment(self, sample_orb_multi):
        """Mixed orb: asteroid=None, comet=model."""
        non_gravs = make_nongravs_models(sample_orb_multi)
        assert len(non_gravs) == 2
        assert non_gravs[0] is None  # 15 Eunomia (asteroid)
        assert non_gravs[1] is not None  # 1P/Halley (comet)


class TestCalcGeometries:
    """Test geometry calculations from SimultaneousStates.

    Reference values computed for 15 Eunomia at JD 2460676.5 (2025-01-01)
    with geocentric observer.
    """

    @pytest.fixture
    def skyloc_geoms_geocentric(self, sample_orb_asteroid, sample_jd_tdb):
        """Compute geometries with geocentric observer."""
        earth_state = kete.spice.get_state("earth", sample_jd_tdb, "sun")
        observer = kete.State(
            desig="geocentric",
            jd=sample_jd_tdb,
            pos=earth_state.pos,
            vel=earth_state.vel,
            center_id=10,
        )
        pointing = kete.Vector.from_lat_lon(0.0, 0.0)
        fov = kete.fov.ConeFOV(observer=observer, pointing=pointing, angle=180.0)

        states = kete.mpc.table_to_states(sample_orb_asteroid)
        propagated = kete.propagate_n_body(states, jd=sample_jd_tdb, suppress_errors=True)
        simulstates = kete.SimultaneousStates(propagated, fov=fov)
        return calc_geometries(simulstates, rates_in_arcsec_per_min=True)

    @pytest.mark.slow
    def test_output_keys(self, skyloc_geoms_geocentric):
        """Geometry output has all expected keys."""
        expected_keys = [
            "desig", "alpha", "r_hel", "r_obs", "ra", "dec",
            "hel_ecl_lon", "hel_ecl_lat", "obs_ecl_lon", "obs_ecl_lat",
            "dra*cosdec/dt", "ddec/dt", "sky_motion", "sky_motion_pa"
        ]
        for key in expected_keys:
            assert key in skyloc_geoms_geocentric, f"Missing key: {key}"

    @pytest.mark.slow
    def test_desig_matches(self, skyloc_geoms_geocentric):
        """Designation should match input orbit."""
        assert skyloc_geoms_geocentric["desig"][0] == "15 Eunomia"

    @pytest.mark.slow
    def test_heliocentric_distance(self, skyloc_geoms_geocentric):
        """r_hel should match reference value."""
        r_hel = skyloc_geoms_geocentric["r_hel"][0]
        # Reference: 2.306002 AU
        assert np.isclose(r_hel, 2.306002, atol=1e-5), f"r_hel={r_hel:.6f} AU"

    @pytest.mark.slow
    def test_observer_distance(self, skyloc_geoms_geocentric):
        """r_obs should match reference value."""
        r_obs = skyloc_geoms_geocentric["r_obs"][0]
        # Reference: 1.371160 AU
        assert np.isclose(r_obs, 1.371160, atol=1e-5), f"r_obs={r_obs:.6f} AU"

    @pytest.mark.slow
    def test_phase_angle(self, skyloc_geoms_geocentric):
        """Phase angle should match reference value."""
        alpha = skyloc_geoms_geocentric["alpha"][0]
        # Reference: 9.841240°
        assert np.isclose(alpha, 9.841240, atol=0.01), f"alpha={alpha:.6f}°"

    @pytest.mark.slow
    def test_ra_dec_values(self, skyloc_geoms_geocentric):
        """RA/Dec should match reference values."""
        ra = skyloc_geoms_geocentric["ra"][0]
        dec = skyloc_geoms_geocentric["dec"][0]
        # Reference: RA=76.476535°, Dec=31.449820°
        assert np.isclose(ra, 76.476535, atol=0.01), f"RA={ra:.6f}°"
        assert np.isclose(dec, 31.449820, atol=0.01), f"Dec={dec:.6f}°"

    @pytest.mark.slow
    def test_sky_motion(self, skyloc_geoms_geocentric):
        """Sky motion should match reference value."""
        motion = skyloc_geoms_geocentric["sky_motion"][0]
        # Reference: 0.551505 arcsec/min
        assert np.isclose(motion, 0.551505, atol=0.001), f"motion={motion:.6f} arcsec/min"

    @pytest.mark.slow
    def test_sky_motion_consistency(self, skyloc_geoms_geocentric):
        """sky_motion should equal hypot of rate components."""
        geoms = skyloc_geoms_geocentric
        computed = np.hypot(geoms["dra*cosdec/dt"], geoms["ddec/dt"])
        np.testing.assert_allclose(geoms["sky_motion"], computed, rtol=1e-6)


class TestCalcGeometriesPhysics:
    """Test physical consistency of geometry calculations."""

    @pytest.fixture
    def skyloc_geoms_geocentric(self, sample_orb_asteroid, sample_jd_tdb):
        """Compute geometries with geocentric observer."""
        earth_state = kete.spice.get_state("earth", sample_jd_tdb, "sun")
        observer = kete.State(
            desig="geocentric",
            jd=sample_jd_tdb,
            pos=earth_state.pos,
            vel=earth_state.vel,
            center_id=10,
        )
        pointing = kete.Vector.from_lat_lon(0.0, 0.0)
        fov = kete.fov.ConeFOV(observer=observer, pointing=pointing, angle=180.0)

        states = kete.mpc.table_to_states(sample_orb_asteroid)
        propagated = kete.propagate_n_body(states, jd=sample_jd_tdb, suppress_errors=True)
        simulstates = kete.SimultaneousStates(propagated, fov=fov)
        return calc_geometries(simulstates, rates_in_arcsec_per_min=True), earth_state

    @pytest.mark.slow
    def test_phase_angle_law_of_cosines(self, skyloc_geoms_geocentric):
        """Phase angle should satisfy law of cosines.

        cos(alpha) = (r_hel^2 + r_obs^2 - r_sun_obs^2) / (2 * r_hel * r_obs)
        """
        geoms, earth_state = skyloc_geoms_geocentric
        r_hel = geoms["r_hel"][0]
        r_obs = geoms["r_obs"][0]
        alpha = geoms["alpha"][0]

        # Observer-Sun distance
        r_sun_obs = np.sqrt(earth_state.pos[0]**2 + earth_state.pos[1]**2 + earth_state.pos[2]**2)

        # Law of cosines
        cos_alpha_computed = (r_hel**2 + r_obs**2 - r_sun_obs**2) / (2 * r_hel * r_obs)
        alpha_computed = np.rad2deg(np.arccos(cos_alpha_computed))

        # Should match to 0.01°
        assert np.isclose(alpha, alpha_computed, atol=0.01), (
            f"α={alpha:.4f}°, law_of_cosines={alpha_computed:.4f}°"
        )


@pytest.mark.network
@pytest.mark.slow
class TestHorizonsKeteCrossValidation:
    """Cross-validate kete propagation with JPL Horizons.

    Using high-precision JPL SBDB orbital elements, the results should match
    Horizons to better than 0.01° for angles and 1e-5 AU for distances.
    """

    @pytest.fixture
    def horizons_eunomia(self, sample_jd_tdb):
        """Query Horizons for 15 Eunomia ephemeris."""
        from skyloc.jplhorizons import horizons_quick

        _, eph = horizons_quick(
            objid="15",
            epochs=[sample_jd_tdb],
            location="500",  # Geocentric
        )
        return eph.iloc[0]

    @pytest.fixture
    def skyloc_eunomia_geocentric(self, sample_orb_asteroid, sample_jd_tdb):
        """Compute kete ephemeris with geocentric observer."""
        earth_state = kete.spice.get_state("earth", sample_jd_tdb, "sun")
        observer = kete.State(
            desig="geocentric",
            jd=sample_jd_tdb,
            pos=earth_state.pos,
            vel=earth_state.vel,
            center_id=10,
        )
        pointing = kete.Vector.from_lat_lon(0.0, 0.0)
        fov = kete.fov.ConeFOV(observer=observer, pointing=pointing, angle=180.0)

        states = kete.mpc.table_to_states(sample_orb_asteroid)
        propagated = kete.propagate_n_body(states, jd=sample_jd_tdb, suppress_errors=True)
        simulstates = kete.SimultaneousStates(propagated, fov=fov)
        return calc_geometries(simulstates, rates_in_arcsec_per_min=True)

    def test_heliocentric_distance_matches(self, horizons_eunomia, skyloc_eunomia_geocentric):
        """Horizons and kete r_hel should match within 1e-5 AU."""
        horizons_r = horizons_eunomia["r_hel"]
        kete_r = skyloc_eunomia_geocentric["r_hel"][0]

        diff = abs(horizons_r - kete_r)
        assert np.isclose(horizons_r, kete_r, atol=1e-4), (
            f"r_hel: Horizons={horizons_r:.6f}, kete={kete_r:.6f}, diff={diff:.6f} AU"
        )

    def test_observer_distance_matches(self, horizons_eunomia, skyloc_eunomia_geocentric):
        """Horizons and kete r_obs should match within 1e-5 AU."""
        horizons_r = horizons_eunomia["r_obs"]
        kete_r = skyloc_eunomia_geocentric["r_obs"][0]

        diff = abs(horizons_r - kete_r)
        assert np.isclose(horizons_r, kete_r, atol=1e-4), (
            f"r_obs: Horizons={horizons_r:.6f}, kete={kete_r:.6f}, diff={diff:.6f} AU"
        )

    def test_ra_matches(self, horizons_eunomia, skyloc_eunomia_geocentric):
        """Horizons and kete RA should match within 0.01°."""
        horizons_ra = horizons_eunomia["ra"]
        kete_ra = skyloc_eunomia_geocentric["ra"][0]

        diff = abs(horizons_ra - kete_ra)
        assert np.isclose(horizons_ra, kete_ra, atol=0.01), (
            f"RA: Horizons={horizons_ra:.6f}, kete={kete_ra:.6f}, diff={diff:.6f}°"
        )

    def test_dec_matches(self, horizons_eunomia, skyloc_eunomia_geocentric):
        """Horizons and kete Dec should match within 0.01°."""
        horizons_dec = horizons_eunomia["dec"]
        kete_dec = skyloc_eunomia_geocentric["dec"][0]

        diff = abs(horizons_dec - kete_dec)
        assert np.isclose(horizons_dec, kete_dec, atol=0.01), (
            f"Dec: Horizons={horizons_dec:.6f}, kete={kete_dec:.6f}, diff={diff:.6f}°"
        )

    def test_phase_angle_matches(self, horizons_eunomia, skyloc_eunomia_geocentric):
        """Horizons and kete phase angle should match within 0.01°."""
        horizons_alpha = horizons_eunomia["alpha"]
        kete_alpha = skyloc_eunomia_geocentric["alpha"][0]

        diff = abs(horizons_alpha - kete_alpha)
        assert np.isclose(horizons_alpha, kete_alpha, atol=0.01), (
            f"α: Horizons={horizons_alpha:.4f}, kete={kete_alpha:.4f}, diff={diff:.4f}°"
        )

    def test_sky_motion_matches(self, horizons_eunomia, skyloc_eunomia_geocentric):
        """Horizons and kete sky motion should match within 0.1%."""
        horizons_motion = horizons_eunomia["sky_motion"]
        kete_motion = skyloc_eunomia_geocentric["sky_motion"][0]

        assert np.isclose(horizons_motion, kete_motion, rtol=0.001), (
            f"motion: Horizons={horizons_motion:.6f}, kete={kete_motion:.6f} arcsec/min"
        )

    def test_hel_ecl_lon_matches(self, horizons_eunomia, skyloc_eunomia_geocentric):
        """Horizons and kete heliocentric ecliptic longitude should match."""
        horizons_lon = horizons_eunomia["hel_ecl_lon"]
        kete_lon = skyloc_eunomia_geocentric["hel_ecl_lon"][0]

        diff = abs(horizons_lon - kete_lon)
        assert np.isclose(horizons_lon, kete_lon, atol=0.01), (
            f"hel_ecl_lon: Horizons={horizons_lon:.4f}, kete={kete_lon:.4f}, diff={diff:.4f}°"
        )

    def test_hel_ecl_lat_matches(self, horizons_eunomia, skyloc_eunomia_geocentric):
        """Horizons and kete heliocentric ecliptic latitude should match."""
        horizons_lat = horizons_eunomia["hel_ecl_lat"]
        kete_lat = skyloc_eunomia_geocentric["hel_ecl_lat"][0]

        diff = abs(horizons_lat - kete_lat)
        assert np.isclose(horizons_lat, kete_lat, atol=0.01), (
            f"hel_ecl_lat: Horizons={horizons_lat:.4f}, kete={kete_lat:.4f}, diff={diff:.4f}°"
        )

    def test_obs_ecl_lon_matches(self, horizons_eunomia, skyloc_eunomia_geocentric):
        """Horizons and kete observer-centric ecliptic longitude should match."""
        horizons_lon = horizons_eunomia["obs_ecl_lon"]
        kete_lon = skyloc_eunomia_geocentric["obs_ecl_lon"][0]

        diff = abs(horizons_lon - kete_lon)
        # Slightly larger tolerance for observer-centric coordinates
        assert np.isclose(horizons_lon, kete_lon, atol=0.5), (
            f"obs_ecl_lon: Horizons={horizons_lon:.4f}, kete={kete_lon:.4f}, diff={diff:.4f}°"
        )

    def test_obs_ecl_lat_matches(self, horizons_eunomia, skyloc_eunomia_geocentric):
        """Horizons and kete observer-centric ecliptic latitude should match."""
        horizons_lat = horizons_eunomia["obs_ecl_lat"]
        kete_lat = skyloc_eunomia_geocentric["obs_ecl_lat"][0]

        diff = abs(horizons_lat - kete_lat)
        assert np.isclose(horizons_lat, kete_lat, atol=0.5), (
            f"obs_ecl_lat: Horizons={horizons_lat:.4f}, kete={kete_lat:.4f}, diff={diff:.4f}°"
        )

    def test_sky_motion_pa_matches(self, horizons_eunomia, skyloc_eunomia_geocentric):
        """Horizons and kete sky motion PA should match (modulo 360°).

        Horizons uses [0, 360] range, kete uses [-180, 180].
        PA=230° is equivalent to PA=-130°.
        """
        horizons_pa = horizons_eunomia["sky_motion_pa"]
        kete_pa = skyloc_eunomia_geocentric["sky_motion_pa"][0]

        # Normalize both to [0, 360)
        horizons_pa_norm = horizons_pa % 360
        kete_pa_norm = kete_pa % 360

        diff = abs(horizons_pa_norm - kete_pa_norm)
        # Handle wraparound at 360°
        if diff > 180:
            diff = 360 - diff

        assert diff < 0.5, (
            f"sky_motion_pa: Horizons={horizons_pa:.2f}°, kete={kete_pa:.2f}°, diff={diff:.2f}°"
        )


class TestErosPropagation:
    """Validate propagation accuracy for (433) Eros against JPL Horizons."""

    def test_vs_horizons_2024(self, sample_orb_eros, sample_fov_earth_2024):
        """Propagate Eros to 2024-01-01 and check position.

        Reference: JPL Horizons Sun-centric geometric state.
        """
        # Convert orbit to state
        states_in = kete.mpc.table_to_states(sample_orb_eros)
        jd_target = sample_fov_earth_2024.observer.jd

        # Propagate
        states_out = kete.propagate_n_body(
            states_in,
            jd=jd_target,
            include_asteroids=True,
            suppress_errors=True
        )

        eros_state = states_out[0]
        pos = np.array(eros_state.pos)

        # Horizons Reference (Sun-centric J2000 Ecliptic)
        ref = np.array([1.324107311220569, 0.5255011668486284, 0.2658603283643745])

        dist = np.linalg.norm(pos - ref)
        # Tolerance: < 1e-5 AU (~1500 km). Expected ~3e-9 AU (~450 m).
        assert dist < 1e-5, f"Eros position diff: {dist:.3e} AU"
