"""Tests for kete-backed propagation and ephemeris restoration."""

import numpy as np
import pytest

import kete

from skyloc.core import calc_ephems
from skyloc.keteutils.propagate import calc_geometries, make_nongravs_models
from tests.ephemeris_reference import (
    DIST_ATOL_AU,
    QUERY_ANGLE_ATOL_DEG,
    REFERENCE_TIMES_UTC,
    SMALL_BODY_ORBITS,
    TARGET_KEYS,
    _angle_delta_deg,
    _sun_observer,
    _to_tdb_jd,
    _wide_fov,
    reference_horizons_ephemeris,
    reference_state,
    reference_tdb_jd,
)

PROPAGATION_EPHEM_TARGETS = (
    "earth",
    "4",
    "15",
    "433",
    "saturn_barycenter",
)


class TestMakeNonGravModels:
    """Test non-gravitational model creation."""

    def test_asteroids_have_no_nongrav_models(self):
        """Asteroid rows with zero A1/A2/A3 should have no nongrav model."""
        rows = SMALL_BODY_ORBITS[SMALL_BODY_ORBITS["desig"].isin(["4", "15", "433"])]
        assert all(model is None for model in make_nongravs_models(rows))

    def test_comet_row_can_make_nongrav_model(self):
        """A comet row with nonzero A terms should receive a nongrav model."""
        row = SMALL_BODY_ORBITS.loc[SMALL_BODY_ORBITS["desig"] == "C/2025 A6"].copy()
        row.loc[:, "A1"] = 1e-8

        models = make_nongravs_models(row)

        assert len(models) == 1
        assert models[0] is not None
        assert isinstance(models[0], kete.propagation.NonGravModel)


def _state_from_reference(utc_iso, target_key, desig=None):
    ref = reference_state(utc_iso, target_key)
    return kete.State(
        desig=target_key.replace("_", " ") if desig is None else desig,
        jd=_to_tdb_jd(utc_iso),
        pos=kete.Vector(ref["pos"], frame=kete.Frames.Ecliptic),
        vel=kete.Vector(ref["vel"], frame=kete.Frames.Ecliptic),
        center_id=10,
    )


def _ephemerides_from_reference_vectors(utc_iso):
    """Calculate light-delay-aware ephemerides from hard-coded Horizons vectors."""
    rows = {}

    sun_fov = _wide_fov(_sun_observer(_to_tdb_jd(utc_iso)))
    earth_state = _state_from_reference(utc_iso, "earth", desig="earth")
    earth_ss0 = kete.SimultaneousStates([earth_state], fov=sun_fov)
    earth_ss = kete.fov.fov_state_check(earth_ss0, [sun_fov], dt_limit=3.0)[0]
    eph, _ = calc_ephems(
        SMALL_BODY_ORBITS,
        [earth_ss],
        output=None,
        sort_by=None,
        add_jds=True,
        compact=False,
    )
    rows["earth"] = eph.iloc[0]

    observer_state = _state_from_reference(utc_iso, "earth", desig="geocenter")
    geo_fov = _wide_fov(observer_state)
    states = [
        _state_from_reference(utc_iso, key)
        for key in ("4", "15", "433", "C/2025 A6", "saturn_barycenter")
    ]
    geo_ss0 = kete.SimultaneousStates(states, fov=geo_fov)
    geo_ss = kete.fov.fov_state_check(geo_ss0, [geo_fov], dt_limit=3.0)[0]
    eph, _ = calc_ephems(
        SMALL_BODY_ORBITS,
        [geo_ss],
        output=None,
        sort_by=None,
        add_jds=True,
        compact=False,
    )

    for _, row in eph.iterrows():
        key = "saturn_barycenter" if row["desig"] == "saturn barycenter" else row["desig"]
        rows[key] = row

    return rows


def _assert_angle(got, expected, atol, label):
    assert _angle_delta_deg(got, expected) <= atol, (
        f"{label}: got {got}, expected {expected}"
    )


class TestReferenceVectorEphemerisRestoration:
    """Restore Horizons ephemerides from hard-coded Horizons vectors via kete."""

    def test_reference_times_are_utc_inputs(self):
        for utc_iso in REFERENCE_TIMES_UTC:
            expected = reference_tdb_jd(utc_iso)
            assert np.isclose(_to_tdb_jd(utc_iso), expected, atol=1e-10)

    @pytest.mark.parametrize("target_key", TARGET_KEYS)
    def test_reference_vectors_are_finite(self, target_key):
        for utc_iso in REFERENCE_TIMES_UTC:
            state = _state_from_reference(utc_iso, target_key)
            pos = np.array([state.pos[0], state.pos[1], state.pos[2]], dtype=float)
            vel = np.array([state.vel[0], state.vel[1], state.vel[2]], dtype=float)
            assert np.all(np.isfinite(pos))
            assert np.all(np.isfinite(vel))

    @pytest.mark.parametrize("target_key", PROPAGATION_EPHEM_TARGETS)
    def test_light_delay_ra_dec_matches_horizons_ephemerides(self, target_key):
        """Kete light-delay-aware RA/Dec matches Horizons to milli-arcsec scale."""
        for utc_iso in REFERENCE_TIMES_UTC:
            row = _ephemerides_from_reference_vectors(utc_iso)[target_key]
            ref = reference_horizons_ephemeris(target_key, utc_iso)

            _assert_angle(
                row["ra"],
                ref["ra"],
                QUERY_ANGLE_ATOL_DEG,
                f"{utc_iso} {target_key} ra",
            )
            _assert_angle(
                row["dec"],
                ref["dec"],
                QUERY_ANGLE_ATOL_DEG,
                f"{utc_iso} {target_key} dec",
            )

    @pytest.mark.parametrize("target_key", PROPAGATION_EPHEM_TARGETS)
    def test_light_delay_distances_match(self, target_key):
        """Kete light-delay-aware distances match Horizons references."""
        for utc_iso in REFERENCE_TIMES_UTC:
            row = _ephemerides_from_reference_vectors(utc_iso)[target_key]
            ref = reference_horizons_ephemeris(target_key, utc_iso)

            for col in ("r_hel", "r_obs"):
                assert np.isclose(row[col], ref[col], atol=DIST_ATOL_AU, rtol=1e-10), (
                    f"{utc_iso} {target_key} {col}: got {row[col]}, expected {ref[col]}"
                )

    @pytest.mark.xfail(
        reason=(
            "Kete fov_state_check uses two-body propagation for the final "
            "light-time correction from an observation-epoch vector; C/2025 A6 "
            "retains a 0.01-0.03 arcsec residual against Horizons."
        ),
        strict=True,
    )
    def test_comet_light_delay_restoration_is_sub_001_arcsec(self):
        """Guard the remaining comet light-delay restoration residual."""
        for utc_iso in REFERENCE_TIMES_UTC:
            row = _ephemerides_from_reference_vectors(utc_iso)["C/2025 A6"]
            ref = reference_horizons_ephemeris("C/2025 A6", utc_iso)
            _assert_angle(
                row["ra"],
                ref["ra"],
                QUERY_ANGLE_ATOL_DEG,
                f"{utc_iso} C/2025 A6 ra",
            )
            _assert_angle(
                row["dec"],
                ref["dec"],
                QUERY_ANGLE_ATOL_DEG,
                f"{utc_iso} C/2025 A6 dec",
            )


class TestCalcGeometriesInvariants:
    """Basic local invariants for calc_geometries."""

    def test_sky_motion_is_rate_norm(self):
        utc_iso = REFERENCE_TIMES_UTC[0]
        observer_state = _state_from_reference(utc_iso, "earth", desig="geocenter")
        target_state = _state_from_reference(utc_iso, "15", desig="15")
        fov = _wide_fov(observer_state)
        ss = kete.SimultaneousStates([target_state], fov=fov)

        geoms = calc_geometries(ss, rates_in_arcsec_per_min=True)
        expected = np.hypot(geoms["racosdec_rate"], geoms["dec_rate"])

        np.testing.assert_allclose(geoms["sky_motion"], expected, rtol=1e-12)

    def test_galactic_coords_present_and_correct(self):
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        utc_iso = REFERENCE_TIMES_UTC[0]
        observer_state = _state_from_reference(utc_iso, "earth", desig="geocenter")
        target_state = _state_from_reference(utc_iso, "15", desig="15")
        fov = _wide_fov(observer_state)
        ss = kete.SimultaneousStates([target_state], fov=fov)

        geoms = calc_geometries(ss)
        assert "glx_lon" in geoms
        assert "glx_lat" in geoms

        c = SkyCoord(ra=geoms["ra"] * u.deg, dec=geoms["dec"] * u.deg, frame="icrs")
        # galactic longitude uses [0, 360); handle wrap when comparing
        l_astropy = c.galactic.l.deg
        diff_lon = np.mod(geoms["glx_lon"] - l_astropy + 180, 360) - 180
        # atol=1e-5 (~0.04 arcsec) covers the systematic from ecl2eq using a
        # fixed IAU 1976 mean obliquity rather than astropy's full frame chain.
        np.testing.assert_allclose(diff_lon, 0.0, atol=1e-5)
        np.testing.assert_allclose(geoms["glx_lat"], c.galactic.b.deg, atol=1e-5)
