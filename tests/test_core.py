"""Tests for skyloc.core module - SSOLocator and related functions."""

import numpy as np
import pandas as pd
import pytest
import kete
import skyloc.core as core_mod
from skyloc.core import SSOLocator, SpiceLocator, StarLocator, calc_ephems
from skyloc.keteutils import (
    KETE_ASTEROIDS_PHYSICS,
    KETE_PLANETS,
    get_default_spice_targets,
    get_spice_asteroid_desigs,
    get_spice_asteroid_names,
    is_spice_resolvable,
)
from skyloc.keteutils.fov import FOVCollection
from tests.ephemeris_reference import (
    DIST_ATOL_AU,
    QUERY_ANGLE_ATOL_DEG,
    QUERY_VEL_ATOL_AU_PER_DAY,
    REFERENCE_TIMES_UTC,
    SMALL_BODY_ORBITS,
    _angle_delta_deg,
    _sun_observer,
    _to_tdb_jd,
    _wide_fov,
    reference_horizons_ephemeris,
    reference_state,
)


CORE_STRICT_EPHEM_TARGETS = (
    "earth",
    "4",
    "15",
    "433",
    "saturn_barycenter",
)


def _state_from_reference(utc_iso, target_key, desig=None):
    ref = reference_state(utc_iso, target_key)
    return kete.State(
        desig=target_key.replace("_", " ") if desig is None else desig,
        jd=_to_tdb_jd(utc_iso),
        pos=kete.Vector(ref["pos"], frame=kete.Frames.Ecliptic),
        vel=kete.Vector(ref["vel"], frame=kete.Frames.Ecliptic),
        center_id=10,
    )


def _strict_core_ephemerides_from_reference_vectors(utc_iso):
    """Calculate core SSOLocator ephemerides from hard-coded Horizons vectors."""
    rows = {}
    jd = _to_tdb_jd(utc_iso)

    sun_fov = _wide_fov(_sun_observer(jd))
    earth_loc = SSOLocator(
        fovs=sun_fov,
        orb=SMALL_BODY_ORBITS,
        non_gravs=False,
        drop_major_asteroids=False,
    )
    earth_loc.states_propagated_jd0 = [
        _state_from_reference(utc_iso, "earth", desig="earth")
    ]
    earth_loc.fov_state_check(use_spice_for_loaded=False)
    earth_loc.calc_ephems(
        sort_by=None,
        compact=False,
        add_obsid=False,
        drop_obsindex=False,
        add_jds=True,
    )
    rows["earth"] = earth_loc.eph.iloc[0]

    observer_state = _state_from_reference(utc_iso, "earth", desig="geocenter")
    geo_loc = SSOLocator(
        fovs=_wide_fov(observer_state),
        orb=SMALL_BODY_ORBITS,
        non_gravs=False,
        drop_major_asteroids=False,
    )
    geo_loc.states_propagated_jd0 = [
        _state_from_reference(utc_iso, key)
        for key in ("4", "15", "433", "C/2025 A6", "saturn_barycenter")
    ]
    geo_loc.fov_state_check(use_spice_for_loaded=False)
    geo_loc.calc_ephems(
        sort_by=None,
        compact=False,
        add_obsid=False,
        drop_obsindex=False,
        add_jds=True,
    )
    for _, row in geo_loc.eph.iterrows():
        key = "saturn_barycenter" if row["desig"] == "saturn barycenter" else row["desig"]
        rows[key] = row

    return rows


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
    """Test SSOLocator propagation methods."""

    def test_propagate_n_body_sets_jd0(
        self, sample_orb_asteroid, sample_fovlist, sample_jd_tdb
    ):
        """Propagation sets jd0 correctly."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        loc.propagate_n_body(jd0=sample_jd_tdb, suppress_errors=True)
        assert np.isclose(loc.jd0, sample_jd_tdb, atol=1e-10)

    def test_propagate_n_body_state_count(
        self, sample_orb_asteroid, sample_fovlist, sample_jd_tdb
    ):
        """Propagation creates correct number of states."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        loc.propagate_n_body(jd0=sample_jd_tdb, suppress_errors=True)
        assert loc.states_propagated_jd0 is not None
        assert len(loc.states_propagated_jd0) == 1

    def test_propagate_with_mean_jd(
        self, sample_orb_asteroid, sample_fovlist, sample_jd_tdb
    ):
        """Propagation with callable jd0 (np.mean) uses mean of FOV JDs."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        loc.propagate_n_body(jd0=np.mean, suppress_errors=True)
        expected_jd = sample_jd_tdb + 1.0
        assert np.isclose(loc.jd0, expected_jd, atol=1e-10)

    def test_propagated_state_desig(
        self, sample_orb_asteroid, sample_fovlist, sample_jd_tdb
    ):
        """Propagated state has correct designation."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        loc.propagate_n_body(jd0=sample_jd_tdb, suppress_errors=True)
        assert loc.states_propagated_jd0[0].desig == "15 Eunomia"

    def test_propagated_state_jd(
        self, sample_orb_asteroid, sample_fovlist, sample_jd_tdb
    ):
        """Propagated state has correct JD."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_asteroid)
        loc.propagate_n_body(jd0=sample_jd_tdb, suppress_errors=True)
        assert np.isclose(loc.states_propagated_jd0[0].jd, sample_jd_tdb, atol=1e-10)

    @pytest.mark.parametrize(
        "target_key",
        ("15", "433", "C/2025 A6"),
    )
    def test_propagated_reference_orbits_match_horizons_vectors(self, target_key):
        """Reference orbit rows propagate back to their hard-coded vectors."""
        for utc_iso in REFERENCE_TIMES_UTC:
            orb = SMALL_BODY_ORBITS.loc[SMALL_BODY_ORBITS["desig"] == target_key]
            loc = SSOLocator(
                fovs=_wide_fov(_sun_observer(_to_tdb_jd(utc_iso))),
                orb=orb,
                non_gravs=target_key == "C/2025 A6",
                drop_major_asteroids=False,
            )
            loc.propagate_n_body(jd0=_to_tdb_jd(utc_iso), suppress_errors=True)
            state = loc.states_propagated_jd0[0]
            pos = np.array([state.pos[0], state.pos[1], state.pos[2]])
            vel = np.array([state.vel[0], state.vel[1], state.vel[2]])
            ref = reference_state(utc_iso, target_key)

            np.testing.assert_allclose(
                pos,
                ref["pos"],
                atol=DIST_ATOL_AU,
                rtol=1e-10,
                err_msg=f"{utc_iso} {target_key} position",
            )
            np.testing.assert_allclose(
                vel,
                ref["vel"],
                atol=QUERY_VEL_ATOL_AU_PER_DAY,
                rtol=1e-10,
                err_msg=f"{utc_iso} {target_key} velocity",
            )

    def test_simstates_init_sets_prepropagated_states(self):
        """simstates init path accepts already propagated states."""
        utc_iso = REFERENCE_TIMES_UTC[0]
        fov = _wide_fov(_sun_observer(_to_tdb_jd(utc_iso)))
        state = _state_from_reference(utc_iso, "15", desig="15")
        simstates = kete.SimultaneousStates([state], fov=fov)

        loc = SSOLocator(fovs=fov, orb=SMALL_BODY_ORBITS, simstates=simstates)

        assert np.isclose(loc.jd0, simstates.jd, atol=1e-10)
        assert len(loc.states_propagated_jd0) == 1
        assert loc.states_propagated_jd0[0].desig == "15"


@pytest.mark.slow
@pytest.mark.integration
class TestSSOLocatorReferenceEphemerides:
    """Strict SSOLocator ephemeris checks against hard-coded Horizons data."""

    @pytest.mark.parametrize("target_key", CORE_STRICT_EPHEM_TARGETS)
    def test_core_ra_dec_matches_horizons_reference(self, target_key):
        """Core fov_state_check/calc_ephems matches Horizons RA/Dec strictly."""
        for utc_iso in REFERENCE_TIMES_UTC:
            row = _strict_core_ephemerides_from_reference_vectors(utc_iso)[target_key]
            ref = reference_horizons_ephemeris(target_key, utc_iso)

            assert _angle_delta_deg(row["ra"], ref["ra"]) <= QUERY_ANGLE_ATOL_DEG
            assert _angle_delta_deg(row["dec"], ref["dec"]) <= QUERY_ANGLE_ATOL_DEG

    @pytest.mark.parametrize("target_key", CORE_STRICT_EPHEM_TARGETS)
    def test_core_distances_match_horizons_reference(self, target_key):
        """Core fov_state_check/calc_ephems distances match Horizons strictly."""
        for utc_iso in REFERENCE_TIMES_UTC:
            row = _strict_core_ephemerides_from_reference_vectors(utc_iso)[target_key]
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
    def test_core_comet_light_delay_restoration_is_sub_001_arcsec(self):
        """Core-level guard for the remaining comet light-delay residual."""
        for utc_iso in REFERENCE_TIMES_UTC:
            row = _strict_core_ephemerides_from_reference_vectors(utc_iso)["C/2025 A6"]
            ref = reference_horizons_ephemeris("C/2025 A6", utc_iso)

            assert _angle_delta_deg(row["ra"], ref["ra"]) <= QUERY_ANGLE_ATOL_DEG
            assert _angle_delta_deg(row["dec"], ref["dec"]) <= QUERY_ANGLE_ATOL_DEG

    @pytest.mark.parametrize("add_obsid", [False, True])
    @pytest.mark.parametrize("drop_obsindex", [False, True])
    @pytest.mark.parametrize("add_jds", [False, True])
    def test_calc_ephems_output_column_options(self, add_obsid, drop_obsindex, add_jds):
        """calc_ephems honors output-shape options for all option combinations."""
        utc_iso = REFERENCE_TIMES_UTC[0]
        observer_state = _state_from_reference(utc_iso, "earth", desig="geocenter")
        loc = SSOLocator(
            fovs=_wide_fov(observer_state),
            orb=SMALL_BODY_ORBITS,
            non_gravs=False,
            drop_major_asteroids=False,
        )
        loc.states_propagated_jd0 = [
            _state_from_reference(utc_iso, "15", desig="15")
        ]
        loc.fov_state_check(use_spice_for_loaded=False)
        loc.calc_ephems(
            sort_by=None,
            compact=False,
            add_obsid=add_obsid,
            drop_obsindex=drop_obsindex,
            add_jds=add_jds,
        )

        assert ("obsid" in loc.eph.columns) is add_obsid
        assert ("obsindex" not in loc.eph.columns) is drop_obsindex
        assert ("jd_tdb" in loc.eph.columns) is add_jds
        assert ("jd_utc" in loc.eph.columns) is add_jds


@pytest.mark.integration
class TestSSOLocatorSpiceReplacement:
    """Test SPICE replacement for loaded asteroids.

    Default (drop_major_asteroids=True): the 5 loaded asteroids are separated
    into _spice_asteroid_orb and re-inserted via fov_spice_check in fov_state_check.

    With drop_major_asteroids=False: raw n-body is used; self-collision may yield NaN.
    """

    def test_drop_major_true_separates_vesta(
        self, sample_orb_vesta, sample_fovlist
    ):
        """With drop_major_asteroids=True, Vesta is in _spice_asteroid_orb not orb."""
        loc = SSOLocator(
            fovs=sample_fovlist,
            orb=sample_orb_vesta,
            drop_major_asteroids=True,
        )
        assert len(loc.orb) == 0
        assert len(loc._spice_asteroid_orb) == 1
        assert loc._spice_asteroid_orb.iloc[0]["desig"] == "4"

    def test_drop_major_false_keeps_vesta_in_orb(
        self, sample_orb_vesta, sample_fovlist
    ):
        """With drop_major_asteroids=False, Vesta stays in orb."""
        loc = SSOLocator(
            fovs=sample_fovlist,
            orb=sample_orb_vesta,
            drop_major_asteroids=False,
        )
        assert len(loc.orb) == 1
        assert len(loc._spice_asteroid_orb) == 1  # still tracked

    def test_drop_major_false_propagation_may_nan(
        self, sample_orb_vesta, sample_fovlist, sample_jd_tdb
    ):
        """With drop_major_asteroids=False + include_asteroids=True, kete may give NaN.

        This is intentional: the user wants to see raw self-collision output.
        The test only checks that propagation runs without exception.
        """
        loc = SSOLocator(
            fovs=sample_fovlist,
            orb=sample_orb_vesta,
            drop_major_asteroids=False,
        )
        loc.propagate_n_body(jd0=sample_jd_tdb, include_asteroids=True)
        assert len(loc.states_propagated_jd0) == 1

    def test_drop_major_true_propagation_no_vesta_in_states(
        self, sample_orb_vesta, sample_fovlist, sample_jd_tdb
    ):
        """With drop_major_asteroids=True, Vesta is excluded from n-body states."""
        loc = SSOLocator(
            fovs=sample_fovlist,
            orb=sample_orb_vesta,
            drop_major_asteroids=True,
        )
        loc.propagate_n_body(jd0=sample_jd_tdb, include_asteroids=True)
        assert len(loc.states_propagated_jd0) == 0

    def test_spice_asteroid_targets_use_spkid(self, sample_orb_vesta, sample_fovlist):
        """_spice_asteroid_spkids should use SPK IDs, not SBDB numbers."""
        loc = SSOLocator(fovs=sample_fovlist, orb=sample_orb_vesta)
        assert loc._spice_asteroid_spkids == ["20000004"]

    def test_use_spice_for_loaded_checks_dropped_asteroids_not_planets(
        self, sample_orb_vesta, sample_fovlist, monkeypatch
    ):
        """use_spice_for_loaded should restore dropped loaded asteroids only."""
        called_targets = []

        def fake_fov_state_check(*args, **kwargs):
            return []

        def fake_fov_spice_check(targets, fovs):
            called_targets.append(list(targets))
            return []

        monkeypatch.setattr(core_mod.kete.fov, "fov_state_check", fake_fov_state_check)
        monkeypatch.setattr(core_mod.kete.fov, "fov_spice_check", fake_fov_spice_check)

        loc = SSOLocator(
            fovs=sample_fovlist,
            orb=sample_orb_vesta,
            drop_major_asteroids=True,
        )
        assert len(loc.orb) == 0

        loc.states_propagated_jd0 = []
        loc.fov_state_check(use_spice_for_loaded=True)

        assert called_targets == [["20000004"]]

    @pytest.mark.parametrize("include_asteroids", [False, True])
    @pytest.mark.parametrize("use_spice_for_loaded", [False, True])
    @pytest.mark.parametrize("include_planets", [False, True])
    @pytest.mark.parametrize("spice_targets", [None, ["599"]])
    def test_fov_state_check_option_combinations_are_forwarded(
        self,
        sample_orb_vesta,
        sample_fovlist,
        monkeypatch,
        include_asteroids,
        use_spice_for_loaded,
        include_planets,
        spice_targets,
    ):
        """fov_state_check forwards n-body and SPICE option combinations."""
        state_calls = []
        spice_calls = []

        def fake_fov_state_check(states, fovs, **kwargs):
            state_calls.append(kwargs)
            return []

        def fake_fov_spice_check(targets, fovs):
            spice_calls.append(list(targets))
            return []

        monkeypatch.setattr(core_mod.kete.fov, "fov_state_check", fake_fov_state_check)
        monkeypatch.setattr(core_mod.kete.fov, "fov_spice_check", fake_fov_spice_check)

        loc = SSOLocator(
            fovs=sample_fovlist,
            orb=sample_orb_vesta,
            drop_major_asteroids=True,
        )
        loc.states_propagated_jd0 = [_state_from_reference(REFERENCE_TIMES_UTC[0], "15")]

        loc.fov_state_check(
            include_asteroids=include_asteroids,
            use_spice_for_loaded=use_spice_for_loaded,
            include_planets=include_planets,
            spice_targets=spice_targets,
        )

        assert state_calls == [
            {"dt_limit": 3.0, "include_asteroids": include_asteroids}
        ]

        expected_targets = []
        if use_spice_for_loaded:
            expected_targets.append("20000004")
        if include_planets:
            expected_targets.extend(
                get_default_spice_targets(
                    include_planets=True,
                    include_asteroids=False,
                )
            )
        if spice_targets is not None:
            expected_targets.extend(spice_targets)
        expected_targets = list(dict.fromkeys(expected_targets))

        if expected_targets:
            assert spice_calls == [expected_targets]
        else:
            assert spice_calls == []

    def test_spice_asteroid_infov_mask_set_after_fov_check(
        self, sample_orb_vesta, sample_fovlist, monkeypatch
    ):
        """fov_state_check sets spice_asteroid_infov_mask."""
        monkeypatch.setattr(core_mod.kete.fov, "fov_state_check", lambda *a, **kw: [])
        monkeypatch.setattr(core_mod.kete.fov, "fov_spice_check", lambda *a, **kw: [])

        loc = SSOLocator(
            fovs=sample_fovlist,
            orb=sample_orb_vesta,
            drop_major_asteroids=True,
        )
        loc.states_propagated_jd0 = []
        loc.fov_state_check(use_spice_for_loaded=True)

        assert hasattr(loc, "spice_asteroid_infov_mask")
        assert len(loc.spice_asteroid_infov_mask) == 1

    def test_full_pipeline_drop_major_true(
        self, sample_orb_vesta, sample_fov_earth_2024, monkeypatch
    ):
        """With drop_major_asteroids=True, Vesta flows through _spice_asteroid_orb."""
        loc = SSOLocator(
            fovs=sample_fov_earth_2024, orb=sample_orb_vesta, drop_major_asteroids=True
        )
        jd = sample_fov_earth_2024.observer.jd
        loc.propagate_n_body(jd0=jd, include_asteroids=False)
        assert len(loc.states_propagated_jd0) == 0

        loc.fov_state_check(include_asteroids=False, use_spice_for_loaded=True)
        assert hasattr(loc, "spice_asteroid_infov_mask")


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


class TestSpiceConfig:
    """Test SPICE target configuration functions."""

    def test_get_default_spice_targets_all(self):
        """Default targets include planets and asteroids."""
        targets = get_default_spice_targets(
            include_planets=True, include_asteroids=True
        )
        assert "499" in targets
        assert "599" in targets
        assert "ceres" in targets
        assert "vesta" in targets
        assert len(targets) == 14

    def test_get_default_spice_targets_planets_only(self):
        """Can get planets only."""
        targets = get_default_spice_targets(
            include_planets=True, include_asteroids=False
        )
        assert "599" in targets
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
        assert "1" in desigs
        assert "2" in desigs
        assert "4" in desigs
        assert "10" in desigs
        assert "704" in desigs

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


class TestSpiceLocator:
    """Test SpiceLocator class."""

    def test_init_default_targets(self, sample_fovlist):
        """SpiceLocator initializes with default targets."""
        sl = SpiceLocator(sample_fovlist)
        assert len(sl.targets) == 14
        assert "599" in sl.targets
        assert "ceres" in sl.targets

    def test_init_custom_targets(self, sample_fovlist):
        """SpiceLocator accepts custom targets."""
        sl = SpiceLocator(sample_fovlist, targets=["mars", "jupiter"])
        assert sl.targets == ["mars", "jupiter"]

    def test_init_planets_only(self, sample_fovlist):
        """SpiceLocator can be planets only."""
        sl = SpiceLocator(sample_fovlist, include_planets=True, include_asteroids=False)
        assert "599" in sl.targets
        assert "ceres" not in sl.targets

    def test_init_asteroids_only(self, sample_fovlist):
        """SpiceLocator can be asteroids only."""
        sl = SpiceLocator(sample_fovlist, include_planets=False, include_asteroids=True)
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
        loaded_names = [name for name, _ in sl.loaded]
        assert "Earth" in loaded_names or "earth" in loaded_names

    @pytest.mark.integration
    def test_fov_spice_check_runs(self, sample_fov_earth_2024):
        """fov_spice_check executes without error."""
        sl = SpiceLocator([sample_fov_earth_2024], targets=["599"])
        sl.fov_spice_check()
        assert sl.fov_check_fov2objs is not None
        assert sl.fov_check_objids is not None


class TestStarLocator:
    """Test StarLocator class."""

    @pytest.fixture
    def sample_stars(self):
        """Sample star catalog with known positions."""
        return pd.DataFrame(
            {
                "desig": ["Vega", "Sirius", "Arcturus"],
                "ra": [279.2347, 101.2875, 213.9153],
                "dec": [38.7837, -16.7161, 19.1824],
                "vmag": [0.03, -1.46, -0.05],
            }
        )

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
        sources = pd.DataFrame(
            {
                "ra": [0.0, 90.0, 180.0],
                "dec": [0.0, 45.0, -45.0],
            }
        )
        sl = StarLocator(sample_fovlist)
        sl.set_sources(sources)
        assert "desig" in sl.sources.columns
        assert sl.sources["desig"].iloc[0] == "star_0"

    def test_set_sources_missing_columns(self, sample_fovlist):
        """Raises error if required columns missing."""
        sources = pd.DataFrame({"ra": [0.0]})
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


class TestSpiceUtils:
    """Test spice utility functions."""

    def test_get_states_single_jd(self):
        """get_states works with single JD."""
        from skyloc.keteutils.spice import get_states

        states = get_states("earth", _to_tdb_jd(REFERENCE_TIMES_UTC[0]), center="sun")
        assert len(states) == 1
        assert states[0].desig.lower() == "earth"

    def test_get_states_multiple_jds(self):
        """get_states works with multiple JDs."""
        from skyloc.keteutils.spice import get_states

        jds = [_to_tdb_jd(utc_iso) for utc_iso in REFERENCE_TIMES_UTC]
        states = get_states("earth", jds, center="sun")
        assert len(states) == len(REFERENCE_TIMES_UTC)

    def test_get_state_arrays(self):
        """get_state_arrays returns numpy arrays."""
        from skyloc.keteutils.spice import get_state_arrays

        jds = [_to_tdb_jd(utc_iso) for utc_iso in REFERENCE_TIMES_UTC[:2]]
        pos, vel = get_state_arrays("earth", jds, center="sun")
        assert pos.shape == (2, 3)
        assert vel.shape == (2, 3)

    def test_is_spk_loaded_earth(self):
        """Earth should be loaded by default."""
        from skyloc.keteutils.spice import is_spk_loaded

        is_loaded, name = is_spk_loaded("earth")
        assert is_loaded is True
        assert name is not None

    def test_is_spk_loaded_horizons_format(self):
        """Handles Horizons "500@" format."""
        from skyloc.keteutils.spice import is_spk_loaded

        is_loaded, name = is_spk_loaded("500@399")
        assert is_loaded is True
        assert name == "399"
