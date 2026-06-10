"""Tests for skyloc.query convenience helpers."""

import numpy as np
import pytest
import requests
from astropy.table import Table
from astropy.time import Time

from skyloc.jplhorizons import horizons_quick
import skyloc.query as query
from tests.ephemeris_reference import (
    HORIZONS_REFERENCES,
    HORIZONS_ANGLE_COLS,
    HORIZONS_DISTANCE_COLS,
    HORIZONS_MAG_COLS,
    HORIZONS_PA_COLS,
    HORIZONS_RATE_COLS,
    QUERY_ANGLE_ATOL_DEG,
    QUERY_DIST_ATOL_AU,
    QUERY_MAG_ATOL,
    QUERY_SKYMOT_ATOL_DEG,
    QUERY_VEL_ATOL_AU_PER_DAY,
    REFERENCE_TIMES_UTC,
    TARGET_KEYS,
    _angle_delta_deg,
    _to_tdb_jd,
    reference_horizons_ephemeris,
    reference_state,
)


HORIZONS_EPHEM_CONTEXT = {
    target_key: {
        "command": target_ref["horizons_command"],
        "location": "@10" if target_key == "earth" else "500",
        "description": target_ref["horizons_target"],
    }
    for target_key, target_ref in HORIZONS_REFERENCES["targets"].items()
}


def test_query_orb_by_desig_queries_only_requested_designations(monkeypatch):
    """query_orb_by_desig should query each requested desig and preserve order."""
    captured_desigs = []

    def fake_query_one(desig):
        captured_desigs.append(desig)
        return "ok", {"desig": desig, "ecc": 0.1}, None

    monkeypatch.setattr(query, "_query_single_orb_by_search", fake_query_one)

    got = query.query_orb_by_desig(
        ["1 Ceres", "4 Vesta"],
        drop_impacted=False,
        drop_unreliable=False,
    )
    assert captured_desigs == ["1 Ceres", "4 Vesta"]
    assert got["desig"].tolist() == ["1 Ceres", "4 Vesta"]
    assert got.index.tolist() == [0, 1]


def test_query_orb_by_desig_normalizes_nested_duplicate_inputs(monkeypatch):
    """Designation input is flattened, stripped, deduplicated, and ordered."""
    captured_desigs = []

    def fake_query_one(desig):
        captured_desigs.append(desig)
        return "ok", {"desig": desig, "ecc": 0.1}, None

    monkeypatch.setattr(query, "_query_single_orb_by_search", fake_query_one)

    got = query.query_orb_by_desig(
        [" 1 Ceres ", ["4 Vesta", "1 Ceres"], None],
        drop_impacted=False,
        drop_unreliable=False,
    )

    assert captured_desigs == ["1 Ceres", "4 Vesta"]
    assert got["desig"].tolist() == ["1 Ceres", "4 Vesta"]


def test_query_orb_by_desig_rejects_empty():
    """Empty designation inputs should fail fast."""
    with pytest.raises(ValueError, match="desigs"):
        query.query_orb_by_desig([])


def test_query_orb_by_desig_strict_raises_on_missing(monkeypatch):
    """strict=True should raise if any requested desig is not found."""

    def fake_query_one(desig):
        if desig == "missing":
            return "not_found", None, "not found"
        return "ok", {"desig": desig, "ecc": 0.1}, None

    monkeypatch.setattr(query, "_query_single_orb_by_search", fake_query_one)

    with pytest.raises(ValueError, match="Failed to query"):
        query.query_orb_by_desig(
            ["1 Ceres", "missing"],
            strict=True,
            drop_impacted=False,
            drop_unreliable=False,
        )


class TestHorizonsVectorQuery:
    """Test horizons_vector query functionality."""

    @pytest.mark.network
    @pytest.mark.slow
    def test_basic_query(self):
        """Basic vector query returns astropy Table."""
        from skyloc.jplhorizons import horizons_vector

        vec = horizons_vector(
            epochs=[2460600.5],
            obsid="399",
            location="@10",
        )
        assert isinstance(vec, Table)
        assert len(vec) == 1

    @pytest.mark.network
    @pytest.mark.slow
    def test_query_columns(self):
        """Query result has expected columns."""
        from skyloc.jplhorizons import horizons_vector

        vec = horizons_vector(
            epochs=[2460600.5],
            obsid="399",
            location="@10",
        )
        expected_cols = ["x", "y", "z", "vx", "vy", "vz", "range", "datetime_jd"]
        for col in expected_cols:
            assert col in vec.colnames, f"Missing column: {col}"

    @pytest.mark.network
    @pytest.mark.slow
    def test_multiple_epochs(self):
        """Query with multiple epochs returns correct number of rows."""
        from skyloc.jplhorizons import horizons_vector

        epochs = [2460600.5, 2460601.5, 2460602.5]
        vec = horizons_vector(
            epochs=epochs,
            obsid="399",
            location="@10",
        )
        assert len(vec) == 3

    @pytest.mark.network
    @pytest.mark.slow
    def test_return_arr(self):
        """return_arr=True returns position and velocity arrays."""
        from skyloc.jplhorizons import horizons_vector

        _, pos, vel = horizons_vector(
            epochs=[2460600.5],
            obsid="399",
            location="@10",
            return_arr=True,
        )
        assert isinstance(pos, np.ndarray)
        assert isinstance(vel, np.ndarray)
        assert pos.shape == (1, 3)
        assert vel.shape == (1, 3)

    @pytest.mark.network
    @pytest.mark.slow
    def test_spice_units(self):
        """spice_units=True returns km and km/s."""
        from skyloc.jplhorizons import horizons_vector

        vec = horizons_vector(
            epochs=[2460600.5],
            obsid="399",
            location="@10",
            spice_units=True,
        )
        assert vec["x"].unit.is_equivalent("km")
        x_val = float(vec["x"][0])
        assert abs(x_val) > 1e7

    @pytest.mark.network
    @pytest.mark.slow
    def test_invert(self):
        """invert=True negates position/velocity vectors."""
        from skyloc.jplhorizons import horizons_vector

        vec_normal = horizons_vector(epochs=[2460600.5], obsid="399", location="@10")
        vec_invert = horizons_vector(
            epochs=[2460600.5], obsid="399", location="@10", invert=True
        )

        np.testing.assert_allclose(
            np.array(vec_normal["x"]), -np.array(vec_invert["x"]), rtol=1e-10
        )

    @pytest.mark.network
    @pytest.mark.slow
    def test_earth_distance_from_sun(self):
        """Earth should be ~1 AU from Sun."""
        from skyloc.jplhorizons import horizons_vector

        vec = horizons_vector(
            epochs=[2460600.5],
            obsid="399",
            location="@10",
        )
        r = float(vec["range"][0])
        assert 0.98 < r < 1.02, f"Earth-Sun distance {r:.4f} AU unexpected"


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("target_key", TARGET_KEYS)
def test_horizons_vector_matches_hardcoded_geometric_reference(target_key):
    """Live Horizons geometric vectors match fixed Horizons vector references."""
    from skyloc.jplhorizons import horizons_vector

    context = HORIZONS_EPHEM_CONTEXT[target_key]
    tdb_jds = [_to_tdb_jd(utc_iso) for utc_iso in REFERENCE_TIMES_UTC]
    vec, pos, vel = horizons_vector(
        epochs=tdb_jds,
        obsid=context["command"],
        location="@10",
        aberrations="geometric",
        refplane="ecliptic",
        return_arr=True,
    )

    assert len(vec) == len(REFERENCE_TIMES_UTC)

    for idx, utc_iso in enumerate(REFERENCE_TIMES_UTC):
        ref = reference_state(utc_iso, target_key)
        np.testing.assert_allclose(
            pos[idx],
            ref["pos"],
            atol=QUERY_DIST_ATOL_AU,
            rtol=1e-10,
            err_msg=f"{utc_iso} {target_key} position",
        )
        np.testing.assert_allclose(
            vel[idx],
            ref["vel"],
            atol=QUERY_VEL_ATOL_AU_PER_DAY,
            rtol=1e-10,
            err_msg=f"{utc_iso} {target_key} velocity",
        )


class TestHorizonsVec2KeteState:
    """Test conversion of Horizons results to kete states."""

    @pytest.mark.network
    @pytest.mark.slow
    def test_basic_conversion(self):
        """Can convert Horizons result to kete states."""
        from skyloc.jplhorizons import horizons_vector, horizonsvec2ketestate
        import kete

        vec = horizons_vector(
            epochs=[2460600.5],
            obsid="399",
            location="@10",
        )
        states = horizonsvec2ketestate(vec)
        assert len(states) == 1
        assert isinstance(states[0], kete.State)

    @pytest.mark.network
    @pytest.mark.slow
    def test_state_properties(self):
        """Converted state has correct JD."""
        from skyloc.jplhorizons import horizons_vector, horizonsvec2ketestate

        jd = 2460600.5
        vec = horizons_vector(epochs=[jd], obsid="399", location="@10")
        states = horizonsvec2ketestate(vec)
        assert np.isclose(states[0].jd, jd, atol=1e-6)

    @pytest.mark.network
    @pytest.mark.slow
    def test_custom_desig(self):
        """Can set custom designations."""
        from skyloc.jplhorizons import horizons_vector, horizonsvec2ketestate

        vec = horizons_vector(epochs=[2460600.5], obsid="399", location="@10")
        states = horizonsvec2ketestate(vec, desigs=["Earth"])
        assert states[0].desig == "Earth"


class TestHorizonsQuick:
    """Test horizons_quick utility function."""

    def test_horizons_sun_target_pa_maps_to_sun_target_pa(self):
        """Horizons sunTargetPA maps directly to skyloc's sun_target_pa column."""
        from skyloc.jplhorizons import _horizons_ephemerides_to_compare

        eph = Table(
            {
                "datetime_jd": [2460600.5, 2460601.5, 2460602.5],
                "RA": [1.0, 2.0, 3.0],
                "DEC": [4.0, 5.0, 6.0],
                "alpha_true": [7.0, 8.0, 9.0],
                "RA_rate": [60.0, 120.0, 180.0],
                "DEC_rate": [-60.0, -120.0, -180.0],
                "r": [1.0, 1.1, 1.2],
                "delta": [2.0, 2.1, 2.2],
                "EclLon": [10.0, 20.0, 30.0],
                "EclLat": [0.1, 0.2, 0.3],
                "ObsEclLon": [40.0, 50.0, 60.0],
                "ObsEclLat": [0.4, 0.5, 0.6],
                "V": [15.0, 16.0, 17.0],
                "Tmag": [14.0, 15.0, 16.0],
                "Nmag": [17.0, 18.0, 19.0],
                "Sky_motion": [0.5, 0.6, 0.7],
                "Sky_mot_PA": [70.0, 80.0, 90.0],
                "sunTargetPA": [10.0, 190.0, 350.0],
            }
        )

        eph2compare = _horizons_ephemerides_to_compare(eph)

        assert "sun_target_pa" in eph2compare.columns
        np.testing.assert_allclose(eph2compare["ra"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(eph2compare["dec"], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(eph2compare["alpha"], [7.0, 8.0, 9.0])
        np.testing.assert_allclose(eph2compare["racosdec_rate"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(eph2compare["dec_rate"], [-1.0, -2.0, -3.0])
        np.testing.assert_allclose(eph2compare["r_hel"], [1.0, 1.1, 1.2])
        np.testing.assert_allclose(eph2compare["r_obs"], [2.0, 2.1, 2.2])
        np.testing.assert_allclose(eph2compare["hel_ecl_lon"], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(eph2compare["hel_ecl_lat"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(eph2compare["obs_ecl_lon"], [40.0, 50.0, 60.0])
        np.testing.assert_allclose(eph2compare["obs_ecl_lat"], [0.4, 0.5, 0.6])
        np.testing.assert_allclose(eph2compare["vmag"], [15.0, 16.0, 17.0])
        np.testing.assert_allclose(eph2compare["Tmag"], [14.0, 15.0, 16.0])
        np.testing.assert_allclose(eph2compare["Nmag"], [17.0, 18.0, 19.0])
        np.testing.assert_allclose(eph2compare["sky_motion"], [0.5, 0.6, 0.7])
        np.testing.assert_allclose(eph2compare["sky_motion_pa"], [70.0, 80.0, 90.0])
        np.testing.assert_allclose(eph2compare["sun_target_pa"], [10.0, -170.0, -10.0])

    @pytest.mark.network
    @pytest.mark.slow
    def test_basic_query(self):
        """horizons_quick returns ephemeris and comparison table."""
        eph, eph2compare = horizons_quick(
            objid="Ceres",
            epochs=[2460600.5],
            location="500",
        )
        assert isinstance(eph, Table)
        assert "ra" in eph2compare.columns
        assert "dec" in eph2compare.columns

    @pytest.mark.network
    @pytest.mark.slow
    def test_columns_renamed(self):
        """eph2compare has renamed columns for skyloc compatibility."""
        _, eph2compare = horizons_quick(
            objid="Ceres",
            epochs=[2460600.5],
            location="500",
        )
        expected_cols = ["ra", "dec", "r_hel", "r_obs", "alpha"]
        for col in expected_cols:
            assert col in eph2compare.columns, f"Missing column: {col}"

    @pytest.mark.network
    @pytest.mark.slow
    def test_rate_conversion(self):
        """Rates should be converted to arcsec/min."""
        _, eph2compare = horizons_quick(
            objid="Ceres",
            epochs=[2460600.5],
            location="500",
        )
        assert "racosdec_rate" in eph2compare.columns
        assert "dec_rate" in eph2compare.columns


def _query_horizons_ephem_or_skip(target_key):
    context = HORIZONS_EPHEM_CONTEXT[target_key]
    utc_jds = [Time(t, scale="utc", format="isot").jd for t in REFERENCE_TIMES_UTC]
    try:
        _, eph = horizons_quick(
            context["command"],
            utc_jds,
            location=context["location"],
            auto_choose_recordnum=True,
        )
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"Horizons network query unavailable: {exc}")
    return eph


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("target_key", TARGET_KEYS)
def test_horizons_ephem_matches_hardcoded_reference(target_key):
    """Live Horizons ephemerides match fixed Horizons ephemerides references."""
    eph = _query_horizons_ephem_or_skip(target_key)
    assert len(eph) == len(REFERENCE_TIMES_UTC)

    for utc_iso, (_, row) in zip(REFERENCE_TIMES_UTC, eph.iterrows()):
        ref = reference_horizons_ephemeris(target_key, utc_iso)

        for col, expected in ref.items():
            got = row[col]
            msg = f"{utc_iso} {target_key} {col}: got {got}, expected {expected}"

            if col in HORIZONS_ANGLE_COLS:
                assert _angle_delta_deg(got, expected) <= QUERY_ANGLE_ATOL_DEG, msg
            elif col in HORIZONS_DISTANCE_COLS:
                assert np.isclose(got, expected, atol=QUERY_DIST_ATOL_AU, rtol=1e-10), msg
            elif col in HORIZONS_RATE_COLS:
                assert abs(got - expected) <= QUERY_SKYMOT_ATOL_DEG, msg
            elif col in HORIZONS_PA_COLS:
                assert _angle_delta_deg(got, expected) <= QUERY_ANGLE_ATOL_DEG, msg
            elif col in HORIZONS_MAG_COLS:
                assert np.isclose(got, expected, atol=QUERY_MAG_ATOL), msg
            else:
                raise AssertionError(f"Unhandled Horizons comparison column: {col}")
