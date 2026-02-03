"""Tests for skyloc.jplhorizons module - Horizons query functionality.

These tests verify the query mechanics work correctly. They do NOT compare
query results with kete calculations (that's done in test_propagate.py).

Note: Tests marked @pytest.mark.network require internet access and may
be slow due to API calls.
"""

import numpy as np
import pytest
from astropy.table import Table


class TestHorizonsVectorQuery:
    """Test horizons_vector query functionality."""

    @pytest.mark.network
    @pytest.mark.slow
    def test_basic_query(self):
        """Basic vector query returns astropy Table."""
        from skyloc.jplhorizons import horizons_vector

        # Query Earth at a single epoch
        vec = horizons_vector(
            epochs=[2460600.5],
            obsid="399",  # Earth
            location="@10",  # Sun center
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

        vec, pos, vel = horizons_vector(
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
        # x should be in km (order of 1e8)
        assert vec["x"].unit.is_equivalent("km")
        x_val = float(vec["x"][0])
        assert abs(x_val) > 1e7  # Earth-Sun is ~1.5e8 km

    @pytest.mark.network
    @pytest.mark.slow
    def test_invert(self):
        """invert=True negates position/velocity vectors."""
        from skyloc.jplhorizons import horizons_vector

        vec_normal = horizons_vector(epochs=[2460600.5], obsid="399", location="@10")
        vec_invert = horizons_vector(
            epochs=[2460600.5], obsid="399", location="@10", invert=True
        )

        # x values should be negated
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

    @pytest.mark.network
    @pytest.mark.slow
    def test_basic_query(self):
        """horizons_quick returns ephemeris and comparison table."""
        from skyloc.jplhorizons import horizons_quick

        eph, eph2compare = horizons_quick(
            objid="Ceres",
            epochs=[2460600.5],
            location="500",  # geocentric
        )
        assert isinstance(eph, Table)
        assert "ra" in eph2compare.columns
        assert "dec" in eph2compare.columns

    @pytest.mark.network
    @pytest.mark.slow
    def test_columns_renamed(self):
        """eph2compare has renamed columns for skyloc compatibility."""
        from skyloc.jplhorizons import horizons_quick

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
        from skyloc.jplhorizons import horizons_quick

        _, eph2compare = horizons_quick(
            objid="Ceres",
            epochs=[2460600.5],
            location="500",
        )
        # Original Horizons is arcsec/hr, should be divided by 60
        assert "racosdec_rate" in eph2compare.columns
        assert "dec_rate" in eph2compare.columns
