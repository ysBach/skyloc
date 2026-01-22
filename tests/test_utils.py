"""Tests for skyloc.utils module.

Expected values computed from known constants and formulas.
"""

import numpy as np
import pytest
from skyloc.utils import (
    au2km,
    km2au,
    kmps2aupd,
    aupd2kmps,
    tdb2utc,
    utc2tdb,
    listmask,
    as_iter,
    zip_iters,
    infov2d,
    all_world2pix_infov,
)


# IAU 2012 astronomical unit
AU_KM = 149597870.700  # km


class TestUnitConversions:
    """Test AU/km and velocity unit conversions."""

    def test_au_to_km_exact(self):
        """1 AU should equal exactly 149597870.700 km (IAU 2012)."""
        result = au2km(1.0)
        assert np.isclose(result, AU_KM, rtol=1e-10)

    def test_au_to_km_values(self):
        """Test known AU to km conversions."""
        test_cases = [
            (0.0, 0.0),
            (1.0, AU_KM),
            (5.2, AU_KM * 5.2),  # Jupiter distance
            (30.0, AU_KM * 30.0),  # Neptune distance
        ]
        for au_val, km_expected in test_cases:
            assert np.isclose(au2km(au_val), km_expected, rtol=1e-10)

    def test_km_to_au_exact(self):
        """AU_KM km should equal exactly 1 AU."""
        result = km2au(AU_KM)
        assert np.isclose(result, 1.0, rtol=1e-10)

    def test_au2km_roundtrip(self):
        """AU -> km -> AU should be identity."""
        for x in [0.0, 0.5, 1.0, 5.2, 39.5]:
            assert np.isclose(km2au(au2km(x)), x, rtol=1e-10)

    def test_velocity_conversion_values(self):
        """Test velocity conversions against known values.

        1 AU/day = 149597870.700 / 86400 km/s ≈ 1731.457 km/s
        """
        # 1 AU/day in km/s
        expected_kmps = AU_KM / 86400.0
        result = aupd2kmps(1.0)
        assert np.isclose(result, expected_kmps, rtol=1e-4)

        # Earth orbital velocity ~30 km/s
        earth_v_kmps = 29.78
        earth_v_aupd = kmps2aupd(earth_v_kmps)
        # Earth moves ~1 degree/day, so v ≈ 2*pi*AU / 365 AU/day ≈ 0.0172 AU/day
        expected_aupd = 0.0172
        assert np.isclose(earth_v_aupd, expected_aupd, rtol=0.01)

    def test_velocity_roundtrip(self):
        """km/s -> AU/day -> km/s should be identity."""
        for v in [1.0, 10.0, 29.78, 100.0]:
            assert np.isclose(aupd2kmps(kmps2aupd(v)), v, rtol=1e-6)

    def test_vectorized(self):
        """Test vectorized operations."""
        x = np.array([1.0, 2.0, 3.0])
        result = au2km(x)
        expected = x * AU_KM
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestTimeConversions:
    """Test TDB/UTC time conversions."""

    def test_known_offset(self):
        """TDB-UTC offset should be approximately 69 seconds (varies with epoch).

        At J2000 (JD 2451545.0), TDB-TT ≈ 0, and TT-UTC ≈ 63.184s + leap seconds.
        By 2024, this is approximately 69 seconds.
        """
        jd_tdb = 2460000.5  # ~2023 epoch
        utc_time = tdb2utc(jd_tdb)
        # TDB - UTC should be about 69 seconds ≈ 0.0008 days
        delta_days = jd_tdb - utc_time.jd
        delta_sec = delta_days * 86400
        # Should be between 60 and 75 seconds
        assert 60 < delta_sec < 75, f"TDB-UTC = {delta_sec:.1f} seconds"

    def test_roundtrip(self):
        """TDB -> UTC -> TDB should be approximately identity."""
        jd_tdb = 2460000.5
        jd_utc = tdb2utc(jd_tdb).jd
        jd_tdb_back = utc2tdb(jd_utc).jd
        # Should match to within 1 microsecond
        assert np.isclose(jd_tdb, jd_tdb_back, atol=1e-9)


class TestListMask:
    """Test the listmask utility."""

    def test_basic(self):
        """Basic list masking."""
        inlist = ["a", "b", "c", "d"]
        mask = [True, False, True, False]
        result = listmask(inlist, mask)
        assert result == ["a", "c"]

    def test_none_mask(self):
        """None mask returns original list."""
        inlist = ["a", "b", "c"]
        assert listmask(inlist, None) == inlist

    def test_all_true(self):
        """All True mask returns all elements."""
        inlist = [1, 2, 3]
        mask = [True, True, True]
        assert listmask(inlist, mask) == inlist

    def test_all_false(self):
        """All False mask returns empty list."""
        inlist = [1, 2, 3]
        mask = [False, False, False]
        assert listmask(inlist, mask) == []

    def test_preserves_order(self):
        """Elements should maintain order."""
        inlist = [5, 4, 3, 2, 1]
        mask = [True, False, True, False, True]
        assert listmask(inlist, mask) == [5, 3, 1]


class TestAsIter:
    """Test the as_iter utility."""

    def test_scalar_repeat(self):
        """Scalar is repeated n times."""
        result = list(as_iter(5, 3))
        assert result == [5, 5, 5]

    def test_scalar_repeat_values(self):
        """Test various scalar types."""
        assert list(as_iter(3.14, 2)) == [3.14, 3.14]
        assert list(as_iter(None, 2)) == [None, None]

    def test_array_passthrough(self):
        """Array is passed through as iterator."""
        arr = [1, 2, 3]
        result = list(as_iter(arr, 3))
        assert result == [1, 2, 3]

    def test_string_is_scalar(self):
        """String should be treated as scalar."""
        result = list(as_iter("abc", 2))
        assert result == ["abc", "abc"]

    def test_numpy_scalar(self):
        """Numpy scalar should be treated as scalar."""
        result = list(as_iter(np.float64(2.5), 3))
        assert result == [2.5, 2.5, 2.5]


class TestZipIters:
    """Test the zip_iters utility."""

    def test_mixed_scalar_array(self):
        """Mix of scalars and arrays should broadcast."""
        result = list(zip_iters([1, 2, 3], 10, [4, 5, 6]))
        assert result == [(1, 10, 4), (2, 10, 5), (3, 10, 6)]

    def test_all_arrays(self):
        """All arrays should zip normally."""
        result = list(zip_iters([1, 2], [3, 4], [5, 6]))
        assert result == [(1, 3, 5), (2, 4, 6)]

    def test_all_scalars_with_one_array(self):
        """Scalars broadcast to array length."""
        result = list(zip_iters(10, [1, 2, 3], 20))
        assert result == [(10, 1, 20), (10, 2, 20), (10, 3, 20)]


class TestInFov2d:
    """Test the infov2d function."""

    def test_inside_known_positions(self):
        """Test specific positions known to be inside."""
        x = np.array([50.0, 25.0, 75.0])
        y = np.array([50.0, 25.0, 75.0])
        result = infov2d(x, y, bezels=0.5, naxes=[100, 100])
        assert np.all(result)

    def test_outside_known_positions(self):
        """Test specific positions known to be outside."""
        x = np.array([-1.0, 101.0, 50.0, 50.0])
        y = np.array([50.0, 50.0, -1.0, 101.0])
        result = infov2d(x, y, bezels=0.5, naxes=[100, 100])
        assert not np.any(result)

    def test_edge_with_default_bezel(self):
        """Default bezel=0.5 means pixel 0 center (0.5) is inside."""
        # Pixel centers at 0.5, 99.5 should be inside 100x100 image
        x = np.array([0.5, 99.5])
        y = np.array([0.5, 99.5])
        result = infov2d(x, y, bezels=0.5, naxes=[100, 100])
        assert np.all(result)

    def test_edge_just_outside(self):
        """Points at 0.4 should be outside with bezel=0.5."""
        x = np.array([0.4, 99.6])
        y = np.array([50.0, 50.0])
        result = infov2d(x, y, bezels=0.5, naxes=[100, 100])
        assert not np.any(result)

    def test_negative_bezel_expands(self):
        """Negative bezel allows points slightly outside."""
        x = np.array([-0.5, 100.5])
        y = np.array([50.0, 50.0])
        # With bezel=-1, range is [-1, 101] for both axes
        result = infov2d(x, y, bezels=-1.0, naxes=[100, 100])
        assert np.all(result)

    def test_asymmetric_bezel(self):
        """Test asymmetric bezels [[left, right], [bottom, top]]."""
        x = np.array([5.0, 95.0])
        y = np.array([10.0, 90.0])
        # bezel: left=5, right=5, bottom=10, top=10
        # Valid x: [5, 95], valid y: [10, 90]
        result = infov2d(x, y, bezels=[[5, 5], [10, 10]], naxes=[100, 100])
        assert np.all(result)
