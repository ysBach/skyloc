import os
import sys

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from skyloc.ioutils import all_world2pix_infov, infov2d
from skyloc.ioutils.wcs import FastTanSipWCS


@pytest.fixture
def temp_fits(tmp_path):
    """Create a temporary FITS file for testing."""
    fname = tmp_path / "test.fits"
    data = np.zeros((100, 100))
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = 20.0
    header["CRPIX1"] = 50.5
    header["CRPIX2"] = 50.5
    header["CDELT1"] = -0.01
    header["CDELT2"] = 0.01
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(fname)
    return fname


def test_infov2d():
    # Test basic functionality moved from utils
    x = np.array([50.0, -1.0])
    y = np.array([50.0, 50.0])
    infov = infov2d(x, y, naxes=[100, 100])
    assert infov[0] == True
    assert infov[1] == False


@pytest.fixture
def user_header():
    raw_header_str = """
CRPIX1  =                 97.5 / Pixel coordinate of reference point
CRPIX2  =                349.5 / Pixel coordinate of reference point
PC1_1   =    -0.00164579523084 / Coordinate transformation matrix element
PC1_2   =    0.000400438382908 / Coordinate transformation matrix element
PC2_1   =    0.000466015197218 / Coordinate transformation matrix element
PC2_2   =     0.00166139357729 / Coordinate transformation matrix element
CDELT1  =                  1.0 / [deg] Coordinate increment at reference point
CDELT2  =                  1.0 / [deg] Coordinate increment at reference point
CUNIT1  = 'deg'                / Units of coordinate increment and value
CUNIT2  = 'deg'                / Units of coordinate increment and value
CTYPE1  = 'RA---TAN-SIP'       / TAN (gnomonic) projection + SIP distortions
CTYPE2  = 'DEC--TAN-SIP'       / TAN (gnomonic) projection + SIP distortions
CRVAL1  =        125.954743005 / [deg] Coordinate value at reference point
CRVAL2  =        23.1220676814 / [deg] Coordinate value at reference point
LONPOLE =                180.0 / [deg] Native longitude of celestial pole
LATPOLE =        23.1220676814 / [deg] Native latitude of celestial pole
A_ORDER =                    3 / SIP polynomial order, axis 0, detector to sky
A_0_0   =      -0.313710215359 / SIP distortion coefficient
A_0_1   =    7.55870053771E-05 / SIP distortion coefficient
A_0_2   =    1.38150098327E-06 / SIP distortion coefficient
A_0_3   =   -3.52592814806E-12 / SIP distortion coefficient
A_1_0   =    6.96800457302E-05 / SIP distortion coefficient
A_1_1   =    4.17215543586E-06 / SIP distortion coefficient
A_1_2   =    5.80218149253E-10 / SIP distortion coefficient
A_2_0   =    1.60689380337E-06 / SIP distortion coefficient
A_2_1   =    9.30913650095E-11 / SIP distortion coefficient
A_3_0   =    5.88439205964E-10 / SIP distortion coefficient
B_ORDER =                    3 / SIP polynomial order, axis 1, detector to sky
B_0_0   =       -1.23712921675 / SIP distortion coefficient
B_0_1   =     0.00010482491727 / SIP distortion coefficient
B_0_2   =    2.02004856688E-06 / SIP distortion coefficient
B_0_3   =    4.52910251664E-10 / SIP distortion coefficient
B_1_0   =   -0.000145105601055 / SIP distortion coefficient
B_1_1   =    7.56822092841E-07 / SIP distortion coefficient
B_1_2   =    7.37331578688E-11 / SIP distortion coefficient
B_2_0   =    6.67768435385E-06 / SIP distortion coefficient
B_2_1   =    6.15502511207E-10 / SIP distortion coefficient
B_3_0   =    2.55517987961E-11 / SIP distortion coefficient
AP_ORDER=                    3 / SIP polynomial order, axis 0, sky to detector
AP_0_0  =       0.311726596638 / SIP distortion coefficient
AP_0_1  =   -8.04295314611E-05 / SIP distortion coefficient
AP_0_2  =   -1.37582764548E-06 / SIP distortion coefficient
AP_0_3  =    1.50343704179E-11 / SIP distortion coefficient
AP_1_0  =    -7.6799056082E-05 / SIP distortion coefficient
AP_1_1  =   -4.15352973491E-06 / SIP distortion coefficient
AP_1_2  =   -5.45547178333E-10 / SIP distortion coefficient
AP_2_0  =   -1.59931419627E-06 / SIP distortion coefficient
AP_2_1  =   -5.10183620747E-11 / SIP distortion coefficient
AP_3_0  =   -5.53322083157E-10 / SIP distortion coefficient
BP_ORDER=                    3 / SIP polynomial order, axis 1, sky to detector
BP_0_0  =        1.23206464223 / SIP distortion coefficient
BP_0_1  =   -0.000110892929001 / SIP distortion coefficient
BP_0_2  =   -2.00778632137E-06 / SIP distortion coefficient
BP_0_3  =   -4.41988757189E-10 / SIP distortion coefficient
BP_1_0  =    0.000139815269652 / SIP distortion coefficient
BP_1_1  =   -7.51286069252E-07 / SIP distortion coefficient
BP_1_2  =   -4.74033147718E-11 / SIP distortion coefficient
BP_2_0  =   -6.65714682174E-06 / SIP distortion coefficient
BP_2_1  =    -5.2894464541E-10 / SIP distortion coefficient
BP_3_0  =     9.1794303868E-13 / SIP distortion coefficient
"""
    return fits.Header.fromstring(raw_header_str, sep="\n")


def test_pix2world_compliance(user_header):
    """Compare FastTanSipWCS pix2world with Astropy."""
    h_dict = dict(user_header)
    w_astro = WCS(user_header)
    w_fast = FastTanSipWCS(h_dict)

    # Test grid
    # Use deterministic grid for reproducibility
    x = np.linspace(0, 2048, 50)
    y = np.linspace(0, 2048, 50)
    xx, yy = np.meshgrid(x, y)
    x = xx.ravel()
    y = yy.ravel()

    ra_a, dec_a = w_astro.all_pix2world(x, y, 0)
    ra_f, dec_f = w_fast.all_pix2world(x, y, 0)

    np.testing.assert_allclose(ra_f, ra_a, rtol=1e-3, atol=1e-8, err_msg="RA mismatch")
    np.testing.assert_allclose(
        dec_f, dec_a, rtol=1e-3, atol=1e-8, err_msg="Dec mismatch"
    )


def test_world2pix_compliance(user_header):
    """Compare FastTanSipWCS world2pix with Astropy."""
    h_dict = dict(user_header)
    w_astro = WCS(user_header)
    w_fast = FastTanSipWCS(h_dict)

    x_in = np.linspace(100, 1900, 20)  # Avoid edges to be safe
    y_in = np.linspace(100, 1900, 20)
    xx, yy = np.meshgrid(x_in, y_in)
    x_in = xx.ravel()
    y_in = yy.ravel()

    ra, dec = w_astro.all_pix2world(x_in, y_in, 0)

    x_a, y_a = w_astro.all_world2pix(ra, dec, 0)
    x_f, y_f = w_fast.all_world2pix(ra, dec, 0)

    # For x~1000, 1e-3 is 1 pixel.
    np.testing.assert_allclose(x_f, x_a, rtol=1e-6, atol=1.0e-3, err_msg="X mismatch")
    np.testing.assert_allclose(y_f, y_a, rtol=1e-6, atol=1.0e-3, err_msg="Y mismatch")


def test_scalar_compliance(user_header):
    """Verify scalar input/output formats match Astropy."""
    h_dict = dict(user_header)
    w_astro = WCS(user_header)
    w_fast = FastTanSipWCS(h_dict)

    x, y = 100.0, 200.0

    # 1. pix2world
    ra_a, dec_a = w_astro.all_pix2world(x, y, 0)
    ra_f, dec_f = w_fast.all_pix2world(x, y, 0)

    # Check that output is scalar(-ish)
    assert np.shape(ra_f) == np.shape(
        ra_a
    ), f"pix2world RA shape mismatch: {np.shape(ra_f)} vs {np.shape(ra_a)}"
    assert np.shape(dec_f) == np.shape(dec_a), f"pix2world Dec shape mismatch"
    # assert type(ra_f) == type(ra_a) # Might be too strict if one is float vs np.float64

    # Check values
    np.testing.assert_allclose(ra_f, ra_a, atol=1e-8)
    np.testing.assert_allclose(dec_f, dec_a, atol=1e-8)

    # 2. world2pix
    # Test reverse
    ra, dec = 125.9, 23.1
    x_a, y_a = w_astro.all_world2pix(ra, dec, 0)
    x_f, y_f = w_fast.all_world2pix(ra, dec, 0)

    assert np.shape(x_f) == np.shape(
        x_a
    ), f"world2pix X shape mismatch: {np.shape(x_f)} vs {np.shape(x_a)}"
    assert np.shape(y_f) == np.shape(y_a), f"world2pix Y shape mismatch"

    np.testing.assert_allclose(x_f, x_a, atol=1e-3)
    np.testing.assert_allclose(y_f, y_a, atol=1e-3)


def test_to_astropy(user_header):
    """Verify FastTanSipWCS.to_astropy() method."""
    # 1. Setup
    ref_wcs = WCS(user_header)
    h_dict = dict(user_header)
    fast_wcs = FastTanSipWCS(h_dict)

    # 2. Convert back to Astropy
    new_wcs = fast_wcs.to_astropy()

    # 3. Basic Header/Attribute Checks
    np.testing.assert_allclose(
        new_wcs.wcs.crval, ref_wcs.wcs.crval, err_msg="CRVAL mismatch"
    )
    np.testing.assert_allclose(
        new_wcs.wcs.crpix, ref_wcs.wcs.crpix, err_msg="CRPIX mismatch"
    )

    # Linear transformation (pixel scale/rotation)
    np.testing.assert_allclose(
        new_wcs.pixel_scale_matrix,
        ref_wcs.pixel_scale_matrix,
        rtol=1e-10,
        err_msg="Linear transform mismatch",
    )

    # SIP Coefficients
    assert new_wcs.sip.a_order == ref_wcs.sip.a_order
    assert new_wcs.sip.b_order == ref_wcs.sip.b_order
    np.testing.assert_allclose(new_wcs.sip.a, ref_wcs.sip.a, err_msg="SIP A mismatch")
    np.testing.assert_allclose(new_wcs.sip.b, ref_wcs.sip.b, err_msg="SIP B mismatch")

    if ref_wcs.sip.ap_order:
        assert new_wcs.sip.ap_order == ref_wcs.sip.ap_order
        np.testing.assert_allclose(
            new_wcs.sip.ap, ref_wcs.sip.ap, err_msg="SIP AP mismatch"
        )
    if ref_wcs.sip.bp_order:
        assert new_wcs.sip.bp_order == ref_wcs.sip.bp_order
        np.testing.assert_allclose(
            new_wcs.sip.bp, ref_wcs.sip.bp, err_msg="SIP BP mismatch"
        )

    # 4. Functional Verification (Round-trip pixels)
    # Use deterministic grid like in test_pix2world_compliance
    x = np.linspace(0, 2048, 20)
    y = np.linspace(0, 2048, 20)
    xx, yy = np.meshgrid(x, y)
    x_flat, y_flat = xx.ravel(), yy.ravel()

    # Reference transform
    ra_ref, dec_ref = ref_wcs.all_pix2world(x_flat, y_flat, 1)  # 1-based origin

    # New WCS transform
    ra_new, dec_new = new_wcs.all_pix2world(x_flat, y_flat, 1)

    np.testing.assert_allclose(
        ra_new, ra_ref, rtol=1e-10, atol=1e-8, err_msg="RA mismatch"
    )
    np.testing.assert_allclose(
        dec_new, dec_ref, rtol=1e-10, atol=1e-8, err_msg="Dec mismatch"
    )

    # 5. Strict Header Comparison
    # Verify that the exported header contains identical critical keys
    h_ref = ref_wcs.to_header(relax=True)
    h_new = new_wcs.to_header(relax=True)

    keys_to_check = [
        "CRVAL1",
        "CRVAL2",
        "CRPIX1",
        "CRPIX2",
        "A_ORDER",
        "B_ORDER",
        "AP_ORDER",
        "BP_ORDER",
    ]

    for key in keys_to_check:
        if key in h_ref:
            # Check numbers with tolerance, integers/strings exact
            if isinstance(h_new[key], (float, np.floating)):
                assert np.isclose(h_new[key], h_ref[key]), f"{key} mismatch"
            else:
                assert h_new[key] == h_ref[key], f"{key} mismatch"


class TestAllWorld2PixInfOV:
    """Tests for all_world2pix_infov function."""

    def test_basic_in_fov(self, user_header):
        """Points near CRVAL should be in FOV."""
        from skyloc.ioutils import all_world2pix_infov

        w = WCS(user_header)
        # Point at reference position (CRVAL)
        ra = np.array([user_header["CRVAL1"]])
        dec = np.array([user_header["CRVAL2"]])

        # Use large naxes to ensure point is within
        pixels, infov = all_world2pix_infov(w, ra, dec, 0, naxes=[2048, 2048])

        assert len(pixels) == 1
        assert pixels.shape == (1, 2)
        assert infov[0] is True or infov[0] == True  # Near CRPIX should be in FOV

    def test_out_of_fov(self, user_header):
        """Points outside image bounds should be out of FOV."""
        from skyloc.ioutils import all_world2pix_infov

        w = WCS(user_header)
        # Point slightly offset from CRVAL - will project to valid pixel but outside small naxes
        # CRVAL is at ~(125.95, 23.12), shift slightly
        ra = np.array([user_header["CRVAL1"] + 1.0])  # ~1 deg offset
        dec = np.array([user_header["CRVAL2"] + 1.0])

        # Use very small naxes so the offset point is definitely out
        pixels, infov = all_world2pix_infov(w, ra, dec, 0, naxes=[10, 10])

        assert len(infov) == 1
        assert infov[0] == False

    def test_custom_bezels(self, user_header):
        """Test with custom bezels parameter."""
        from skyloc.ioutils import all_world2pix_infov

        w = WCS(user_header)
        ra = np.array([user_header["CRVAL1"]])
        dec = np.array([user_header["CRVAL2"]])

        # Symmetric bezels
        pixels, infov = all_world2pix_infov(
            w, ra, dec, 0, naxes=[2048, 2048], bezels=10
        )
        assert len(infov) == 1

        # Asymmetric bezels
        pixels, infov = all_world2pix_infov(
            w, ra, dec, 0, naxes=[2048, 2048], bezels=[[5, 5], [10, 10]]
        )
        assert len(infov) == 1

    def test_with_fast_wcs(self, user_header):
        """Test that FastTanSipWCS works with all_world2pix_infov."""
        from skyloc.ioutils import all_world2pix_infov

        h_dict = dict(user_header)
        w_fast = FastTanSipWCS(h_dict)

        ra = np.array([user_header["CRVAL1"]])
        dec = np.array([user_header["CRVAL2"]])

        pixels, infov = all_world2pix_infov(w_fast, ra, dec, 0, naxes=[2048, 2048])

        assert len(pixels) == 1
        assert infov[0] == True

    def test_multiple_points(self, user_header):
        """Test with multiple points, some in and some out of FOV."""
        from skyloc.ioutils import all_world2pix_infov

        w = WCS(user_header)

        # Mix of points: one at CRVAL (in FOV), one offset (out of small FOV)
        ra = np.array([user_header["CRVAL1"], user_header["CRVAL1"] + 2.0])
        dec = np.array([user_header["CRVAL2"], user_header["CRVAL2"] + 2.0])

        # Use naxes centered on CRPIX=(97.5, 349.5) - first point at CRPIX, second far off
        pixels, infov = all_world2pix_infov(w, ra, dec, 0, naxes=[200, 700])

        assert len(infov) == 2
        # First point at CRVAL projects to CRPIX which is in [200, 700], second is offset
        assert infov[0] == True
        assert infov[1] == False


class TestCompactEphemParqCols:
    """Tests for compact_ephem_parq_cols function."""

    def test_basic_factor_scaling(self):
        """Test that columns are scaled by factor correctly."""
        from skyloc.ioutils.ephemeris import compact_ephem_parq_cols, EPH_DTYPES_BASE

        # Create sample DataFrame
        eph = pd.DataFrame(
            {
                "alpha": [45.0, 90.0, 135.0],
                "r_hel": [1.0, 2.5, 5.0],
                "vmag": [10.0, 15.5, 20.0],
                "other_col": [1, 2, 3],  # Should be unchanged
            }
        )

        result = compact_ephem_parq_cols(eph, EPH_DTYPES_BASE)

        # Check that scaled columns exist
        assert "alpha*360" in result.columns
        assert "r_hel*1000" in result.columns
        assert "vmag*2000" in result.columns

        # Check that original columns are dropped
        assert "alpha" not in result.columns
        assert "r_hel" not in result.columns
        assert "vmag" not in result.columns

        # Other columns should remain
        assert "other_col" in result.columns

        # Verify scaling (alpha * 360)
        np.testing.assert_array_almost_equal(
            result["alpha*360"].values.astype("float64"),
            [45.0 * 360, 90.0 * 360, 135.0 * 360],
        )

    def test_nan_handling(self):
        """Test that NaN values are replaced with navalue."""
        from skyloc.ioutils.ephemeris import compact_ephem_parq_cols, EPH_DTYPES_BASE

        eph = pd.DataFrame({"alpha": [45.0, np.nan, 90.0]})

        result = compact_ephem_parq_cols(eph, EPH_DTYPES_BASE)

        # NaN should be replaced with navalue (65535 for alpha)
        assert result["alpha*360"].iloc[1] == 65535

    def test_healpix_encoding(self):
        """Test HEALPix coordinate encoding."""
        from skyloc.ioutils.ephemeris import compact_ephem_parq_cols, EPH_DTYPES_BASE

        # Create DataFrame with coordinate columns
        eph = pd.DataFrame(
            {
                "ra": [180.0, 90.0, 270.0],
                "dec": [0.0, 45.0, -30.0],
            }
        )

        # Use smaller nside for testing (2^8)
        result = compact_ephem_parq_cols(eph, EPH_DTYPES_BASE, nside=2**8)

        # HEALPix column should exist
        hp_col = "eqj2000_hpidx_ring_2^8"
        assert hp_col in result.columns

        # Original columns should be dropped
        assert "ra" not in result.columns
        assert "dec" not in result.columns

        # HEALPix values should be valid uint64
        assert result[hp_col].dtype == np.uint64

    def test_drop_cols(self):
        """Test that drop_cols parameter works."""
        from skyloc.ioutils.ephemeris import compact_ephem_parq_cols, EPH_DTYPES_BASE

        eph = pd.DataFrame(
            {
                "alpha": [45.0],
                "to_drop": [100],
                "keep_me": [200],
            }
        )

        result = compact_ephem_parq_cols(eph, EPH_DTYPES_BASE, drop_cols=["to_drop"])

        assert "to_drop" not in result.columns
        assert "keep_me" in result.columns

    def test_custom_scheme(self):
        """Test nested HEALPix scheme."""
        from skyloc.ioutils.ephemeris import compact_ephem_parq_cols, EPH_DTYPES_BASE

        eph = pd.DataFrame({"ra": [180.0], "dec": [0.0]})

        result = compact_ephem_parq_cols(
            eph, EPH_DTYPES_BASE, nside=2**8, scheme="nested"
        )

        # Check for nested scheme in column name
        assert "eqj2000_hpidx_nested_2^8" in result.columns


class TestLoadCompactParqEphem:
    """Tests for load_compact_parq_ephem function."""

    def test_round_trip(self, tmp_path):
        """Test compact then load recovers original values within precision."""
        from skyloc.ioutils.ephemeris import (
            compact_ephem_parq_cols,
            load_compact_parq_ephem,
            EPH_DTYPES_BASE,
        )

        # Create sample DataFrame
        original = pd.DataFrame(
            {
                "alpha": [45.123, 90.456, 135.789],
                "r_hel": [1.234, 2.567, 5.890],
                "vmag": [10.001, 15.502, 20.003],
                "ra": [180.0, 90.0, 270.0],
                "dec": [0.0, 45.0, -30.0],
            }
        )

        # Compact
        compacted = compact_ephem_parq_cols(original, EPH_DTYPES_BASE, nside=2**16)

        # Save to parquet
        fpath = tmp_path / "test_eph.parquet"
        compacted.to_parquet(fpath)

        # Load back
        loaded = load_compact_parq_ephem(fpath, EPH_DTYPES_BASE)

        # Check that original columns are restored
        assert "alpha" in loaded.columns
        assert "r_hel" in loaded.columns
        assert "vmag" in loaded.columns
        assert "ra" in loaded.columns
        assert "dec" in loaded.columns

        # Check values within precision (alpha: factor=360, so ~0.003 deg precision)
        np.testing.assert_array_almost_equal(
            loaded["alpha"].values, original["alpha"].values, decimal=2
        )
        np.testing.assert_array_almost_equal(
            loaded["r_hel"].values, original["r_hel"].values, decimal=2
        )
        np.testing.assert_array_almost_equal(
            loaded["vmag"].values, original["vmag"].values, decimal=3
        )

        # Coordinates (HEALPix precision depends on nside)
        np.testing.assert_array_almost_equal(
            loaded["ra"].values, original["ra"].values, decimal=1
        )
        np.testing.assert_array_almost_equal(
            loaded["dec"].values, original["dec"].values, decimal=1
        )

    def test_filter_on_factored_column(self, tmp_path):
        """Test filtering on factored columns uses original values."""
        from skyloc.ioutils.ephemeris import (
            compact_ephem_parq_cols,
            load_compact_parq_ephem,
            EPH_DTYPES_BASE,
        )

        original = pd.DataFrame(
            {
                "alpha": [30.0, 60.0, 90.0, 120.0, 150.0],
            }
        )

        compacted = compact_ephem_parq_cols(original, EPH_DTYPES_BASE)
        fpath = tmp_path / "test_filter.parquet"
        compacted.to_parquet(fpath)

        # Filter using original column name and value
        loaded = load_compact_parq_ephem(
            fpath, EPH_DTYPES_BASE, filters=[("alpha", ">", 100)]
        )

        # Should only get rows with alpha > 100
        assert len(loaded) == 2
        assert all(loaded["alpha"] > 100 - 1)  # Allow small precision error

    def test_filter_dnf_format(self, tmp_path):
        """Test DNF format filters (OR between groups)."""
        from skyloc.ioutils.ephemeris import (
            compact_ephem_parq_cols,
            load_compact_parq_ephem,
            EPH_DTYPES_BASE,
        )

        original = pd.DataFrame(
            {
                "alpha": [30.0, 60.0, 90.0, 120.0, 150.0],
                "vmag": [5.0, 10.0, 15.0, 20.0, 25.0],
            }
        )

        compacted = compact_ephem_parq_cols(original, EPH_DTYPES_BASE)
        fpath = tmp_path / "test_dnf.parquet"
        compacted.to_parquet(fpath)

        # DNF: (alpha < 50) OR (vmag > 20)
        loaded = load_compact_parq_ephem(
            fpath, EPH_DTYPES_BASE, filters=[[("alpha", "<", 50)], [("vmag", ">", 20)]]
        )

        # Should get rows with alpha=30 and vmag=25
        assert len(loaded) == 2

    def test_columns_selection(self, tmp_path):
        """Test loading only specific columns."""
        from skyloc.ioutils.ephemeris import (
            compact_ephem_parq_cols,
            load_compact_parq_ephem,
            EPH_DTYPES_BASE,
        )

        original = pd.DataFrame(
            {
                "alpha": [45.0, 90.0],
                "r_hel": [1.0, 2.0],
                "vmag": [10.0, 15.0],
            }
        )

        compacted = compact_ephem_parq_cols(original, EPH_DTYPES_BASE)
        fpath = tmp_path / "test_cols.parquet"
        compacted.to_parquet(fpath)

        # Load only alpha
        loaded = load_compact_parq_ephem(fpath, EPH_DTYPES_BASE, columns=["alpha"])

        assert "alpha" in loaded.columns
        assert "r_hel" not in loaded.columns
        assert "vmag" not in loaded.columns

    def test_coordinate_filter_raises(self, tmp_path):
        """Test that filtering on coordinate columns raises ValueError."""
        from skyloc.ioutils.ephemeris import (
            compact_ephem_parq_cols,
            load_compact_parq_ephem,
            EPH_DTYPES_BASE,
        )

        original = pd.DataFrame({"ra": [180.0], "dec": [0.0]})
        compacted = compact_ephem_parq_cols(original, EPH_DTYPES_BASE, nside=2**8)
        fpath = tmp_path / "test_coord_filter.parquet"
        compacted.to_parquet(fpath)

        with pytest.raises(ValueError, match="Filtering on coordinate column"):
            load_compact_parq_ephem(fpath, EPH_DTYPES_BASE, filters=[("ra", ">", 100)])


class TestParseCompactEphem:
    """Tests for parse_compact_ephem (DataFrame-level decompression)."""

    def test_round_trip_df(self):
        """Compact then parse_compact_ephem recovers original values."""
        from skyloc.ioutils.ephemeris import (
            compact_ephem_parq_cols,
            parse_compact_ephem,
            EPH_DTYPES_BASE,
        )

        original = pd.DataFrame(
            {
                "alpha": [45.123, 90.456, 135.789],
                "r_hel": [1.234, 2.567, 5.890],
                "vmag": [10.001, 15.502, 20.003],
                "ra": [180.0, 90.0, 270.0],
                "dec": [0.0, 45.0, -30.0],
            }
        )

        compacted = compact_ephem_parq_cols(original, EPH_DTYPES_BASE, nside=2**16)
        recovered = parse_compact_ephem(compacted, dtypes=EPH_DTYPES_BASE)

        for col in ["alpha", "r_hel", "vmag", "ra", "dec"]:
            assert col in recovered.columns

        np.testing.assert_array_almost_equal(
            recovered["alpha"].values, original["alpha"].values, decimal=2
        )
        np.testing.assert_array_almost_equal(
            recovered["r_hel"].values, original["r_hel"].values, decimal=2
        )
        np.testing.assert_array_almost_equal(
            recovered["vmag"].values, original["vmag"].values, decimal=3
        )
        np.testing.assert_array_almost_equal(
            recovered["ra"].values, original["ra"].values, decimal=1
        )
        np.testing.assert_array_almost_equal(
            recovered["dec"].values, original["dec"].values, decimal=1
        )

    def test_columns_selection(self):
        """parse_compact_ephem respects `columns` parameter."""
        from skyloc.ioutils.ephemeris import (
            compact_ephem_parq_cols,
            parse_compact_ephem,
            EPH_DTYPES_BASE,
        )

        original = pd.DataFrame(
            {"alpha": [45.0, 90.0], "r_hel": [1.0, 2.0], "vmag": [10.0, 15.0]}
        )
        compacted = compact_ephem_parq_cols(original, EPH_DTYPES_BASE)
        recovered = parse_compact_ephem(
            compacted, dtypes=EPH_DTYPES_BASE, columns=["alpha"]
        )

        assert list(recovered.columns) == ["alpha"]

    def test_does_not_mutate_input(self):
        """parse_compact_ephem must not modify the input DataFrame."""
        from skyloc.ioutils.ephemeris import (
            compact_ephem_parq_cols,
            parse_compact_ephem,
            EPH_DTYPES_BASE,
        )

        original = pd.DataFrame({"alpha": [45.0, 90.0]})
        compacted = compact_ephem_parq_cols(original, EPH_DTYPES_BASE)
        cols_before = list(compacted.columns)

        _ = parse_compact_ephem(compacted, dtypes=EPH_DTYPES_BASE)

        assert list(compacted.columns) == cols_before


# Import pandas for ephemeris tests
import pandas as pd
