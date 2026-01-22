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
