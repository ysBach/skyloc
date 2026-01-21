import os
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from skyloc.ioutils import load_header, get_wcs, infov2d, all_world2pix_infov


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


def test_load_header(temp_fits):
    header = load_header(temp_fits)
    assert isinstance(header, fits.Header)
    assert header["CRVAL1"] == 10.0


def test_get_wcs_from_file(temp_fits):
    wcs = get_wcs(temp_fits)
    assert isinstance(wcs, WCS)
    assert wcs.wcs.crval[0] == 10.0


def test_get_wcs_from_header(temp_fits):
    header = load_header(temp_fits)
    wcs = get_wcs(header)
    assert isinstance(wcs, WCS)
    assert wcs.wcs.crval[0] == 10.0


def test_infov2d():
    # Test basic functionality moved from utils
    x = np.array([50.0, -1.0])
    y = np.array([50.0, 50.0])
    infov = infov2d(x, y, naxes=[100, 100])
    assert infov[0] == True
    assert infov[1] == False


def test_all_world2pix_infov(temp_fits):
    wcs = get_wcs(temp_fits)
    # RA=10, DEC=20 is at CRPIX=50.5 (1-based), so 49.5 (0-based)
    pixels, infov = all_world2pix_infov(wcs, [10.0], [20.0], 0)
    assert np.isclose(pixels[0, 0], 49.5)
    assert np.isclose(pixels[0, 1], 49.5)
    assert infov[0] == True
