"""Tests for skyloc.keteutils.fov module - FOVCollection.

Tests verify specific behaviors with known values.
"""

import numpy as np
import pytest
import kete
from skyloc.keteutils.fov import FOVCollection


class TestFOVCollection:
    """Test FOVCollection wrapper class."""

    def test_init_from_fovlist(self, sample_fovlist):
        """Can initialize from FOVList."""
        fovc = FOVCollection(sample_fovlist)
        assert len(fovc) == len(sample_fovlist)
        assert len(fovc) == 3

    def test_init_from_single_fov(self, sample_fov):
        """Can initialize from a single FOV (wrapped in list)."""
        fovc = FOVCollection([sample_fov])
        assert len(fovc) == 1

    def test_indexing_returns_correct_fov(self, sample_fovlist, sample_jd_tdb):
        """Integer indexing returns correct FOV."""
        fovc = FOVCollection(sample_fovlist)
        fov0 = fovc[0]
        # First FOV should have JD = sample_jd_tdb
        assert np.isclose(fov0.observer.jd, sample_jd_tdb, atol=1e-6)

    def test_slicing_returns_fovcollection(self, sample_fovlist):
        """Slice indexing returns new FOVCollection."""
        fovc = FOVCollection(sample_fovlist)
        sliced = fovc[0:2]
        assert isinstance(sliced, (list, np.ndarray))
        assert len(sliced) == 2

    def test_mask_indexing_values(self, sample_fovlist, sample_jd_tdb):
        """Boolean mask indexing returns correct FOVs."""
        fovc = FOVCollection(sample_fovlist)
        mask = np.array([True, False, True])
        masked = fovc[mask]
        assert len(masked) == 2
        # Should have first (jd=sample_jd_tdb) and third (jd=sample_jd_tdb+2)
        jds = [fov.observer.jd for fov in masked]
        assert np.isclose(jds[0], sample_jd_tdb, atol=1e-6)
        assert np.isclose(jds[1], sample_jd_tdb + 2, atol=1e-6)

    def test_fov_jds_values(self, sample_fovlist, sample_jd_tdb):
        """JDs are correct and in order."""
        fovc = FOVCollection(sample_fovlist)
        jds = fovc.fov_jds
        assert len(jds) == 3
        expected_jds = [sample_jd_tdb, sample_jd_tdb + 1, sample_jd_tdb + 2]
        np.testing.assert_allclose(jds, expected_jds, atol=1e-6)

    def test_fov_desigs_values(self, sample_fovlist):
        """Designations match expected pattern."""
        fovc = FOVCollection(sample_fovlist)
        desigs = fovc.fov_desigs
        assert len(desigs) == 3
        assert desigs[0] == "test_obs_0"
        assert desigs[1] == "test_obs_1"
        assert desigs[2] == "test_obs_2"

    def test_mask_by_desig_single(self, sample_fovlist):
        """mask_by_desig with single designation."""
        fovc = FOVCollection(sample_fovlist)
        mask = fovc.mask_by_desig(["test_obs_1"])
        assert np.sum(mask) == 1
        assert mask[1] == True
        assert mask[0] == False
        assert mask[2] == False

    def test_mask_by_desig_multiple(self, sample_fovlist):
        """mask_by_desig with multiple designations."""
        fovc = FOVCollection(sample_fovlist)
        mask = fovc.mask_by_desig(["test_obs_0", "test_obs_2"])
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_mask_by_desig_nonexistent(self, sample_fovlist):
        """mask_by_desig with nonexistent designation returns all False."""
        fovc = FOVCollection(sample_fovlist)
        mask = fovc.mask_by_desig(["nonexistent"])
        assert np.sum(mask) == 0

    def test_fov_jds_by_desig(self, sample_fovlist, sample_jd_tdb):
        """fov_jds_by_desig returns correct JDs for given designations."""
        fovc = FOVCollection(sample_fovlist)
        jds = fovc.fov_jds_by_desig(["test_obs_0", "test_obs_2"])
        assert len(jds) == 2
        np.testing.assert_allclose(jds, [sample_jd_tdb, sample_jd_tdb + 2], atol=1e-6)
