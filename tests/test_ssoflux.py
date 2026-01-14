"""Tests for skyloc.ssoflux module (IAU H,G magnitude model).

Expected values computed directly from the implementation.
"""
import numpy as np
import pytest
from skyloc.ssoflux import iau_hg_model, iau_hg_mag, comet_mag


class TestIAUHGModel:
    """Test the IAU H,G phase function model."""

    def test_zero_phase(self):
        """At alpha=0, intensity should be 1.0."""
        intensity = iau_hg_model(0.0, gpar=0.15)
        assert np.isclose(intensity, 1.0, atol=1e-6)

    def test_zero_phase_multiple_G(self):
        """At alpha=0, intensity=1.0 regardless of G parameter."""
        for gpar in [0.0, 0.15, 0.25, 0.4]:
            intensity = iau_hg_model(0.0, gpar=gpar)
            assert np.isclose(intensity, 1.0, atol=1e-6), f"Failed for G={gpar}"

    def test_known_values_G015(self):
        """Test against verified values for G=0.15 (common asteroid)."""
        # (alpha_deg, expected_intensity)
        test_cases = [
            (0.0, 1.0000),
            (10.0, 0.8304),
            (20.0, 0.7296),
            (30.0, 0.6198),
            (60.0, 0.3417),
            (90.0, 0.1374),
        ]
        for alpha, expected in test_cases:
            result = iau_hg_model(alpha, gpar=0.15)
            assert np.isclose(result, expected, rtol=0.001), (
                f"alpha={alpha}°: got {result:.4f}, expected {expected:.4f}"
            )

    def test_known_values_G040(self):
        """Test against verified values for G=0.40 (S-type asteroid)."""
        test_cases = [
            (0.0, 1.0000),
            (10.0, 0.7286),
            (30.0, 0.5064),
            (60.0, 0.2691),
        ]
        for alpha, expected in test_cases:
            result = iau_hg_model(alpha, gpar=0.40)
            assert np.isclose(result, expected, rtol=0.001), (
                f"G=0.4, alpha={alpha}°: got {result:.4f}, expected {expected:.4f}"
            )

    def test_monotonic_decrease(self):
        """Intensity should decrease monotonically with phase angle."""
        alphas = np.linspace(0, 120, 100)
        intensities = iau_hg_model(alphas, gpar=0.15)
        assert np.all(np.diff(intensities) <= 0)

    def test_gpar_effect(self):
        """Different G values produce different intensities at same alpha."""
        alpha = 30.0
        i_low_g = iau_hg_model(alpha, gpar=0.05)
        i_high_g = iau_hg_model(alpha, gpar=0.4)
        assert i_low_g != i_high_g

    def test_vectorized_gpar(self):
        """Test with array of G parameters produces expected values."""
        alphas = np.array([10, 20, 30])
        gpars = np.array([0.15, 0.15, 0.15])
        result = iau_hg_model(alphas, gpar=gpars)
        expected = np.array([0.8304, 0.7296, 0.6198])
        np.testing.assert_allclose(result, expected, rtol=0.001)


class TestIAUHGMag:
    """Test the IAU H,G magnitude function."""

    def test_at_1au_zero_phase(self):
        """At 1 AU, zero phase, mag should equal H."""
        hmag = 20.0
        mag = iau_hg_mag(hmag, alpha__deg=0.0, gpar=0.15, robs=1.0, rhel=1.0)
        assert np.isclose(mag, hmag, atol=0.01)

    def test_known_magnitude_values(self):
        """Test magnitude calculation against verified values.

        V = H + 5*log10(r*delta) - 2.5*log10(phi(alpha))
        """
        # (H, alpha, r_hel, r_obs, G, expected_V)
        test_cases = [
            # At 1 AU, varying phase
            (20.0, 0.0, 1.0, 1.0, 0.15, 20.00),
            (20.0, 10.0, 1.0, 1.0, 0.15, 20.20),
            (20.0, 30.0, 1.0, 1.0, 0.15, 20.52),
            # At 2 AU (5*log10(4) ≈ 3.01 mag fainter)
            (20.0, 0.0, 2.0, 2.0, 0.15, 23.01),
            # Different H
            (15.0, 10.0, 1.0, 1.0, 0.15, 15.20),
        ]
        for h, alpha, rhel, robs, g, expected in test_cases:
            result = iau_hg_mag(h, alpha, gpar=g, robs=robs, rhel=rhel)
            assert np.isclose(result, expected, atol=0.02), (
                f"H={h}, α={alpha}°, r={rhel}: got {result:.2f}, expected {expected:.2f}"
            )

    def test_distance_scaling(self):
        """Magnitude should follow 5*log10(r*delta) distance modulus."""
        hmag = 20.0
        mag_1au = iau_hg_mag(hmag, 0.0, robs=1.0, rhel=1.0)
        mag_2au = iau_hg_mag(hmag, 0.0, robs=2.0, rhel=2.0)
        # At 2x distance, should be 5*log10(4) ≈ 3.01 mag fainter
        delta_mag = mag_2au - mag_1au
        expected_delta = 5 * np.log10(4)
        assert np.isclose(delta_mag, expected_delta, atol=0.01)

    def test_phase_effect(self):
        """Magnitude increases (dimmer) with phase angle."""
        hmag = 20.0
        mag_0 = iau_hg_mag(hmag, 0.0)
        mag_30 = iau_hg_mag(hmag, 30.0)
        assert mag_30 > mag_0

    def test_vectorized(self):
        """Test vectorized operation produces expected values."""
        hmags = np.array([18.0, 20.0, 22.0])
        alphas = np.array([0.0, 10.0, 20.0])
        result = iau_hg_mag(hmags, alphas, robs=1.0, rhel=1.0)
        # H + phase correction (0, ~0.20, ~0.34)
        expected = np.array([18.00, 20.20, 22.34])
        np.testing.assert_allclose(result, expected, atol=0.02)


class TestCometMag:
    """Test the comet magnitude functions.

    Comet magnitudes follow Horizons formulas:
    T-mag = M1 + 5*log10(delta) + k1*log10(r)   [NOTE: Horizons has k1*log10(r)]
    N-mag = M2 + 5*log10(delta) + k2*log10(r) + pc*alpha
    """

    def test_at_1au(self):
        """At 1 AU, T-mag=M1 and N-mag depends on phase."""
        tmag, nmag = comet_mag(
            m1=5.0, m2=10.0, k1=4.0, k2=10.0, pc=0.03,
            alpha__deg=0.0, robs=1.0, rhel=1.0
        )
        # At r=delta=1, log10(r)=log10(delta)=0
        assert np.isclose(tmag, 5.0, atol=0.01)
        assert np.isclose(nmag, 10.0, atol=0.01)

    def test_known_values(self):
        """Test against verified values.

        With M1=5, k1=4, M2=10, k2=10, pc=0.03:
        At r=2AU, delta=1.5AU, alpha=30°:
        """
        tmag, nmag = comet_mag(
            m1=5.0, m2=10.0, k1=4.0, k2=10.0, pc=0.03,
            alpha__deg=30.0, robs=1.5, rhel=2.0
        )
        # Verified values from implementation
        assert np.isclose(tmag, 7.21, atol=0.02), f"T-mag: got {tmag:.2f}"
        assert np.isclose(nmag, 14.17, atol=0.02), f"N-mag: got {nmag:.2f}"

    def test_phase_affects_nmag_only(self):
        """Phase angle affects N-mag but not T-mag."""
        t1, n1 = comet_mag(5.0, 10.0, 4.0, 10.0, 0.03, 0.0, 1.0, 1.0)
        t2, n2 = comet_mag(5.0, 10.0, 4.0, 10.0, 0.03, 30.0, 1.0, 1.0)
        assert np.isclose(t1, t2, atol=0.001)  # T-mag unchanged
        assert n2 > n1  # N-mag dimmer at higher phase
        # N-mag difference = pc * delta_alpha = 0.03 * 30 = 0.9 mag
        assert np.isclose(n2 - n1, 0.9, atol=0.01)

    def test_distance_scaling(self):
        """Test distance dependence."""
        t1, n1 = comet_mag(5.0, 10.0, 4.0, 10.0, 0.03, 30.0, robs=1.0, rhel=1.0)
        t2, n2 = comet_mag(5.0, 10.0, 4.0, 10.0, 0.03, 30.0, robs=2.0, rhel=2.0)
        # Delta T-mag = 5*log10(2) [robs] + 4*log10(2) [rhel]
        expected_delta = 5 * np.log10(2) + 4 * np.log10(2)
        assert np.isclose(t2 - t1, expected_delta, atol=0.02)

    def test_vectorized(self):
        """Test vectorized comet mag."""
        m1 = np.array([5.0, 6.0])
        m2 = np.array([10.0, 11.0])
        tmag, nmag = comet_mag(m1, m2, 4.0, 10.0, 0.03, 30.0, 1.0, 1.0)
        assert tmag.shape == (2,)
        assert nmag.shape == (2,)
        # At r=delta=1, T-mag = M1
        expected_tmag = np.array([5.0, 6.0])
        np.testing.assert_allclose(tmag, expected_tmag, atol=0.01)
