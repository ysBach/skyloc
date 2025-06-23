import numpy as np
import numba as nb


__all__ = [
    "iau_hg_model",
    "iau_hg_mag",
    "comet_mag",
]


_D2R = np.pi / 180.0


# numba makes it ~3x faster than the pure numpy version.
@nb.njit(fastmath=True, cache=True)
def iau_hg_model(alpha, gpar=0.15):
    """The IAU HG phase function model in intensity (1 at alpha=0).

    Parameters
    ----------
    alpha__deg : array_like
        The phase angle (Sun-Target-Observer angle) in degrees.

    gpar : float, optional
        The slope parameter ($G$) in the IAU H, G modeling. See Notes.
        By default ``0.15``.

    Notes
    -----
    Semi-empirical model of the phase function of the Moon, asteroids, and
    other (airless) solar system objects. The phase function defined at
    $(0^\circ \le \alpha \le 120^\circ)$ for the phase angle $\alpha$. It
    is given by the following equation:

    .. math::
        \Phi_\mathrm{HG}(\alpha, G) = G \Phi_{HG1}(\alpha) + (1-G) \Phi_{HG2}(\alpha)

    where

    .. math::
        \Phi_{HG i}(\alpha) = W \left ( 1-\frac{C_i \sin \alpha}{0.119+1.341 \sin \alpha-0.754 \sin ^2 \alpha} \right )
        + (1 - W) \times \exp \left \{ -A_i \left [ \tan \frac{\alpha}{2} \right ]^{B_i} \right \}

    and

    .. math::
        W(\alpha) = \exp \left \{ -90.56 \tan^2 \frac{\alpha}{2} \right \}

    The parameters $A_i$, $B_i$, and $C_i$ are given by:

    .. math::
        A_1, A_2 &= 3.332, 1.862 \sep
        B_1, B_2 = 0.631, 1.218 \sep
        C_1, C_2 = 0.986, 0.238

    Reference: Bowell et al. 1989
    https://ui.adsabs.harvard.edu/abs/1989aste.conf..524B/abstract
    """
    n = alpha.shape[0]
    intensity = np.empty(n, dtype=np.float64)
    # onemgpar = 1.0 - gpar
    phi1 = np.empty(n, dtype=np.float64)
    phi2 = np.empty(n, dtype=np.float64)
    for i in range(n):
        # convert degrees to radians
        ar = np.abs(alpha[i]) * _D2R

        # intermediate trig and weighting terms
        sa = np.sin(ar)
        fa = sa / (0.119 + 1.341 * sa - 0.754 * sa * sa)
        tah = np.tan(ar * 0.5)
        w = np.exp(-90.56 * tah * tah)

        # smooth (s) and linear (l) components
        phi1_s = 1.0 - 0.986 * fa
        phi2_s = 1.0 - 0.238 * fa
        phi1_l = np.exp(-3.332 * np.power(tah, 0.631))
        phi2_l = np.exp(-1.862 * np.power(tah, 1.218))

        # mix them
        # intensity[i] = gpar[i] * (w * phi1_s + (1.0 - w) * phi1_l) + onemgpar[i] * (
        #     w * phi2_s + (1.0 - w) * phi2_l
        # )
        phi1[i] = w * phi1_s + (1.0 - w) * phi1_l
        phi2[i] = w * phi2_s + (1.0 - w) * phi2_l

    intensity = gpar * phi1 + (1.0 - gpar) * phi2
    return intensity


@nb.njit(fastmath=True, cache=True)
def iau_hg_mag(hmag, alpha__deg, gpar=0.15, robs=1, rhel=1):
    """The IAU HG phase function model in magnitudes scale.

    Parameters
    ----------
    hmag : float
        The absolute magnitude of the object.

    alpha__deg : array_like
        The phase angle (Sun-Target-Observer angle) in degrees.

    gpar : float, optional
        The slope parameter ($G$) in the IAU H, G modeling. See Notes.
        By default ``0.15``.

    robs, rhel : float, optional
        The observer and heliocentric distance in au. By default 1 au.

    Returns
    -------
    mag : ndarray
        The apparent magnitude of the object at the given phase angle.
    """
    return (
        hmag
        + 5 * np.log10(robs * rhel)
        - 2.5 * np.log10(iau_hg_model(alpha__deg, gpar))
    )


@nb.njit(fastmath=True, cache=True)
def comet_mag(m1, m2, k1, k2, pc, alpha__deg, robs=1, rhel=1):
    """Calculate the apparent magnitude of a comet.

    Parameters
    ----------
    m1, m2 : float, array_like
        The magnitude related parameters (equivalent to asteroid's H-mag) for
        the total magnitude and nucleus magnitude, respectively.

    k1, k2 : float, array_like
        The heliocentric distance power law index for the total and nucleus
        magnitude, respectively. (equivalent to asteroids if 5)

    pc : float, array_like
        The phase coefficient for the nucleus magnitude.

    alpha__deg : float, array_like
        The phase angle (Sun-Target-Observer angle) in degrees.

    robs, rhel : float, array_like
        The observer and heliocentric distance in au. By default 1 au.

    Notes
    -----
    https://ssd.jpl.nasa.gov/horizons/manual.html#obsquan
    T-mag=M1 + 5*log10(delta) + k1*log10(r)
    N-mag=M2 + 5*log10(delta) + k2*log10(r) + pc*alpha
    """
    _rh = np.log10(rhel)
    _ro = np.log10(robs)
    tmag = m1 + 5 * _rh + k1 * _ro
    nmag = m2 + 5 * _rh + k2 * _ro + pc * alpha__deg
    return tmag, nmag
