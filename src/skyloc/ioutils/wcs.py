import numpy as np
from numba import njit, float64, int64, boolean, prange
from numba.experimental import jitclass

__all__ = ["FastTanSipWCS", "infov2d", "all_world2pix_infov"]

# Solver Constants
SOLVER_TOL = 1e-6
SOLVER_MAX_ITER = 30
SOLVER_H = 0.1
SOLVER_H_INV = 1.0 / SOLVER_H
D2R = np.pi / 180.0
R2D = 180.0 / np.pi

spec = [
    ("crval", float64[:]),
    ("crpix", float64[:]),
    ("pc", float64[:, :]),
    ("pc_inv", float64[:, :]),
    ("sip_a", float64[:, :]),
    ("sip_b", float64[:, :]),
    ("sip_ap", float64[:, :]),
    ("sip_bp", float64[:, :]),
    ("a_order", int64),
    ("b_order", int64),
    ("ap_order", int64),
    ("bp_order", int64),
    ("has_sip", boolean),
    ("has_sip_inv", boolean),
]


@njit(fastmath=True, inline="always")
def _apply_sip_fused(u, v, sip_a, sip_b, order):
    """
    Apply SIP distortion for both A and B matrices simultaneously using 2D
    Horner's scheme. This avoids explicit powers and shares loop logic. Assumes
    sip_a and sip_b are same shape (order+1, order+1) and zero-padded. No
    conditional branches used.
    """
    res_a = 0.0
    res_b = 0.0

    # Loop p from order down to 0
    for p in range(order, -1, -1):
        # Compute row polynomials in v for this p
        row_a = 0.0
        row_b = 0.0
        max_q = order - p
        for q in range(max_q, -1, -1):
            # Horner step: val = val * v + coeff
            # Trust zero padding - no bounds checking
            row_a = row_a * v + sip_a[p, q]
            row_b = row_b * v + sip_b[p, q]

        res_a = res_a * u + row_a
        res_b = res_b * u + row_b

    return res_a, res_b


@njit(fastmath=True, inline="always")
def _pix2world_kernel(x, y, crpix, crval, pc, sip_a, sip_b, a_order, b_order, has_sip):
    """Pixel (0-based) -> World (degrees)"""
    # 1. Relative pixel coordinates (u, v)
    u = x - (crpix[0] - 1.0)
    v = y - (crpix[1] - 1.0)

    # 2. SIP
    if has_sip:
        # a_order equals b_order due to padding
        du, dv = _apply_sip_fused(u, v, sip_a, sip_b, a_order)
        u += du
        v += dv

    # 3. Linear (Intermediate World Coords in degrees)
    xi = D2R * (pc[0, 0] * u + pc[0, 1] * v)
    eta = D2R * (pc[1, 0] * u + pc[1, 1] * v)

    # 4. TAN Projection (Standard Plane -> Native Sky)
    r2 = xi * xi + eta * eta
    if r2 == 0:
        phi = 0.0
        theta = np.pi / 2.0
        cos_phi = 1.0
        sin_phi = 0.0
        sint = 1.0
        cost = 0.0
    else:
        phi = np.arctan2(xi, -eta)
        theta = np.arctan2(1.0, np.sqrt(r2))
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        sint = np.sin(theta)
        cost = np.cos(theta)

    # 5. Rotation to Celestial, assuming LONPOLE = 180
    raz = crval[0] * D2R
    decz = crval[1] * D2R
    sin_decz = np.sin(decz)
    cos_decz = np.cos(decz)

    dec_rad = np.arcsin(sint * sin_decz - cost * cos_decz * cos_phi)

    y_ra = cost * sin_phi
    x_ra = sint * cos_decz + cost * sin_decz * cos_phi

    ra_diff = np.arctan2(y_ra, x_ra)
    ra_rad = raz + ra_diff

    return (R2D * ra_rad) % 360.0, R2D * dec_rad


@njit(fastmath=True, inline="always")
def _sky_to_linear_exact(ra, dec, crval, pc_inv):
    """
    Analytically invert the Sky -> TAN -> Linear transformation.
    Returns the 'linear' intermediate coordinates (u, v) before SIP.
    """
    raz = crval[0] * D2R
    decz = crval[1] * D2R
    ra_rad = ra * D2R
    dec_rad = dec * D2R

    # Rotate Sky -> Native
    sind = np.sin(dec_rad)
    cosd = np.cos(dec_rad)
    sind0 = np.sin(decz)
    cosd0 = np.cos(decz)
    sinda = np.sin(ra_rad - raz)
    cosda = np.cos(ra_rad - raz)

    sin_theta = sind * sind0 + cosd * cosd0 * cosda
    if sin_theta > 1.0:
        sin_theta = 1.0
    if sin_theta < -1.0:
        sin_theta = -1.0

    y_phi = cosd * sinda
    x_phi = sind * cosd0 - cosd * sind0 * cosda

    dphi = np.arctan2(-y_phi, x_phi)
    phi = dphi + np.pi

    if sin_theta <= 0:
        xi = 0.0
        eta = 0.0
    else:
        r_theta = R2D * (np.sqrt(1.0 / (sin_theta * sin_theta) - 1.0))
        xi = r_theta * np.sin(phi)
        eta = -r_theta * np.cos(phi)

    u = pc_inv[0, 0] * xi + pc_inv[0, 1] * eta
    v = pc_inv[1, 0] * xi + pc_inv[1, 1] * eta

    return u, v


@njit(fastmath=False, inline="never")
def _world2pix_kernel(
    ra,
    dec,
    crpix,
    crval,
    pc,
    pc_inv,
    sip_a,
    sip_b,
    sip_ap,
    sip_bp,
    a_order,
    b_order,
    ap_order,
    bp_order,
    has_sip,
    has_sip_inv,
):
    """
    World -> Pixel with Split-Loop Newton-Raphson.
    1. Analytically inverse Sky -> Linear (u_target, v_target).
    2. Iteratively solve u + SIP(u) = u_target.
    """
    # 1. Exact Linear Coordinates (Target for SIP solver)
    u_lin, v_lin = _sky_to_linear_exact(ra, dec, crval, pc_inv)

    # 2. Initial Guess using Inverse SIP (if available)
    if has_sip_inv:
        # ap_order equals bp_order due to padding
        du, dv = _apply_sip_fused(u_lin, v_lin, sip_ap, sip_bp, ap_order)
        u = u_lin + du
        v = v_lin + dv
    else:
        u = u_lin
        v = v_lin

    # 3. Newton-Raphson on SIP Loop
    # We want to find (u, v) such that:
    # F(u, v) = u + SIP(u, v) - u_lin = 0
    # G(u, v) = v + SIP(u, v) - v_lin = 0

    for i in range(SOLVER_MAX_ITER):
        # Forward SIP evaluation at current guess
        du_sip, dv_sip = _apply_sip_fused(u, v, sip_a, sip_b, a_order)

        # Residuals
        u_est = u + du_sip
        v_est = v + dv_sip

        res_u = u_est - u_lin
        res_v = v_est - v_lin

        # Check convergence on residuals
        if abs(res_u) < SOLVER_TOL and abs(res_v) < SOLVER_TOL:
            break

        # Jacobian via Finite Differences (SIP only!)
        # Much cheaper than full transform
        du_sip_h, dv_sip_h = _apply_sip_fused(u + SOLVER_H, v, sip_a, sip_b, a_order)
        du_sip_v, dv_sip_v = _apply_sip_fused(u, v + SOLVER_H, sip_a, sip_b, a_order)

        # d(res_u)/du = 1 + d(sip_u)/du
        dRu_du = 1.0 + (du_sip_h - du_sip) * SOLVER_H_INV
        dRv_du = (dv_sip_h - dv_sip) * SOLVER_H_INV

        dRu_dv = (du_sip_v - du_sip) * SOLVER_H_INV
        dRv_dv = 1.0 + (dv_sip_v - dv_sip) * SOLVER_H_INV

        det = dRu_du * dRv_dv - dRu_dv * dRv_du

        if abs(det) < 1e-20:
            break

        inv_det = 1.0 / det

        # Newton Step: -J_inv * Res
        step_u = (dRv_dv * (-res_u) - dRu_dv * (-res_v)) * inv_det
        step_v = (-dRv_du * (-res_u) + dRu_du * (-res_v)) * inv_det

        u += step_u
        v += step_v

        # Check convergence on steps? (Optional, residuals usually better)

    # Final Result
    x = u + (crpix[0] - 1.0)
    y = v + (crpix[1] - 1.0)

    return x, y


@jitclass(spec)
class _FastTanSipWCSStruct:
    def __init__(self, crval, crpix, pc, sip_a, sip_b, sip_ap, sip_bp):
        self.crval = crval
        self.crpix = crpix
        self.pc = pc
        self.sip_a = sip_a
        self.sip_b = sip_b
        self.sip_ap = sip_ap
        self.sip_bp = sip_bp

        self.a_order = int(max(sip_a.shape) - 1 if sip_a.size > 0 else 0)
        self.b_order = int(max(sip_b.shape) - 1 if sip_b.size > 0 else 0)
        self.ap_order = int(max(sip_ap.shape) - 1 if sip_ap.size > 0 else 0)
        self.bp_order = int(max(sip_bp.shape) - 1 if sip_bp.size > 0 else 0)

        self.has_sip = self.a_order > 0 or self.b_order > 0
        self.has_sip_inv = self.ap_order > 0 or self.bp_order > 0

        det = pc[0, 0] * pc[1, 1] - pc[0, 1] * pc[1, 0]
        if det == 0:
            self.pc_inv = np.zeros((2, 2), dtype=np.float64)
        else:
            self.pc_inv = (
                np.array(
                    [[pc[1, 1], -pc[0, 1]], [-pc[1, 0], pc[0, 0]]], dtype=np.float64
                )
                / det
            )


def create_wcs_from_dict(hdr):
    """Factory independent of class."""
    crval = np.array([hdr.get("CRVAL1", 0.0), hdr.get("CRVAL2", 0.0)], dtype=np.float64)
    crpix = np.array([hdr.get("CRPIX1", 0.0), hdr.get("CRPIX2", 0.0)], dtype=np.float64)
    cdelt = np.array([hdr.get("CDELT1", 1.0), hdr.get("CDELT2", 1.0)], dtype=np.float64)
    pc = np.array(
        [
            [hdr.get("PC1_1", 1.0), hdr.get("PC1_2", 0.0)],
            [hdr.get("PC2_1", 0.0), hdr.get("PC2_2", 1.0)],
        ],
        dtype=np.float64,
    )

    pc[0, :] *= cdelt[0]
    pc[1, :] *= cdelt[1]

    # Compute max orders for zero-padding optimization
    a_ord = int(hdr.get("A_ORDER", 0))
    b_ord = int(hdr.get("B_ORDER", 0))
    max_ab_order = max(a_ord, b_ord)

    ap_ord = int(hdr.get("AP_ORDER", 0))
    bp_ord = int(hdr.get("BP_ORDER", 0))
    max_apbp_order = max(ap_ord, bp_ord)

    def get_sip_matrix(prefix, target_order):
        mat = np.zeros((target_order + 1, target_order + 1), dtype=np.float64)

        if target_order == 0:
            return mat

        # Read available coefficients up to target_order
        for i in range(target_order + 1):
            for j in range(target_order + 1):
                if i + j <= target_order:
                    mat[i, j] = float(hdr.get(f"{prefix}_{i}_{j}", 0.0))
        return mat

    # Force A and B to have same shape (max_ab_order)
    sip_a = get_sip_matrix("A", max_ab_order)
    sip_b = get_sip_matrix("B", max_ab_order)

    # Force AP and BP to have same shape (max_apbp_order)
    sip_ap = get_sip_matrix("AP", max_apbp_order)
    sip_bp = get_sip_matrix("BP", max_apbp_order)

    return _FastTanSipWCSStruct(crval, crpix, pc, sip_a, sip_b, sip_ap, sip_bp)


class FastTanSipWCS:
    """
    Lightweight, Numba-optimized WCS implementation for tangent (TAN) projection
    with SIP distortion.

    This class is designed as a replacement for `astropy.wcs.WCS`
    specifically for handling SPHEREx-like WCS headers where speed is critical.
    Initial purpose was when we have >>10^3 headers already extracted as
    dictionaries or other DB (parquet, SQL) and need fast WCS constructions and
    transformations (typical for modern survey data).

    Assumptions & Limitations
    -------------------------
    If any of these assumptions are violated, results should not be trusted.

    1. **Projection**: Strictly supports `RA---TAN-SIP` / `DEC--TAN-SIP` (gnomonic).
    2. **Units**: `CRVAL` inputs are assumed to be in **Degrees**.
    3. **LONPOLE**: Assumed to be **180.0 degrees** (standard for TAN).
    4. **SIP**: Supports `A_ORDER`, `B_ORDER`, `AP_ORDER`, `BP_ORDER` distortions.
       - A/B matrices apply to relative pixel coordinates (u, v).
       - AP/BP matrices apply to intermediate world coordinates (for inverse).
       - Zero-padding optimization is used for SIP matrices.
    5. **Origin**: `all_pix2world` and `all_world2pix` take `origin` argument
       (0-based or 1-based pixel indexing).
       - Internal kernels use 1-based (FITS standard) logic.

    Performance
    -----------
    See `bench_fast_wcs.py`. Tested on MBP 14" [2024, macOS 15.2, M4Pro(8P+4E/G20c/N16c/48G)]:

    - Construction from header dict: ~50x faster than Astropy WCS.
    - Uses Numba-compiled kernels with Fused 2D Horner's Scheme for SIP evaluation.
       - `all_pix2world`: several times faster than Astropy (10ms for 1M points).
       - `all_world2pix`: several times faster than Astropy (300ms for 1M points;
         Slower than all_pix2world, as it uses a Newton-Raphson iterative
         solver for precision <0.001 pix).

    Parameters
    ----------
    hdr_dict : dict
        Dictionary-like object containing WCS header keywords.

    parallel_threshold : int, optional
        Threshold for switching to parallel processing in batch transformations.
        For input arrays smaller than this size, a serial loop is used to avoid
        parallel overhead. Applies only to `all_pix2world`. `all_world2pix` is
        always serial due to Newton-Raphson thread divergence. Default is ``2000``.
    """

    __slots__ = ("_wcs", "parallel_threshold")

    def __init__(self, hdr_dict, parallel_threshold=2000):
        self._wcs = create_wcs_from_dict(hdr_dict)
        self.parallel_threshold = parallel_threshold

    def all_pix2world(self, x, y, origin):
        """ Convert pixel coordinates to world coordinates.
        Parameters
        ----------
        x, y : array-like
            The pixel coordinates.

        origin : int
            The origin of the pixel coordinates (0 or 1).

        Returns
        -------
        ra, dec : array-like
            The world coordinates (RA, Dec) corresponding to the input pixels.

        Notes
        -----
        Mimics the interface of `astropy.wcs.WCS.all_pix2world`.
        """
        # Prepare inputs
        x = np.asanyarray(x)
        y = np.asanyarray(y)
        orig_shape = x.shape

        x = np.atleast_1d(x).astype(np.float64)
        y = np.atleast_1d(y).astype(np.float64)

        # Optim: Flatten to 1D to help Numba vectorization
        x_flat = x.ravel()
        y_flat = y.ravel()

        if x_flat.size != y_flat.size:
            raise ValueError("x and y must have same size")

        ra, dec = _batch_pix2world(x_flat, y_flat, self._wcs, origin, self.parallel_threshold)

        return ra.reshape(orig_shape), dec.reshape(orig_shape)

    def all_world2pix(self, ra, dec, origin):
        """ Convert world coordinates to pixel coordinates.

        Parameters
        ----------
        ra, dec : array-like
            The world coordinates (RA, Dec).
        origin : int
            The origin of the pixel coordinates (0 or 1).

        Returns
        -------
        x, y : array-like
            The pixel coordinates corresponding to the input world coordinates.

        Notes
        -----
        Mimics the interface of `astropy.wcs.WCS.all_world2pix`.
        """
        # Prepare inputs
        ra = np.asanyarray(ra)
        dec = np.asanyarray(dec)
        orig_shape = ra.shape

        ra = np.atleast_1d(ra).astype(np.float64)
        dec = np.atleast_1d(dec).astype(np.float64)

        ra_flat = ra.ravel()
        dec_flat = dec.ravel()

        # Newton-Raphson works better in serial
        x, y = _batch_world2pix(ra_flat, dec_flat, self._wcs, origin)

        return x.reshape(orig_shape), y.reshape(orig_shape)

    def __repr__(self):
        return f"<FastTanSipWCS: CRVAL={self._wcs.crval}, CRPIX={self._wcs.crpix}>"


@njit(parallel=True, fastmath=True)
def _batch_pix2world(x, y, wcs, origin, threshold):
    # Assume x, y are contiguous 1D arrays
    n = x.shape[0]
    ra = np.empty(n, dtype=np.float64)
    dec = np.empty(n, dtype=np.float64)
    offset = 0.0 if origin == 0 else 1.0

    # Avoid parallel overhead for small N
    if n < threshold:
        for i in range(n):
            ra[i], dec[i] = _pix2world_kernel(
                x[i] - offset,
                y[i] - offset,
                wcs.crpix,
                wcs.crval,
                wcs.pc,
                wcs.sip_a,
                wcs.sip_b,
                wcs.a_order,
                wcs.b_order,
                wcs.has_sip,
            )
        return ra, dec

    # Prange with direct indexing is better for SIMD
    for i in prange(n):
        ra[i], dec[i] = _pix2world_kernel(
            x[i] - offset,
            y[i] - offset,
            wcs.crpix,
            wcs.crval,
            wcs.pc,
            wcs.sip_a,
            wcs.sip_b,
            wcs.a_order,
            wcs.b_order,
            wcs.has_sip,
        )
    return ra, dec


@njit(fastmath=False)
def _batch_world2pix(ra, dec, wcs, origin):
    n = ra.shape[0]
    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    offset = 0.0 if origin == 0 else 1.0

    # Avoid parallel overhead for small N
    for i in range(n):
        fx, fy = _world2pix_kernel(
            ra[i],
            dec[i],
            wcs.crpix,
            wcs.crval,
            wcs.pc,
            wcs.pc_inv,
            wcs.sip_a,
            wcs.sip_b,
            wcs.sip_ap,
            wcs.sip_bp,
            wcs.a_order,
            wcs.b_order,
            wcs.ap_order,
            wcs.bp_order,
            wcs.has_sip,
            wcs.has_sip_inv,
        )
        x[i] = fx + offset
        y[i] = fy + offset
    return x, y


def all_world2pix_infov(wcs, *args, naxes=None, bezels=0.5, **kwargs):
    """
    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        The WCS object.

    naxes : array-like of int, optional
        The number of pixels along each axis. If not provided, it defaults
        first to `wcs._naxis`, and if it does not exist, the maximum values of
        `x` and `y`.

    bezels : int or 2-D array-like, optional
        The number of pixels to drop from each edge of the image. If an int,
        it is applied to all edges. If 2-D, it should be like ``[[left, right], [bottom, top]]``. Use negative values to accept objects slightly
        outside the image. Default is ``0.5``.

    Returns
    -------
    pixels : ndarray
        The pixel coordinates (N, 2).
    infov : 1-D boolean array
        A boolean array indicating if the points are within the image.

    """
    try:
        xpix, ypix = wcs.all_world2pix(*args, **kwargs)
        xpix = np.asarray(xpix)
        ypix = np.asarray(ypix)
        pixels = np.column_stack((xpix, ypix))  # shape (N, 2)
    except ValueError:  # args is (N, 2) shape
        pixels = wcs.all_world2pix(*args, **kwargs)

    if naxes is None:
        naxes = getattr(wcs, "_naxis", None)
        if naxes is not None:
            for _n in np.atleast_1d(naxes):
                if _n < 1:
                    raise ValueError(
                        "`wcs._naxis`, which should be the number of pixels along each axis, "
                        f"must be a positive integer or array-like. Got this: {naxes = }."
                        f" Either use proper WCS or give `naxes` explicitly."
                    )
    infov = infov2d(pixels[:, 0], pixels[:, 1], bezels=bezels, naxes=naxes)

    return pixels, infov


def infov2d(x, y, bezels=0.5, naxes=None):
    """Create a 2-D boolean array indicating if the points are within the image
    defined by `naxes` and `bezels`.

    Parameters
    ----------
    x, y : array-like
        The x and y coordinates of the points.

    bezels : int or 2-D array-like, optional
        The number of pixels to drop from each edge of the image. If an int,
        it is applied to all edges. If 2-D, it should be like ``[[left, right], [bottom, top]]``. Use negative values to accept objects slightly
        outside the image.
        Default is ``0.5`` because python assigns values -0.5 to 0.5 to be the 0-th
        pixel.

    naxes : array-like of int, optional
        The number of pixels along each axis. If not provided, it defaults to
        the maximum values of `x` and `y`. Default is `None`.

    Returns
    -------
    infov : 1-D boolean array
        A boolean array indicating if the points are within the image.
    """
    if naxes is None:
        naxes = [max(x), max(y)]
    elif isinstance(naxes, (int, float)):
        naxes = [naxes, naxes]
    elif len(naxes) != 2:
        raise ValueError(f"naxes must have 2 entries, got {len(naxes)=}.")

    if isinstance(bezels, (int, float)):
        bezels = [[bezels, bezels]] * 2

    infov = (
        (bezels[0][0] <= x)
        & (x <= naxes[0] - bezels[0][1])
        & (bezels[1][0] <= y)
        & (y <= naxes[1] - bezels[1][1])
    )

    return infov


