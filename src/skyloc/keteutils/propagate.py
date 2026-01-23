from pathlib import Path
import logging

import kete
import numpy as np
import pandas as pd

from ._constants import KETE_ASTEROIDS_PHYSICS

logger = logging.getLogger(__name__)

__all__ = [
    "make_nongravs_models",
    "orb2state_propagate",
    "calc_geometries",
    "replace_loaded_with_spice",
]

DEGPERDAY2ARCSECPERMIN = 3600 / (24 * 60)


def replace_loaded_with_spice(states, jd):
    """Replace states of loaded asteroids with SPICE data.

    When `include_asteroids=True` is used in `kete.propagate_n_body`, the 5 large
    asteroids (Ceres, Pallas, Vesta, Hygiea, Interamnia) are used as perturbers.
    If these objects are also being propagated, `kete` may produce NaN or inaccurate
    states due to self-collision detection. This function replaces those states
    with authoritative SPICE data.

    Parameters
    ----------
    states : list of `~kete.State`
        The propagated states from `kete.propagate_n_body`.

    jd : float
        The JD (TDB) at which to fetch SPICE states.

    Returns
    -------
    new_states : list of `~kete.State`
        States with loaded asteroids replaced by SPICE data.

    Notes
    -----
    This function checks each state's designation against the 5 loaded asteroids
    (by SBDB number or name) and replaces matching states with SPICE-fetched data.
    Non-matching states are passed through unchanged.
    """
    # Loaded asteroid designations (SBDB numbers) and names
    loaded_desigs = set(KETE_ASTEROIDS_PHYSICS.keys())  # {"1", "2", "4", "10", "704"}
    loaded_names = {v["name"]: d for d, v in KETE_ASTEROIDS_PHYSICS.items()}

    new_states = []
    for state in states:
        desig = state.desig
        matched_name = None

        # Check by designation (e.g., "2") or by name (e.g., "pallas")
        if desig in loaded_desigs:
            matched_name = KETE_ASTEROIDS_PHYSICS[desig]["name"]
        elif desig.lower() in loaded_names:
            matched_name = desig.lower()

        if matched_name is not None:
            spice_state = kete.spice.get_state(matched_name, jd)
            new_states.append(spice_state)
            logger.debug("Replaced n-body state for %s with SPICE data", desig)
        else:
            new_states.append(state)

    return new_states


def make_nongravs_models(
    orb,
    m_ng=None,
    c_a1="A1",
    c_a2="A2",
    c_a3="A3",
    c_dt="DT",
    alpha=0.1112620426,
    r_0=2.808,
    m=2.15,
    n=5.093,
    k=4.6142,
    c_alpha="alpha",
    c_r0="r_0",
    c_m="m",
    c_n="n",
    c_k="k",
):
    """Create a list of non-gravitational models for the objects in the orbit file.

    Parameters
    ----------
    orb : pd.DataFrame
        Orbit file with columns of orbital elements.

    m_ng : bool, array-like of bool, optional
        A boolean mask indicating which objects have non-gravitational
        parameters. If not provided, it will be calculated based on the
        columns `c_a1`, `c_a2`, `c_a3`, and `c_dt` (mask is `True` when any
        of these columns is not zero).

    c_a1, c_a2, c_a3, c_dt : str, optional
        Column names in the orbit file for the non-gravitational parameters.
        Default is "A1", "A2", "A3", and "DT".

    c_alpha, c_r0, c_m, c_n, c_k : str, optional
        Column names in the orbit file for the non-gravitational model
        parameters `r_0`, `m`, `n`, and `k`, respectively.

    alpha, r_0, m, n, k : float or array-like, optional
        Default parameters for the non-gravitational model. If `orb` contains
        columns for these (`c_alpha`, `c_r0`, `c_m`, `c_n`, `c_k`), the
        values will be used as `.fillna()` defaults if scalar or `orb[col] =
        value` if array-like.

    Returns
    -------
    non_gravs : list of `~kete.NonGravModel`
        A list of non-gravitational models for each object in the orbit file.
        If an object does not have a non-gravitational model, it will be `None`.

    """
    # Broadcast parameters to match orb length if needed
    nobj = len(orb)

    if m_ng is None:  # If not provided...
        m_ng = (
            ((orb[c_a1] != 0.0) & ~pd.isna(orb[c_a1]))
            | ((orb[c_a2] != 0.0) & ~pd.isna(orb[c_a2]))
            | ((orb[c_a3] != 0.0) & ~pd.isna(orb[c_a3]))
            | ((orb[c_dt] != 0.0) & ~pd.isna(orb[c_dt]))
        )

    if isinstance(m_ng, bool):
        m_ng = np.array([m_ng] * nobj)
    else:  # must be array-like of bool
        m_ng = np.asarray(m_ng, dtype=bool)  # avoid copy if possible
        if len(m_ng) != nobj:
            raise ValueError(f"{len(m_ng)=} does not match number of objects {nobj}")

    def _set_nongrav_col(df, colname, value):
        """Set the column in the orbit DataFrame to the value."""
        if colname in orb.columns:
            # If scalar value
            df[colname] = df[colname].fillna(value)
        else:  # If the column does not exist in the DataFrame,
            # Fill with scalar or array value
            df[colname] = value

    orb_ng = orb.loc[m_ng].copy()
    for col, val in zip(
        [c_alpha, c_r0, c_m, c_n, c_k],
        [alpha, r_0, m, n, k],
    ):
        _set_nongrav_col(orb_ng, col, val)

    non_gravs = np.array([None] * nobj)

    # Extract column arrays once (much faster than .iterrows() row-by-row access)
    ng_indices = np.where(m_ng)[0]
    a1_vals = orb_ng[c_a1].values
    a2_vals = orb_ng[c_a2].values
    a3_vals = orb_ng[c_a3].values
    dt_vals = orb_ng[c_dt].values
    alpha_vals = orb_ng[c_alpha].values
    r0_vals = orb_ng[c_r0].values
    m_vals = orb_ng[c_m].values
    n_vals = orb_ng[c_n].values
    k_vals = orb_ng[c_k].values

    for j, i in enumerate(ng_indices):
        ng = kete.propagation.NonGravModel.new_comet(
            a1=a1_vals[j],
            a2=a2_vals[j],
            a3=a3_vals[j],
            dt=dt_vals[j],
            alpha=alpha_vals[j],
            r_0=r0_vals[j],
            m=m_vals[j],
            n=n_vals[j],
            k=k_vals[j],
        )
        non_gravs[i] = ng

    return non_gravs


def orb2state_propagate(
    orb,
    jd,
    include_asteroids=False,
    non_gravs=None,
    suppress_errors=True,
    output=None,
    overwrite=False,
):
    """Convert the orbit file to state vector and propagate the orbit.

    Parameters
    ----------
    orb : pd.DataFrame
        Orbit file with columns of orbital elements.

    jd : float
        A JD (TDB) to propagate the initial states to.

    include_asteroids : bool, optional
        If this is `True`, the computation will include the largest 5 asteroids.
        The asteroids are: Ceres, Pallas, Interamnia, Hygiea, and Vesta.
        Default is `False`.

    non_gravs : list
        A list of non-gravitational terms for each object. If provided, then
        every object must have an associated `NonGravModel` or `None`.
        Default is `None`.

    suppress_errors : bool, optional
        If `True`, errors during propagation will return NaN for the relevant
        state vectors, but propagation will continue. Default is `True`.

    output : str, optional
        If provided, save the propagated states to a parquet file.
        Default is `None`.

    overwrite : bool, optional
        If `True`, re-do the calculation and overwrite the existing output file
        if it exists. If `False`, the `output` will be loaded if it exists.

    Returns
    -------
    states : list of `~kete.State`
        The initial states of the objects in the orbit file.

    states0 : list of `~kete.State`
        The propagated states of the objects at the given JD.

    """
    if output is not None and Path(output).exists() and not overwrite:
        states = None
        states0 = kete.SimultaneousStates.load_parquet(output)

    else:
        # Convert to kete State object
        try:
            states = kete.mpc.table_to_states(orb)
        except AttributeError:
            # in case desig is index, not in column.
            orb["desig"] = orb.index
            states = kete.mpc.table_to_states(orb)
            del orb["desig"]

        states0 = kete.propagate_n_body(
            states,
            jd=jd,
            include_asteroids=include_asteroids,
            non_gravs=non_gravs,
            suppress_errors=suppress_errors,
        )

        if output is not None:
            kete.SimultaneousStates(states0).save_parquet(str(output))

    return states, states0


def calc_geometries(
    simulstates,
    center=10,
    do_alpha=True,
    do_dists=True,
    do_radec=True,
    do_hel_ecl=True,
    do_obs_ecl=True,
    rates_in_arcsec_per_min=True,
):
    """Calculate phase angle alpha, distances, and lon/lats.

    Parameters
    ----------
    simulstates : `~kete.SimultaneousStates`
        The simultaneous states of the objects.

    center : int, optional
        The center of the light source. Default is 10 (Sun).

    do_alpha, do_dists, do_radec, do_hel_ecl, do_obs_ecl : bool, optional
        If `True`, calculate the phase angle, heliocentric/observer-centric
        distances, right ascension and declination, heliocentric ecliptic
        coordinates, and observer-centric ecliptic coordinates, respectively.
        Default is `True` for all. Note that if `do_alpha=True`, distances
        (r_hel, r_obs) are always computed and included in the output because
        they are required for the phase angle calculation.

    rates_in_arcsec_per_min : bool, optional
        If `True`, the rates of right ascension and declination will be
        converted to arcsec/min. Default is `True`.
        If `False`, they will be in degrees/day (kete default).

    Returns
    -------
    geoms : dict
        The geometries of the objects, including the following:

        - alpha : np.ndarray
            The phase angles in degrees.
        - r_hel, r_obs : np.ndarray
            The heliocentric and observer-centric distances in AU.
        - ra, dec : np.ndarray
            The observer-centric (J2000) right ascension and declination in degrees.
        - dra*cosdec/dt, ddec/dt : np.ndarray
            The observer-centric (J2000) right ascension and declination rates
            in degrees per day (or arcsec/min if `rates_in_arcsec_per_min` is
            `True`).
        - hel_ecl_lons, hel_ecl_lats : np.ndarray
            The heliocentric ecliptic longitudes and latitudes in degrees.
        - obs_ecl_lons, obs_ecl_lats : np.ndarray
            The observer-centric ecliptic longitudes and latitudes in degrees
            (e.g., the observer is the Earth).
        - sky_motion, sky_motion_pa : np.ndarray
            The apparent angular rate of the target in the plane-of-sky in
            degrees/day (or arcsec/min if `rates_in_arcsec_per_min` is `True`),
            and the position angle in degrees. PA is positive if RA rate is
            positive (i.e., measured counter-clockwise from the apparent
            of-date north pole direction, which is identical to JPL Horizons)

    Notes
    -----
    Takes ~1 ms for 1000 objects (MBP 14" [2024, macOS 15.2,
    M4Pro(8P+4E/G20c/N16c/48G)])
    """
    # obs2objs = simulstates.obs_vecs

    _obs = simulstates.fov.observer
    if _obs.center_id != center:
        sun2obs_pos = _obs.change_center(center).pos
    else:
        sun2obs_pos = _obs.pos

    # Convert observer position to numpy array (3,)
    sun2obs_xyz = np.array([sun2obs_pos[0], sun2obs_pos[1], sun2obs_pos[2]])

    n_states = len(simulstates.states)
    desigs = []
    sun2obj_xyz = np.empty((n_states, 3), dtype=np.float64)

    for i, _state in enumerate(simulstates):
        desigs.append(_state.desig)
        # heliocentric position
        if _state.center_id != center:
            # Generally _state is relative to the Sun by kete, so it will be
            # rare for this `change_center` (~10us) to actually be calculated.
            _pos = _state.change_center(center).pos
        else:
            _pos = _state.pos
        sun2obj_xyz[i, 0] = _pos[0]
        sun2obj_xyz[i, 1] = _pos[1]
        sun2obj_xyz[i, 2] = _pos[2]

    # Observer-to-object vectors: obs2obj = sun2obj - sun2obs
    # Shape: (n_states, 3)
    obs2obj_xyz = sun2obj_xyz - sun2obs_xyz

    # Initialize result dictionary
    geoms = {"desig": np.array(desigs)}

    # Phase angle alpha: angle between (-obs2obj) and (-sun2obj)
    # This is the Sun-Object-Observer angle
    # cos(alpha) = dot(-obs2obj, -sun2obj) / (|obs2obj| * |sun2obj|)
    #            = dot(obs2obj, sun2obj) / (|obs2obj| * |sun2obj|)
    if do_alpha or do_dists:
        # NOTE: r_hel and r_obs are computed if do_alpha or do_dists is True.
        r_hel = np.linalg.norm(sun2obj_xyz, axis=1)  # |sun2obj|
        r_obs = np.linalg.norm(obs2obj_xyz, axis=1)  # |obs2obj|

        if do_alpha:
            # Dot product of obs2obj and sun2obj for each object
            dot_product = np.sum(obs2obj_xyz * sun2obj_xyz, axis=1)
            # Clamp to [-1, 1] to avoid numerical issues with arccos
            cos_alpha = dot_product / (r_obs * r_hel)
            cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
            geoms["alpha"] = np.rad2deg(np.arccos(cos_alpha))

        # Helio/observer-centric distances
        # If do_alpha=True, distances are always returned (required for alpha)
        if do_alpha or do_dists:
            geoms["r_hel"] = r_hel
            geoms["r_obs"] = r_obs

    # Heliocentric ecliptic coordinates of the object
    # lon = atan2(y, x), lat = atan2(z, sqrt(x^2 + y^2))
    if do_hel_ecl:
        geoms["hel_ecl_lon"] = np.rad2deg(np.arctan2(sun2obj_xyz[:, 1], sun2obj_xyz[:, 0]))
        xy_hel = np.hypot(sun2obj_xyz[:, 0], sun2obj_xyz[:, 1])
        geoms["hel_ecl_lat"] = np.rad2deg(np.arctan2(sun2obj_xyz[:, 2], xy_hel))

    # Observer-centric ecliptic coordinates of the object
    if do_obs_ecl:
        geoms["obs_ecl_lon"] = np.rad2deg(np.arctan2(obs2obj_xyz[:, 1], obs2obj_xyz[:, 0]))
        xy_obs = np.hypot(obs2obj_xyz[:, 0], obs2obj_xyz[:, 1])
        geoms["obs_ecl_lat"] = np.rad2deg(np.arctan2(obs2obj_xyz[:, 2], xy_obs))

    # RA/Dec and rates from kete's built-in method (already vectorized)
    if do_radec:
        radec = np.atleast_2d(simulstates.ra_dec_with_rates)
        geoms["ra"] = radec[:, 0]
        geoms["dec"] = radec[:, 1]
        if rates_in_arcsec_per_min:
            geoms["dra*cosdec/dt"] = radec[:, 2] * DEGPERDAY2ARCSECPERMIN
            geoms["ddec/dt"] = radec[:, 3] * DEGPERDAY2ARCSECPERMIN
        else:
            geoms["dra*cosdec/dt"] = radec[:, 2]
            geoms["ddec/dt"] = radec[:, 3]
        geoms["sky_motion"] = np.hypot(geoms["dra*cosdec/dt"], geoms["ddec/dt"])
        geoms["sky_motion_pa"] = np.rad2deg(
            np.arctan2(geoms["dra*cosdec/dt"], geoms["ddec/dt"])
        )
    return geoms


def horizonscov2elems(obj, n_samples=1000, seed=None):
    """ Generate samples from a covariance matrix for HorizonsProperties

    Parameters
    ----------
    obj : `~kete.HorizonsProperties`
        The HorizonsProperties object to sample from.

    n_samples : int, optional
        The number of samples to generate. Default is 1000.

    seed : int, optional
        The random seed for reproducibility. Default is None.

    Notes
    -----
    kete only returns `state` while some may want just the elements.
    """
    pass



'''
Deprecated - left for reference
I didn't know there was ra_dec_with_rates in the SimultaneousStates class.

def calc_skymotion(simulstates, jd_mid, observer_pos=None, dt_sec=60):
    """Calculate apparent angular rate of the target in the plane-of-sky.

    Parameters
    ----------
    simulstates : `~kete.SimultaneousStates`
        The simultaneous states of the objects.

    jd_mid : float
        The TDB time of the mid-point of the observation in JD.

    observer_pos : `~kete.Vector`, optional
        The position of the observer in au, to be used in
        `~kete.propagate_two_body`. If this is provided then the estimated
        states will be returned as a result of light propagation delay.

    dt_sec : float, optional
        The time step in seconds. Motion will be calculated based on the
        difference between the two states at `jd_mid - dt_sec/2` and
        `jd_mid + dt_sec/2`. Default is 60 seconds.
    """
    _dt = dt_sec / 2 / 86400  # convert to days
    sta1 = kete.propagate_two_body(simulstates, jd_mid - _dt, observer_pos=observer_pos)
    sta2 = kete.propagate_two_body(simulstates, jd_mid + _dt, observer_pos=observer_pos)

    sky_motion = []
    sky_motion_pa = []

    for sta1_, sta2_ in zip(sta1, sta2):
        # apparent angular rate of the target in the plane-of-sky
        # Interestingly,
        #    erfa.pap(sta1_eq.pos, sta2_eq.pos): 3.0 us
        #    erfa.pas(np.deg2rad(sta1_eq.pos.ra), np.deg2rad(sta1_eq.pos.dec),
        #             np.deg2rad(sta2_eq.pos.ra), np.deg2rad(sta2_eq.pos.dec)): 1.9 us
        # Almost identical for sepp VS seps.
        # This is likely because pyERFA wrapper uses higher-level tricks for sepp/pap,
        # similar to astropy skycoord. Thus, better avoid them.
        sta1_eq = sta1_.as_equatorial.pos
        sta2_eq = sta2_.as_equatorial.pos


    geoms = dict(desig=[], alpha=[], r_hel=[], r_obs=[], ra=[], dec=[],
                 hel_ecl_lon=[], hel_ecl_lat=[], obs_ecl_lon=[], obs_ecl_lat=[])

    # _eq: equatorial; otherwise, ecliptic.
    sun2obj_poss = []
    # sun2obj_eq_poss = []
    sun2obj_eq_vels = []
    obs2obj_poss = []
    obs2obj_eq_poss = []
    obs2obj_eq_vels = []

    for _state in simulstates:
        geoms["desig"].append(_state.desig)

        # heliocentric position
        if _state.center_id != center:
            sun2obj_state = _state.change_center(center).as_ecliptic
        else:
            sun2obj_state = _state.as_ecliptic

        sun2obj_poss.append(sun2obj_state.pos)
        # sun2obj_eq_poss.append(sun2obj_state.pos.as_equatorial)
        sun2obj_eq_vels.append(sun2obj_state.vel.as_equatorial)

        obs2obj_pos = sun2obj_state.pos - sun2obs_state.pos
        obs2obj_vel = sun2obj_state.vel - sun2obs_state.vel

        obs2obj_poss.append(obs2obj_pos)
        obs2obj_eq_poss.append(obs2obj_pos.as_equatorial)
        obs2obj_eq_vels.append(obs2obj_vel.as_equatorial)

    # kete Vector can easily be converted to numpy arrays:
    # sun2obj_poss = np.atleast_2d(sun2obj_poss)
    # # sun2obj_eq_poss = np.atleast_2d(sun2obj_eq_poss)
    # sun2obj_eq_vels = np.atleast_2d(sun2obj_eq_vels)
    # obs2obj_poss = np.atleast_2d(obs2obj_poss)
    obs2obj_eq_poss = np.atleast_2d(obs2obj_eq_poss)
    obs2obj_eq_vels = np.atleast_2d(obs2obj_eq_vels)

    # r_hels = np.linalg.norm(sun2obj_poss, axis=1)
    r_obss = np.linalg.norm(obs2obj_poss, axis=1)
    # Phase angle alpha
    # alphas = np.rad2deg(np.arccos(
        # np.sum(sun2obj_poss*obs2obj_poss, axis=1) / (r_hels * r_obss)
    # ))
    # hel_ecl_lons = np.rad2deg(np.arctan2(sun2obj_poss[:, 1], sun2obj_poss[:, 0])) % 360
    # hel_ecl_lats = np.rad2deg(np.arcsin(sun2obj_poss[:, 2] / r_hels))
    # obs_ecl_lons = np.rad2deg(np.arctan2(obs2obj_poss[:, 1], obs2obj_poss[:, 0])) % 360
    # obs_ecl_lats = np.rad2deg(np.arcsin(obs2obj_poss[:, 2] / r_obss))
    # RA/Dec
    ra = np.rad2deg(np.arctan2(obs2obj_eq_poss[:, 1], obs2obj_eq_poss[:, 0])) % 360
    dec = np.rad2deg(np.arcsin(obs2obj_eq_poss[:, 2] / r_obss))
    # RA/Dec rates


    # print(hel_ecl_lons[0], hel_ecl_lats[0], obs_ecl_lons[0], obs_ecl_lats[0])
    # print(ra[0], dec[0])
    # print(ra)
    # print(dec)
    return

'''
