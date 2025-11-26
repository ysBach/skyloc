from functools import cached_property
from pathlib import Path

import kete
import numpy as np
import pandas as pd

from .configs import MINIMUM_ORB_COLS
from .keteutils.fov import FOVCollection
from .keteutils.propagate import calc_geometries, make_nongravs_models
from .keteutils._util import KETE_LOADED_ASTEROIDS
from .ssoflux import comet_mag, iau_hg_mag
from .utils import listmask, tdb2utc

__all__ = ["locator_twice", "SSOLocator", "SpiceLocator", "calc_ephems"]

_EPH_DTYPES = {
    "alpha": np.float32,
    "vmag": np.float32,
    "r_obs": np.float32,
    "r_hel": np.float32,
    "dra*cosdec/dt": np.float32,
    "ddec/dt": np.float32,
    "sky_motion": np.float32,
    "sky_motion_pa": np.float32,
    "obsindex": np.uint32,
}


def locator_twice(
    fovs,
    orb,
    simstates=None,
    drop_major_asteroids=True,
    jd0=(np.mean, np.mean),
    include_asteroids=(False, True),
    dt_limit=(5.0, 0.5),
    suppress_errors=True,
    add_obsid=True,
    drop_obsindex=True,
    add_jds=True,
    calc_ephems_crude=False,
):
    """Simple utility function to run SSOLocator twice

    Parameters
    ----------
    fovs : FOVCollection, iterable of `kete.fov.FOV`, or `kete.fov.FOVList`
        Field of View(s) to check the propagated states against.

    orb : `pandas.DataFrame`
        Orbit file with columns of orbital elements. See
        `query.fetch_orb()`.

    simstates : `~kete.SimultaneousStates`, path-like, optional
        If provided, use it as the initial states for propagation.
        load the existing propagated states from the given
        parquet file instead of propagating the orbits again for the first
        calculation. Default is `None`.

    Notes
    -----
    Arguments `jd0`, `include_asteroids`, and `dt_limit` must be 2-tuple, such
    that 0-th is used for the first crude calculation and 1-th is used for the
    second refined calculation.

    First crude calculation is meant to be used with `include_asteroids=False`
    and large `dt_limit`, to check which objects will be in the FoVs.

    In contrast, the second refined calculation is meant to be used with
    `include_asteroids=True` and a smaller `dt_limit`, to get more accurate
    results for the objects that are actually in the FoVs.

    Even for crude case, non_gravs=True is hard-coded because it may give big
    diffences. Anyways, there are few objects with non-grav terms.
    """
    if simstates is not None:
        if not isinstance(simstates, kete.SimultaneousStates):
            simstates = kete.SimultaneousStates.load_parquet(simstates)

    sl1 = SSOLocator(
        fovs=fovs,
        orb=orb,
        simstates=simstates,
        non_gravs=True,
        drop_major_asteroids=drop_major_asteroids
    )
    sl1.propagate_n_body(
        jd0=jd0[0],
        suppress_errors=suppress_errors,
        include_asteroids=include_asteroids[0],
    )
    sl1.fov_state_check(dt_limit=dt_limit[0], include_asteroids=include_asteroids[0])

    if calc_ephems_crude:
        sl1.calc_ephems(
            add_obsid=add_obsid, drop_obsindex=drop_obsindex, add_jds=add_jds
        )

    sl2 = SSOLocator(
        fovs=sl1.fovc_hasobj,
        orb=sl1.orb.loc[sl1.orb_infov_mask].copy(),
        simstates=simstates,
        non_gravs=True,
        drop_major_asteroids=drop_major_asteroids,
    )
    sl2.propagate_n_body(
        jd0=jd0[1],
        suppress_errors=suppress_errors,
        include_asteroids=include_asteroids[1],
    )
    sl2.fov_state_check(dt_limit=dt_limit[1], include_asteroids=include_asteroids[1])
    sl2.calc_ephems(
        add_obsid=add_obsid, drop_obsindex=drop_obsindex, add_jds=add_jds
    )

    return sl1, sl2


class Locator:
    """A base class for locators."""

    def __init__(self, fovs):
        if fovs is None:
            self._fovc = None
        elif isinstance(fovs, FOVCollection):
            self._fovc = fovs
        else:
            self._fovc = FOVCollection(fovs)

    @property
    def fovc(self):
        """Get the FOVCollection."""
        return self._fovc

    @fovc.setter
    def fovc(self, fovs):
        """Set the FOVCollection."""
        if fovs is None:
            self._fovc = None
        elif isinstance(fovs, FOVCollection):
            self._fovc = fovs
        else:
            self._fovc = FOVCollection(fovs)

    @fovc.deleter
    def fovc(self):
        """Delete the FOVCollection."""
        self._fovc = None

    def fov_static_check(self):
        """Check which objects are in the FOVs."""

        pass

    # for planet in ["mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]:
    #     # do for planets
    #     simstas.extend(kete.fov.fov_spice_check(planet, self.fovc.fovlist))


class SpiceLocator(Locator):
    """A class to locate objects in a given Field of View (FOV) using SPICE.

    Good examples are planets and satellites, but it can be used for any
    object that has a SPICE kernel loaded.
    """

    def __init__(self, fovs, desigs):
        super().__init__(fovs)
        self.desigs = desigs
        self.loaded = kete.spice.loaded_objects()

    def load_spice(self, fpaths):
        kete.spice.load_spice(fpaths)
        self.loaded = kete.spice.loaded_objects()


class StarLocator(Locator):
    """A class to locate sidereal objects in a given Field of View (FOV).

    This class provides methods to check if stars are in the FOV and to
    calculate their positions.

    Plan: include proper motions

    Attributes
    ----------
    fovs : FOVCollection
        A collection of FOVs to check against.
    """

    def __init__(self, fovs):
        super().__init__(fovs)

    def fov_static_check(self):
        """Check which stars are in the FOVs."""
        # Implement star-specific logic here
        pass


class SSOLocator(Locator):
    """A class to locate Solar System Objects (SSOs) in a given Field of View (FOV).

    This class provides methods to propagate orbits, check if objects are in the
    FOV, and calculate ephemerides for the objects.

    Attributes
    ----------
    fovs : FOVCollection
        A collection of FOVs to check against.
    """

    def __init__(
        self, fovs, orb, simstates=None, non_gravs=True, copy_orb=False, drop_major_asteroids=True
    ):
        """
        Parameters
        ----------
        fovs : FOVCollection, iterable of `kete.fov.FOV`, or `kete.fov.FOVList`
            Field of View(s) to check the propagated states against.

        orb : `pandas.DataFrame`
            Orbit file with columns of orbital elements. See
            `query.fetch_orb()`.

        simstates : `~kete.SimultaneousStates`, path-like, optional
            If provided, use it as the initial states for propagation.

        non_gravs : list, bool, optional
            A list of non-gravitational terms for each object. If provided, then
            every object must have an associated `kete.NonGravModel` or `None`.
            If `True`, `keteutils.make_nongravs_models` will be used, i.e.,
            objects with non-gravitational terms will have non-gravitational
            models by `kete.NonGravModel.new_comet()`.
            If `False`, no non-gravitational terms will be used for any object.
            Default is `True`.

        drop_major_asteroids : bool, optional
            If `True`, drop major asteroids (KETE_LOADED_ASTEROIDS) from the
            `orb`.
            Default is `True`.
        """
        super().__init__(fovs)
        if copy_orb:
            orb = orb.copy()
        self._validate_orb(orb)
        if drop_major_asteroids:
            orb = orb[~orb.desig.isin(KETE_LOADED_ASTEROIDS)]
        self._orb = orb

        if isinstance(non_gravs, bool):
            if non_gravs:
                self.non_gravs = make_nongravs_models(orb)
            else:
                self.non_gravs = [None] * len(orb)
        else:
            self.non_gravs = list(non_gravs)

        if simstates is None:
            self.jd0 = None
            self.states_propagated_jd0 = None
        else:
            try:
                self.jd0 = simstates.jd
                self.states_propagated_jd0 = simstates
            except AttributeError:
                # simstates in path like
                simstates = kete.SimultaneousStates.load_parquet(simstates)
                self.jd0 = simstates.jd
                self.states_propagated_jd0 = simstates.states

    def _validate_orb(self, orb):
        """Validate the orbit DataFrame."""
        if not isinstance(orb, pd.DataFrame):
            raise TypeError("orb must be a pandas DataFrame.")

        missing = [col for col in MINIMUM_ORB_COLS if col not in orb.columns]
        if missing:
            raise ValueError(
                "The orbit DataFrame is missing required columns: " + ", ".join(missing)
            )

    @property
    def orb(self):
        return self._orb

    @orb.setter
    def orb(self, orb):
        self._validate_orb(orb)
        self._orb = orb

    @orb.deleter
    def orb(self):
        """Delete the orbit DataFrame."""
        self._orb = None

    @cached_property
    def states_from_orb(self):
        """Get the initial states of the objects in the orbit file.
        ~1s per 1M objects in orb at first call
        on MBP 14" [2024, macOS 15.2, M4Pro(8P+4E/G20c/N16c/48G)]
        """
        return kete.mpc.table_to_states(self.orb)

    def propagate_n_body(
        self,
        jd0=np.mean,
        suppress_errors=True,
        include_asteroids=False,
        objmask=None,
        output=None,
        overwrite=False,
    ):
        """Propagate the orbits to the mean JD of the FOVs.

        Parameters
        ----------
        jd0 : float, callable, optional
            A JD (TDB) to propagate the initial states to. If callable (e.g.,
            `np.mean`), it will be applied to the `.observer.jd` of the FOVs
            (e.g., `np.mean([fov.observer.jd for fov in self.fovc.fovlist])`).
            Default is `np.mean`, which uses the mean JD of the FOVs.

        suppress_errors : bool, optional
            If `True`, errors during propagation will return NaN for the relevant
            state vectors, but propagation will continue. Default is `True`.

        include_asteroids : bool, optional
            If `True`, include asteroids in the propagation. Default is `False`.

        objmask : array-like, optional
            A boolean mask to select which objects to propagate. It will select
            such as `self.states_from_orb[objmask]` to run the propagation on subset of
            the objects. Thus, it must have the length same as `self.states_from_orb`,
            i.e., same as `self.orb`.

        output : str or `~pathlib.Path`, optional
            If provided, save the propagated states to a parquet file.
            Default is `None`.

        overwrite : bool, optional
            If `True`, re-do the calculation and overwrite the existing output file
            if it exists. If `False`, the `output` will be loaded if it exists.

        Notes
        -----
        If `output` is provided and the file already exists, and `overwrite` is
        `False`, the function will load the existing file instead of redoing
        the calculation. This means a completely meaningless results can be
        returned if the file is not up to date.
        """
        output_exists = False
        if output is not None:
            output = str(output)
            output_exists = Path(output).exists()

        if not isinstance(jd0, (float, int)):
            # Use the mean JD of the FOVs
            jd0 = jd0(self.fovc.fov_jds)

        if output_exists and not overwrite:
            states0 = kete.SimultaneousStates.load_parquet(output).states
            self.jd0 = jd0
            self.states_propagated_jd0 = states0
            return

        states0 = kete.propagate_n_body(
            listmask(self.states_from_orb, objmask),
            jd=jd0,
            include_asteroids=include_asteroids,
            non_gravs=listmask(self.non_gravs, objmask),
            suppress_errors=suppress_errors,
        )

        if output is not None or overwrite:
            kete.SimultaneousStates(states0).save_parquet(output)
            # Load the states from the file to ensure consistency
            states0 = kete.SimultaneousStates.load_parquet(output).states

        self.jd0 = jd0
        self.states_propagated_jd0 = states0

    def fov_state_check(self, dt_limit=3.0, include_asteroids=False):
        """Check which objects are in the FOVs after propagation.

        Parameters
        ----------
        dt_limit : float, optional
            Length of time in days where 2-body mechanics is a good approximation.
            Default is 3.0 days.

        include_asteroids : bool, optional
            If `True`, include asteroids in the check. Default is `False`.

        Notes
        -----
        self.fov_check_simstates : list of `kete.SimultaneousStates`
            The states of the objects in the FOVs after propagation.
            Basically the output of `kete.fov.fov_state_check`.

        self.fov_check_fov2objs : dict
            A dictionary mapping FOV designations to lists of object designations
            that are in the FOVs.

        self.fov_check_objids : list of str
            A list of unique object designations that are in the FOVs.
        """
        simstas = kete.fov.fov_state_check(
            self.states_propagated_jd0,
            self.fovc.fovlist,
            dt_limit=dt_limit,
            include_asteroids=include_asteroids,
        )

        # Convenience: Collect the designations of the FOVs and objects
        fov2objs = {}
        # key = FOV's designations (str)
        # value = list of object designations in that FOV (list[str])
        for _ss in simstas:
            _objids = []
            for _s in _ss.states:
                _objids.append(_s.desig)
            fov2objs.setdefault(_ss.fov.observer.desig, []).extend(_objids)

        # For ~1000 elements, below is ~ 10x faster than
        #    np.unique(np.concatenate(list(fov2objs.values())))
        uniq_objids = set()
        for objids in fov2objs.values():
            uniq_objids.update(objids)
        uniq_objids = list(uniq_objids)

        self.fov_check_simstates = simstas
        self.fov_check_fov2objs = fov2objs
        self.fov_check_objids = uniq_objids

        # FOVs with objects
        self.fov_mask_hasobj = self.fovc.mask_by_desig(fov2objs.keys())
        self.fovc_hasobj = FOVCollection(self.fovc[self.fov_mask_hasobj])

        # objects in at least one of FOV
        self.orb_infov_mask = self.orb["desig"].isin(uniq_objids)

    def calc_ephems(
        self,
        gpar_default=0.15,
        sort_by=["vmag"],
        dtypes=_EPH_DTYPES,
        add_obsid=True,
        drop_obsindex=True,
        add_jds=True,
        output=None,
        overwrite=False,
    ):
        """Calculate ephemerides for the objects in the FOVs.

        add_obsid : bool, optional
            If `True`, the ephemerides will have a column "obsid" with the
            observer designations. This may increase memory usage
            significantly, especially if there are many `str` obsids for
            millions of rows. Default is `True`.

        drop_obsindex : bool, optional
            If `True`, the "obsindex" column will be dropped from the
            ephemerides. This is useful especially if `add_obsid` is `True`.
            Default is `True`.

        """
        if self.fov_check_simstates:
            eph, obsindex = calc_ephems(
                self.orb,
                self.fov_check_simstates,
                gpar_default=gpar_default,
                sort_by=sort_by,
                add_jds=add_jds,
                dtypes=dtypes,
                output=output,
                overwrite=overwrite,
            )
            if add_obsid:
                # Add the observer designations to the ephemerides
                eph["obsid"] = eph["obsindex"].apply(lambda x: obsindex[x])
            if drop_obsindex:
                eph.drop(columns=["obsindex"], inplace=True)

            self.eph = eph
            self.eph_obsindex = obsindex

        else:
            Warning("No valid FOV states for ephemeris calculation.")
            self.eph = None
            self.eph_obsindex = None

    @cached_property
    def eph_obsids(self):
        """Get the observer designations for the ephemerides.

        Useful as `self.eph["obsid"] = self.eph_obsids`
        """
        try:
            return self.eph["obsindex"].apply(lambda x: self.eph_obsindex[x])
        except KeyError:
            # if obsindex is dropped in calc_ephems()
            return np.array(self.eph["obsid"])

    def world2pix():
        pass


def _calc_ephem(
    orb, simulstates, gpar_default=0.15, rates_in_arcsec_per_min=True, sort_by=None
):
    """Calculate the ephemerides for the objects in the FOVs.
    Parameters
    ----------
    orb : `~pandas.DataFrame`
        Orbit data with columns of orbital elements.

    simulstates : `~kete.SimultaneousStates`
        Simultaneous states of the objects in the FOV.

    gpar_default : float, optional
        Default slope parameter (G in the IAU H, G model) for the objects.
        Default is 0.15.

    rates_in_arcsec_per_min : bool, optional
        If `True`, the rates will be in arcsec/min instead of degrees/day.
        Default is `True`.

    sort_by : list, optional
        List of columns to sort the output DataFrame by. Default is ["vmag"].

    """
    # NOTE: This is generally a very fast function compared to other SSO
    #   related calculations. I did not put much effort into optimizing it.
    geoms = calc_geometries(
        simulstates, rates_in_arcsec_per_min=rates_in_arcsec_per_min
    )

    # orb = orb.loc[orb["desig"].isin(geoms["desig"])]
    inmask = orb["desig"].isin(geoms["desig"])

    if not any(inmask):
        raise ValueError("No matching objects between `orb` and `simulstates`: "
                         + f"{orb['desig'].tolist()} vs {geoms['desig'].tolist()}")

    orb["G"] = orb["G"].fillna(gpar_default)
    _orb = orb[["H", "G", "M1", "M2", "K1", "K2", "PC"]].to_numpy()[inmask, :]

    vmags = iau_hg_mag(
        _orb[:, 0],
        geoms["alpha"],
        gpar=_orb[:, 1],
        robs=geoms["r_obs"],
        rhel=geoms["r_hel"],
    )
    tmags, nmags = comet_mag(
        m1=_orb[:, 2],
        m2=_orb[:, 3],
        k1=_orb[:, 4],
        k2=_orb[:, 5],
        pc=_orb[:, 6],
        alpha__deg=geoms["alpha"],
        robs=geoms["r_obs"],
        rhel=geoms["r_hel"],
    )
    # either vmag or tmag/nmag, whichever is the brightest:
    vmag = np.nanmin([vmags, tmags, nmags], axis=0)
    vmag[np.isnan(vmag)] = 99.0  # fill with a large value
    geoms["vmag"] = vmag  # fill with a large value

    geoms = pd.DataFrame.from_dict(geoms)

    if sort_by is not None:
        geoms = geoms.sort_values(sort_by).reset_index(drop=True)

    return geoms, orb


def calc_ephems(
    orb,
    simulstates,
    gpar_default=0.15,
    rates_in_arcsec_per_min=True,
    add_jds=False,
    sort_by=["vmag"],
    dtypes=_EPH_DTYPES,
    output="eph.parq",
    overwrite=False,
    **kwargs,
):
    """Calculate ephemerides for the objects in the FOV.
    Run _calc_ephem for multiple SimultaneousStates

    Parameters
    ----------
    orb : `~pandas.DataFrame`
        Orbit data with columns of orbital elements.

    simulstates : iterable of `~kete.SimultaneousStates`
        Simultaneous states of the objects in the FOV. It is recommended
        each element has `.fov.observer.desig`.

    gpar_default : float, optional
        Default slope parameter (G in the IAU H, G model) for the objects.
        Default is 0.15.

    rates_in_arcsec_per_min : bool, optional
        If `True`, the rates will be in arcsec/min instead of degrees/day.
        Default is `True`.

    add_jds : bool, optional
        If `True`, add two columns ``"jd_tdb"``, ``"jd_utc"`` for the Julian date (TDB) and (UTC) to the output DataFrame. Default is `False`.
        Many times it is redundant to calculate both, but because general
        users prefer UTC while all N-body calculations are done in TDB,
        generating both at the same time makes code easier. Depending on
        the disk/ram usage, the user can drop the unnecessary column.

    sort_by : list, optional
        List of columns to sort the output DataFrame by. Default is ["vmag"].

    dtypes : dict, optional
        Data types for the output DataFrame. Default is `_EPH_DTYPES`.

    output : str or `~pathlib.Path`, optional
        Path to the output file where the ephemerides will be saved in
        Parquet format. Default is "eph.parq". If `None`, no output file is
        saved.

    overwrite : bool, optional
        If `True`, overwrite the output file if it already exists.
        Default is `False`.

    **kwargs : dict, optional
        Additional keyword arguments to pass to `pandas.DataFrame.to_parquet()`.

    Returns
    -------
    dfs : `~pandas.DataFrame`
        DataFrame containing the ephemerides of the objects in the FOV.

    obsids : list of str
        List of observer designations for "obsindex" column in `dfs`.
        The true designation can be found as `dfs["obsindex"].apply(lambda x:
        obsids[x])`. But doing this *might* increase the memory usage **a lot**
        because of long strings in the `obsids` list.

    Notes
    -----
    Included columns are::
        - alpha :
            The phase angles in degrees.
        - r_hel, r_obs :
            The heliocentric and observer-centric distances in AU.
        - ra, dec :
            The observer-centric (J2000) right ascension and declination in degrees.
        - dra*cosdec/dt, ddec/dt :
            The observer-centric (J2000) right ascension and declination rates
            in degrees per day (or arcsec/min if `rates_in_arcsec_per_min` is
            `True`).
        - hel_ecl_lons, hel_ecl_lats :
            The heliocentric ecliptic longitudes and latitudes in degrees.
        - obs_ecl_lons, obs_ecl_lats :
            The observer-centric ecliptic longitudes and latitudes in degrees
            (e.g., the observer is the Earth).
        - sky_motion, sky_motion_pa :
            The apparent angular rate of the target in the plane-of-sky in
            degrees/day (or arcsec/min if `rates_in_arcsec_per_min` is `True`),
            and the position angle in degrees. PA is positive if RA rate is
            positive (i.e., measured counter-clockwise from the apparent
            of-date north pole direction, which is identical to JPL Horizons)
        - vmag :
            The apparent magnitude of the object in V-band based on IAU H, G
            magnitude system *or* comet N-/T-mag system from Horizons. `nanmin`
            of those values will be used (i.e., T-mag for comets whenever
            possible).
        - obsindex :
            The index of the observer of `simulstates`, starting from 0.
        - jd_tdb, jd_utc :
            The Julian date (TDB) and (UTC) in days. (only if `add_jds` is `True`)
    """
    dfs = []
    try:
        _orb = orb[["desig", "H", "G", "M1", "M2", "K1", "K2", "PC"]].copy()
    except KeyError:
        # Some columns are missing, add nan:
        _orb = orb.copy()
        for col in ["H", "G", "M1", "M2", "K1", "K2", "PC"]:
            if col not in _orb.columns:
                _orb[col] = np.nan
        _orb = _orb[["desig", "H", "G", "M1", "M2", "K1", "K2", "PC"]]

    obsids = []

    if isinstance(simulstates, kete.SimultaneousStates):
        # If a single SimultaneousStates is provided, convert it to a list
        simulstates = [simulstates]

    for idx, _simulstates in enumerate(list(simulstates)):
        eph, _ = _calc_ephem(
            _orb,
            _simulstates,
            gpar_default=gpar_default,
            rates_in_arcsec_per_min=rates_in_arcsec_per_min,
            sort_by=None,
        )
        eph["obsindex"] = idx
        if add_jds:
            eph["jd_tdb"] = _simulstates.jd
        obsids.append(_simulstates.fov.observer.desig)
        dfs.append(eph)

    dfs = pd.concat(dfs, ignore_index=True)

    if add_jds:
        dfs["jd_utc"] = dfs["jd_tdb"].apply(lambda x: tdb2utc(x).jd)

    if sort_by is not None:
        dfs = dfs.sort_values(["obsindex"] + sort_by).reset_index(drop=True)

    if dtypes is not None:
        for c, d in dtypes.items():
            dfs[c] = dfs[c].astype(d)

    if output is not None:
        output = Path(output)
        if overwrite or not output.exists():
            dfs.to_parquet(output, **kwargs)

    # return dfs
    return dfs, obsids
