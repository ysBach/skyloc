from functools import cached_property
from pathlib import Path

import kete
import numpy as np
import pandas as pd

from .configs import MINIMUM_ORB_COLS
from .keteutils.fov import FOVCollection
from .keteutils.propagate import calc_geometries, make_nongravs_models
from .keteutils.spice import KETE_LOADED_ASTEROIDS
from .ssoflux import comet_mag, iau_hg_mag
from .utils import listmask

__all__ = ["SSOLocator", "SpiceLocator", "calc_ephems"]

_EPH_DTYPES = {
    "alpha": np.float32,
    "vmag": np.float16,  # ~ +/- 0.01 mag error expected
    "r_obs": np.float32,
    "r_hel": np.float32,
    "dra*cosdec/dt": np.float32,
    "ddec/dt": np.float32,
    "sky_motion": np.float32,
    "sky_motion_pa": np.float32,
    "obsindex": np.uint32,
}


class Locator:
    """A base class for locators."""

    def __init__(self, fovs):
        self._fovc = fovs

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
    """A class to locate objects in a given Field of View (FOV) using SPICE."""

    def __init__(self, fovs, desigs):
        super().__init__(fovs)
        self.desigs = desigs
        self.loaded = kete.spice.loaded_objects()

    def load_spice(self, fpaths):
        kete.spice.load_spice(fpaths)
        self.loaded = kete.spice.loaded_objects()


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
        self, fovs, orb, non_gravs=True, copy_orb=False, drop_major_asteroids=True
    ):
        """
        Parameters
        ----------
        fovs : FOVCollection, iterable of `kete.fov.FOV`, or `kete.fov.FOVList`
            Field of View(s) to check the propagated states against.

        orb : `pandas.DataFrame`
            Orbit file with columns of orbital elements. See
            `query.fetch_orb()`.

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

        if output_exists and not overwrite:
            return kete.SimultaneousStates.load_parquet(output)

        if not isinstance(jd0, (float, int)):
            # Use the mean JD of the FOVs
            jd0 = jd0(self.fovc.fov_jds)

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
            states0 = kete.SimultaneousStates.load_parquet(output)

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
        output=None,
        overwrite=False,
    ):
        """Calculate ephemerides for the objects in the FOVs."""
        eph, obsindex = calc_ephems(
            self.orb,
            self.fov_check_simstates,
            gpar_default=gpar_default,
            sort_by=sort_by,
            dtypes=dtypes,
            output=output,
            overwrite=overwrite,
        )
        self.eph = eph
        self.eph_obsindex = obsindex

    @cached_property
    def eph_obsids(self):
        """Get the observer designations for the ephemerides.

        Useful as `self.eph["obsid"] = self.eph_obsids`
        """
        return self.eph["obsindex"].apply(lambda x: self.eph_obsindex[x])

    def world2pix():
        pass


def _calc_ephem(orb, simulstates, gpar_default=0.15, sort_by=["vmag"]):
    """
    Parameters
    ----------
    orb : `~pandas.DataFrame`
        Orbit data with columns of orbital elements.

    simulstates : `~kete.SimultaneousStates`
        Simultaneous states of the objects in the FOV.

    gpar_default : float, optional
        Default slope parameter (G in the IAU H, G model) for the objects.
        Default is 0.15.

    sort_by : list, optional
        List of columns to sort the output DataFrame by. Default is ["vmag"].

    """
    # NOTE: This is generally a very fast function compared to other SSO
    #   related calculations. I did not put much effort into optimizing it.
    geoms = calc_geometries(simulstates)

    # orb = orb.loc[orb["desig"].isin(geoms["desig"])]
    inmask = orb["desig"].isin(geoms["desig"])

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
    """
    dfs = []
    _orb = orb[["desig", "H", "G", "M1", "M2", "K1", "K2", "PC"]].copy()
    obsids = []
    for idx, _simulstates in enumerate(list(simulstates)):
        eph, _ = _calc_ephem(
            _orb, _simulstates, gpar_default=gpar_default, sort_by=None
        )
        eph["obsindex"] = idx
        obsids.append(_simulstates.fov.observer.desig)
        dfs.append(eph)
    dfs = pd.concat(dfs, ignore_index=True)

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
