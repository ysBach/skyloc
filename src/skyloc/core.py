from functools import cached_property
from pathlib import Path
import logging

from .keteutils._kete_import import kete, require_kete
import numpy as np
import pandas as pd

from .configs import MINIMUM_ORB_COLS
from .keteutils.fov import FOVCollection
from .keteutils.propagate import (
    calc_geometries,
    make_nongravs_models,
    replace_loaded_with_spice,
)
from .keteutils._util import get_kete_loaded_objects
from .keteutils import (
    get_default_spice_targets,
    map_spice_name_to_desig,
    KETE_ASTEROIDS_PHYSICS,
)
from .ioutils.ephemeris import compact_ephem_parq_cols, EPH_DTYPES_BASE
from .ssoflux import comet_mag, iau_hg_mag
from .utils import listmask, tdb2utc

logger = logging.getLogger(__name__)

__all__ = ["locator_twice", "SSOLocator", "SpiceLocator", "StarLocator", "calc_ephems"]


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
    non_gravs=[True, True],
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

    drop_major_asteroids : bool, optional
        If `True`, drop major asteroids (those loaded in kete) from the
        `orb`. Default is `True`.

    jd0 : tuple of callable or float, optional
        A 2-tuple of JDs (or callables) for the first and second propagation.
        Default is ``(np.mean, np.mean)``.

    include_asteroids : tuple of bool, optional
        A 2-tuple indicating whether to include asteroids in the first and
        second propagation. Default is ``(False, True)``.

    dt_limit : tuple of float, optional
        A 2-tuple of time limits (days) for 2-body approximation in the first
        and second check. Default is ``(5.0, 0.5)``.

    suppress_errors : bool, optional
        If `True`, suppress errors during propagation. Default is `True`.

    add_obsid : bool, optional
        If `True`, add "obsid" column to the result. Default is `True`.

    drop_obsindex : bool, optional
        If `True`, drop "obsindex" column from the result. Default is `True`.

    add_jds : bool, optional
        If `True`, add "jd_tdb" and "jd_utc" columns to the result.
        Default is `True`.

    non_gravs : list of bool, optional
        A 2-element list indicating whether to use non-gravitational terms
        for the first and second propagation. Default is ``[True, True]``.

    calc_ephems_crude : bool, optional
        If `True`, calculate ephemerides after the first crude step.
        Default is `False`.

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

    """
    if simstates is not None:
        if not isinstance(simstates, kete.SimultaneousStates):
            simstates = kete.SimultaneousStates.load_parquet(simstates)

    sl1 = SSOLocator(
        fovs=fovs,
        orb=orb,
        simstates=simstates,
        non_gravs=non_gravs[0],
        drop_major_asteroids=drop_major_asteroids,
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
        non_gravs=non_gravs[1],
        drop_major_asteroids=drop_major_asteroids,
    )
    sl2.propagate_n_body(
        jd0=jd0[1],
        suppress_errors=suppress_errors,
        include_asteroids=include_asteroids[1],
    )
    sl2.fov_state_check(dt_limit=dt_limit[1], include_asteroids=include_asteroids[1])
    sl2.calc_ephems(add_obsid=add_obsid, drop_obsindex=drop_obsindex, add_jds=add_jds)

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

    This locator uses SPICE kernels to compute ephemerides for planets,
    satellites, and large asteroids. It bypasses n-body propagation which
    can produce NaN for objects that are treated as perturbers (like the
    5 large asteroids in kete).

    Parameters
    ----------
    fovs : FOVCollection, iterable of `kete.FOV`, or `kete.FOVList`
        Field of View(s) to check against.

    targets : list of str, optional
        List of target names to locate. These must be SPICE-resolvable names
        (planets, satellites, or large asteroids loaded in kete).
        If `None`, uses default targets from `get_default_spice_targets()`.
        Default is `None`.

    include_planets : bool, optional
        If `True` and targets is `None`, include default planets.
        Default is `True`.

    include_asteroids : bool, optional
        If `True` and targets is `None`, include kete's 5 large asteroids.
        Default is `True`.

    Attributes
    ----------
    targets : list of str
        The target names being located.

    fov_check_simstates : list of `kete.SimultaneousStates`
        Results from fov_spice_check after calling `fov_spice_check()`.

    fov_check_fov2objs : dict
        Mapping of FOV designations to lists of visible object names.

    fov_check_objids : list of str
        Unique object names visible in at least one FOV.
    """

    def __init__(
        self,
        fovs,
        targets=None,
        include_planets=True,
        include_asteroids=True,
    ):
        super().__init__(fovs)

        if targets is None:
            self.targets = get_default_spice_targets(
                include_planets=include_planets,
                include_asteroids=include_asteroids,
            )
        else:
            self.targets = list(targets)

        self.loaded = kete.spice.loaded_objects()

        # Results storage (populated by fov_spice_check)
        self.fov_check_simstates = None
        self.fov_check_fov2objs = None
        self.fov_check_objids = None

    def load_spice(self, fpaths):
        """Load additional SPICE kernels.

        Parameters
        ----------
        fpaths : str or list of str
            Path(s) to SPICE kernel files to load.
        """
        kete.spice.load_spice(fpaths)
        self.loaded = kete.spice.loaded_objects()

    def fov_spice_check(self):
        """Check which SPICE targets are visible in the FOVs.

        Uses `kete.fov.fov_spice_check()` to determine visibility of
        planets, satellites, and large asteroids.

        Results are stored in:
        - `self.fov_check_simstates`: List of SimultaneousStates
        - `self.fov_check_fov2objs`: Dict mapping FOV desig -> list of visible objects
        - `self.fov_check_objids`: List of unique visible object names
        """
        simstas = []
        for target in self.targets:
            try:
                # make target a list if not iterable
                if not isinstance(target, (list, tuple)):
                    target = [target]
                result = kete.fov.fov_spice_check(target, self.fovc.fovlist)
                simstas.extend(result)
            except Exception as e:
                logger.warning("fov_spice_check failed for %s: %s", target, e)
                continue

        # Collect designations of FOVs and objects
        fov2objs = {}
        for _ss in simstas:
            _objids = [_s.desig for _s in _ss.states]
            fov2objs.setdefault(_ss.fov.observer.desig, []).extend(_objids)

        # Get unique object IDs
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

        logger.info(
            "SPICE FOV check: %d objects in %d FOVs", len(uniq_objids), len(fov2objs)
        )

    def calc_ephems(
        self,
        sort_by=["vmag"],
        compact=True,
        add_obsid=True,
        drop_obsindex=True,
        add_jds=True,
        output=None,
        overwrite=False,
    ):
        """Calculate ephemerides for SPICE objects in the FOVs.

        Parameters
        ----------
        sort_by : list, optional
            Columns to sort by. Default is ``["vmag"]``.

        compact : bool, optional
            If `True`, apply advanced compression using
            `~skyloc.ioutils.ephemeris.compact_ephem_parq_cols` before saving.
            The saved file can be loaded with
            `~skyloc.ioutils.ephemeris.load_compact_parq_ephem`.
            Default is `True`.

        add_obsid : bool, optional
            If `True`, add "obsid" column. Default is `True`.

        drop_obsindex : bool, optional
            If `True`, drop "obsindex" column. Default is `True`.

        add_jds : bool, optional
            If `True`, add "jd_tdb" and "jd_utc" columns. Default is `True`.

        output : str or Path, optional
            Output parquet file path. Default is `None`.

        overwrite : bool, optional
            If `True`, overwrite existing output file. Default is `False`.
        """
        if not self.fov_check_simstates:
            logger.warning("No valid SPICE FOV states for ephemeris calculation.")
            self.eph = None
            self.eph_obsindex = None
            return

        # Build ephemerides from SimultaneousStates
        dfs = []
        obsids = []

        for idx, _simulstates in enumerate(self.fov_check_simstates):
            geoms = calc_geometries(_simulstates, rates_in_arcsec_per_min=True)

            # For SPICE objects, we don't have H/G magnitudes, use placeholder
            geoms["vmag"] = np.full(len(geoms["desig"]), 99.0, dtype=np.float32)

            # Add magnitude estimates for large asteroids if available
            for i, desig in enumerate(geoms["desig"]):
                for num, info in KETE_ASTEROIDS_PHYSICS.items():
                    if desig.lower() == info["name"]:
                        h_mag = info["H"]
                        g_par = info["G"]
                        vmag = iau_hg_mag(
                            h_mag,
                            geoms["alpha"][i],
                            gpar=g_par,
                            robs=geoms["r_obs"][i],
                            rhel=geoms["r_hel"][i],
                        )
                        geoms["vmag"][i] = vmag
                        break

            eph = pd.DataFrame.from_dict(geoms)
            eph["obsindex"] = idx
            if add_jds:
                eph["jd_tdb"] = _simulstates.jd
            obsids.append(_simulstates.fov.observer.desig)
            dfs.append(eph)

        if not dfs:
            self.eph = None
            self.eph_obsindex = None
            return

        dfs = pd.concat(dfs, ignore_index=True)

        if add_jds:
            dfs["jd_utc"] = tdb2utc(dfs["jd_tdb"].values, format="jd").jd

        if sort_by is not None:
            dfs = dfs.sort_values(["obsindex"] + sort_by).reset_index(drop=True)

        if add_obsid:
            obsindex_arr = np.array(obsids)
            dfs["obsid"] = obsindex_arr[dfs["obsindex"].values]

        if drop_obsindex:
            dfs.drop(columns=["obsindex"], inplace=True)

        if output is not None:
            output = Path(output)
            if overwrite or not output.exists():
                if compact:
                    dfs_to_save = compact_ephem_parq_cols(dfs, dtypes=EPH_DTYPES_BASE)
                else:
                    dfs_to_save = dfs
                dfs_to_save.to_parquet(output)
                logger.debug("Saved SPICE ephemeris to %s", output)

        self.eph = dfs
        self.eph_obsindex = obsids


class StarLocator(Locator):
    """A class to locate sidereal objects (fixed RA/Dec) in a given Field of View.

    This class uses kete's fov_static_check to determine visibility of objects
    with fixed celestial coordinates (stars, galaxies, QSOs, etc.).

    Parameters
    ----------
    fovs : FOVCollection, iterable of `kete.FOV`, or `kete.FOVList`
        Field of View(s) to check against.

    sources : pandas.DataFrame, optional
        DataFrame with columns "ra", "dec" (in degrees, J2000) and optionally
        "desig" for source designations. Other columns (e.g., "vmag") are preserved.
        If `None`, sources must be set later via `set_sources()`.
        Default is `None`.

    Attributes
    ----------
    sources : pandas.DataFrame
        The source catalog being checked.

    fov_check_results : list
        Results from fov_static_check after calling `fov_static_check()`.

    fov_check_fov2objs : dict
        Mapping of FOV designations to lists of visible source indices.

    Notes
    -----
    Future enhancement: Add proper motion support for epoch propagation.
    """

    def __init__(self, fovs, sources=None):
        super().__init__(fovs)
        self._sources = None
        if sources is not None:
            self.set_sources(sources)

        # Results storage
        self.fov_check_results = None
        self.fov_check_fov2objs = None
        self.sources_infov_mask = None

    def set_sources(self, sources):
        """Set the source catalog.

        Parameters
        ----------
        sources : pandas.DataFrame
            DataFrame with columns "ra", "dec" (degrees, J2000).
            Optional: "desig" for source designations.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        if not isinstance(sources, pd.DataFrame):
            raise TypeError("sources must be a pandas DataFrame")

        required = ["ra", "dec"]
        missing = [c for c in required if c not in sources.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if "desig" not in sources.columns:
            # Generate default designations
            sources = sources.copy()
            sources["desig"] = [f"star_{i}" for i in range(len(sources))]

        self._sources = sources

    @property
    def sources(self):
        """Get the source catalog."""
        return self._sources

    def fov_static_check(self):
        """Check which static sources are visible in the FOVs.

        Uses `kete.fov.fov_static_check()` to determine visibility of
        fixed RA/Dec sources. Sources are converted to kete Vector objects
        and checked against each FOV.

        Results are stored in:
        - `self.fov_check_results`: Raw results from kete
        - `self.fov_check_fov2objs`: Dict mapping FOV desig -> list of source desigs
        - `self.sources_infov_mask`: Boolean mask of sources visible in any FOV
        """
        if self._sources is None:
            raise ValueError("No sources set. Use set_sources() first.")

        # Convert RA/Dec to kete Vectors
        ra_arr = self._sources["ra"].values
        dec_arr = self._sources["dec"].values
        desigs = self._sources["desig"].values

        # Create direction vectors for each source
        source_vectors = []
        for ra, dec in zip(ra_arr, dec_arr):
            vec = kete.Vector.from_ra_dec(ra=ra, dec=dec)
            source_vectors.append(vec)

        # Run static check for each FOV
        fov2objs = {}
        all_visible_indices = set()

        for fov in self.fovc.fovlist:
            visible_indices = []
            for i, vec in enumerate(source_vectors):
                # Check if vector is within FOV
                # fov_static_check takes a list of vectors and returns those in FOV
                try:
                    result = kete.fov.fov_static_check([vec], [fov])
                    if result:
                        visible_indices.append(i)
                        all_visible_indices.add(i)
                except Exception as e:
                    logger.debug("fov_static_check failed for source %d: %s", i, e)
                    continue

            if visible_indices:
                visible_desigs = [desigs[i] for i in visible_indices]
                fov2objs[fov.observer.desig] = visible_desigs

        # Create mask for sources in any FOV
        self.sources_infov_mask = np.zeros(len(self._sources), dtype=bool)
        self.sources_infov_mask[list(all_visible_indices)] = True

        self.fov_check_fov2objs = fov2objs
        self.fov_check_results = fov2objs  # For compatibility

        # FOVs with objects
        self.fov_mask_hasobj = self.fovc.mask_by_desig(fov2objs.keys())
        if np.any(self.fov_mask_hasobj):
            self.fovc_hasobj = FOVCollection(self.fovc[self.fov_mask_hasobj])
        else:
            self.fovc_hasobj = None

        logger.info(
            "Static FOV check: %d sources in %d FOVs",
            len(all_visible_indices),
            len(fov2objs),
        )

    def get_visible_sources(self):
        """Get DataFrame of sources visible in at least one FOV.

        Returns
        -------
        pandas.DataFrame
            Subset of sources that are visible in any FOV.
        """
        if self.sources_infov_mask is None:
            raise ValueError("Run fov_static_check() first.")
        return self._sources[self.sources_infov_mask].copy()


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
        self,
        fovs,
        orb,
        simstates=None,
        non_gravs=True,
        copy_orb=False,
        drop_major_asteroids=True,
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
            Default is `None`.

        non_gravs : list, bool, optional
            A list of non-gravitational terms for each object. If provided, then
            every object must have an associated `kete.NonGravModel` or `None`.
            If `True`, `keteutils.make_nongravs_models` will be used, i.e.,
            objects with non-gravitational terms will have non-gravitational
            models by `kete.NonGravModel.new_comet()`.
            If `False`, no non-gravitational terms will be used for any object.
            Default is `True`.

        copy_orb : bool, optional
            If `True`, copy the `orb` DataFrame. Default is `False`.

        drop_major_asteroids : bool, optional
            If `True`, drop major asteroids (those loaded in kete) from the
            `orb`.
            Default is `True`.
        """
        super().__init__(fovs)
        if copy_orb:
            orb = orb.copy()
        self._validate_orb(orb)
        if drop_major_asteroids:
            cache = get_kete_loaded_objects()
            orb = orb[~orb.desig.isin(cache["asteroids"])]
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
            i.e., same as `self.orb`. Default is `None`.

        output : str or `~pathlib.Path`, optional
            If provided, save the propagated states to a parquet file.
            Default is `None`.

        overwrite : bool, optional
            If `True`, re-do the calculation and overwrite the existing output file
            if it exists. If `False`, the `output` will be loaded if it exists.
            Default is `False`.

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

        # Determine whether to compute fresh or load from cache
        load_from_cache = output_exists and not overwrite

        if load_from_cache:
            states0 = kete.SimultaneousStates.load_parquet(output).states
            logger.debug("Loaded cached states from %s", output)
        else:
            states0 = kete.propagate_n_body(
                listmask(self.states_from_orb, objmask),
                jd=jd0,
                include_asteroids=include_asteroids,
                non_gravs=listmask(self.non_gravs, objmask),
                suppress_errors=suppress_errors,
            )
            logger.info("Propagated %d states to JD %.2f", len(states0), jd0)

        # Replace loaded asteroids with SPICE states to avoid self-impact issues
        # This runs ALWAYS (whether loaded from cache or freshly computed)
        if include_asteroids:
            states0 = replace_loaded_with_spice(states0, jd0)

        # Save to output file if requested (and not loaded from cache)
        if output is not None and not load_from_cache:
            kete.SimultaneousStates(states0).save_parquet(output)
            # Reload to ensure consistency
            states0 = kete.SimultaneousStates.load_parquet(output).states
            logger.debug("Saved propagated states to %s", output)

        self.jd0 = jd0
        self.states_propagated_jd0 = states0

    def fov_state_check(
        self,
        dt_limit=3.0,
        include_asteroids=False,
        use_spice_for_loaded=True,
        spice_targets=None,
    ):
        """Check which objects are in the FOVs after propagation.

        Parameters
        ----------
        dt_limit : float, optional
            Length of time in days where 2-body mechanics is a good approximation.
            Default is ``3.0`` days.

        include_asteroids : bool, optional
            If `True`, include asteroids in the check. Default is `False`.

        use_spice_for_loaded : bool, optional
            If `True`, use SPICE kernels (via `fov_spice_check`) for planets
            and the 5 large asteroids that kete uses as perturbers. This avoids
            NaN results from n-body propagation for these objects.
            Default is `True`.

        spice_targets : list of str, optional
            Additional SPICE targets to check beyond the default planets and
            large asteroids. These will be checked via `fov_spice_check`.
            Default is `None`.

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

        self.fov_check_spice_simstates : list of `kete.SimultaneousStates`
            SPICE-derived states (only populated if `use_spice_for_loaded=True`).
        """
        simstas = kete.fov.fov_state_check(
            self.states_propagated_jd0,
            self.fovc.fovlist,
            dt_limit=dt_limit,
            include_asteroids=include_asteroids,
        )

        # Handle SPICE-based objects (planets only)
        # Note: Loaded asteroids are already SPICE-replaced in propagate_n_body,
        # so we don't need fov_spice_check for them (which causes fuzzy matching issues).
        spice_simstas = []
        if use_spice_for_loaded:
            # Only check planets - asteroids already handled via n-body replacement
            planet_targets = get_default_spice_targets(
                include_planets=True, include_asteroids=False
            )
            targets_to_check = list(planet_targets)

            # Add any additional user-specified targets
            if spice_targets is not None:
                targets_to_check.extend(spice_targets)

            # Run fov_spice_check for each target (planets use SPICE IDs, which work)
            for target in targets_to_check:
                try:
                    result = kete.fov.fov_spice_check(target, self.fovc.fovlist)
                    spice_simstas.extend(result)
                except Exception as e:
                    logger.debug("fov_spice_check failed for %s: %s", target, e)
                    continue

            if spice_simstas:
                logger.info(
                    "SPICE check found %d SimultaneousStates for planets",
                    len(spice_simstas),
                )

        # Merge n-body and SPICE results
        all_simstas = list(simstas) + spice_simstas

        # Convenience: Collect the designations of the FOVs and objects
        fov2objs = {}
        # key = FOV's designations (str)
        # value = list of object designations in that FOV (list[str])
        for _ss in all_simstas:
            # Use list comprehension instead of inner loop with append
            _objids = [map_spice_name_to_desig(_s.desig) for _s in _ss.states]
            fov2objs.setdefault(_ss.fov.observer.desig, []).extend(_objids)

        # For ~1000 elements, below is ~ 10x faster than
        #    np.unique(np.concatenate(list(fov2objs.values())))
        uniq_objids = set()
        for objids in fov2objs.values():
            uniq_objids.update(objids)
        uniq_objids = list(uniq_objids)

        self.fov_check_simstates = all_simstas
        self.fov_check_spice_simstates = spice_simstas  # Store separately for reference
        self.fov_check_fov2objs = fov2objs
        self.fov_check_objids = uniq_objids

        # FOVs with objects
        self.fov_mask_hasobj = self.fovc.mask_by_desig(fov2objs.keys())
        self.fovc_hasobj = FOVCollection(self.fovc[self.fov_mask_hasobj])

        # objects in at least one of FOV
        self.orb_infov_mask = self.orb["desig"].isin(uniq_objids)

        logger.info("FOV check: %d objects in %d FOVs", len(uniq_objids), len(fov2objs))

    def calc_ephems(
        self,
        gpar_default=0.15,
        sort_by=["vmag"],
        compact=True,
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
                compact=compact,
                output=output,
                overwrite=overwrite,
            )
            if add_obsid:
                # Add the observer designations to the ephemerides
                # Use NumPy array indexing instead of slow .apply()
                obsindex_arr = np.array(obsindex)
                eph["obsid"] = obsindex_arr[eph["obsindex"].values]
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
            # Use NumPy array indexing instead of slow .apply()
            obsindex_arr = np.array(self.eph_obsindex)
            return obsindex_arr[self.eph["obsindex"].values]
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
        Default is ``0.15``.

    rates_in_arcsec_per_min : bool, optional
        If `True`, the rates will be in arcsec/min instead of degrees/day.
        Default is `True`.

    sort_by : list, optional
        List of columns to sort the output DataFrame by. Default is ``["vmag"]``.

    """
    # NOTE: This is generally a very fast function compared to other SSO
    #   related calculations. I did not put much effort into optimizing it.
    geoms = calc_geometries(
        simulstates, rates_in_arcsec_per_min=rates_in_arcsec_per_min
    )
    # Ensure designations match orb (e.g., "vesta" -> "4")
    geoms["desig"] = [map_spice_name_to_desig(d) for d in geoms["desig"]]

    # Align orb to geoms to ensure 1-to-1 correspondence for magnitude calculation
    # Only keep orbital elements for objects present in geoms, in the order of geoms.
    # Note: orb["desig"] must be unique for correct reindexing.
    if orb["desig"].duplicated().any():
        logger.warning(
            "Duplicate designations found in orbit file. Magnitude calculation may be incorrect."
        )
        orb = orb.drop_duplicates(subset="desig")

    # Check for matches
    available_desigs = set(orb["desig"])
    has_orb_mask = np.array([d in available_desigs for d in geoms["desig"]])

    unmatched = [d for d in geoms["desig"] if d not in available_desigs]
    if unmatched:
        logger.debug(
            "Objects without orbit data (SPICE-only, will have NaN vmag): %s...",
            unmatched[:5],
        )

    # Initialize vmag array with NaN for all objects
    n_objects = len(geoms["desig"])
    vmag = np.full(n_objects, np.nan)

    # Only calculate magnitudes for objects with orbit data
    if any(has_orb_mask):
        # Get indices and designations of objects with orbit data
        orb_desigs = [d for i, d in enumerate(geoms["desig"]) if has_orb_mask[i]]
        orb_indices = np.where(has_orb_mask)[0]

        # Reindex orb to match the subset with orbit data
        orb_indexed = orb.set_index("desig")
        orb_aligned = orb_indexed.reindex(orb_desigs)

        # Extract parameters for magnitude calculation
        g_vals = orb_aligned["G"].fillna(gpar_default).to_numpy()
        h_vals = orb_aligned["H"].to_numpy()
        m1 = orb_aligned.get(
            "M1", pd.Series(np.nan, index=orb_aligned.index)
        ).to_numpy()
        m2 = orb_aligned.get(
            "M2", pd.Series(np.nan, index=orb_aligned.index)
        ).to_numpy()
        k1 = orb_aligned.get(
            "K1", pd.Series(np.nan, index=orb_aligned.index)
        ).to_numpy()
        k2 = orb_aligned.get(
            "K2", pd.Series(np.nan, index=orb_aligned.index)
        ).to_numpy()
        pc = orb_aligned.get(
            "PC", pd.Series(np.nan, index=orb_aligned.index)
        ).to_numpy()

        # Extract geometry for these objects
        alpha_sub = np.array(geoms["alpha"])[orb_indices]
        r_obs_sub = np.array(geoms["r_obs"])[orb_indices]
        r_hel_sub = np.array(geoms["r_hel"])[orb_indices]

        vmags = iau_hg_mag(
            h_vals, alpha_sub, gpar=g_vals, robs=r_obs_sub, rhel=r_hel_sub
        )
        tmags, nmags = comet_mag(
            m1=m1,
            m2=m2,
            k1=k1,
            k2=k2,
            pc=pc,
            alpha__deg=alpha_sub,
            robs=r_obs_sub,
            rhel=r_hel_sub,
        )
        # Either vmag or tmag/nmag, whichever is the brightest
        mag_sub = np.nanmin([vmags, tmags, nmags], axis=0)

        # Place calculated magnitudes into the full array
        vmag[orb_indices] = mag_sub

    # Fill remaining NaN vmag with a large value (99.0 = invisible)
    vmag[np.isnan(vmag)] = 99.0
    geoms["vmag"] = vmag

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
    compact=True,
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
        Default is ``0.15``.

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
        List of columns to sort the output DataFrame by. Default is ``["vmag"]``.

    compact : bool, optional
        If `True`, apply advanced compression to the output DataFrame using
        `~skyloc.ioutils.ephemeris.compact_ephem_parq_cols` (integer scaling +
        HEALPix coordinate encoding) before saving. This significantly reduces
        file size at the cost of precision (typically sub-arcsecond for
        coordinates, <1 mmag for vmag). The saved file can be loaded with
        `~skyloc.ioutils.ephemeris.load_compact_parq_ephem`.
        Default is `True`.

    output : str or `~pathlib.Path`, optional
        Path to the output file where the ephemerides will be saved in
        Parquet format. Default is ``"eph.parq"``. If `None`, no output file is
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
        - racosdec_rate, dec_rate :
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
        # Vectorized TDB to UTC conversion (astropy.time.Time handles arrays)
        dfs["jd_utc"] = tdb2utc(dfs["jd_tdb"].values, format="jd").jd

    if sort_by is not None:
        dfs = dfs.sort_values(["obsindex"] + sort_by).reset_index(drop=True)

    if output is not None:
        output = Path(output)
        if overwrite or not output.exists():
            if compact:
                dfs_to_save = compact_ephem_parq_cols(dfs, dtypes=EPH_DTYPES_BASE)
            else:
                dfs_to_save = dfs
            dfs_to_save.to_parquet(output, **kwargs)
            logger.debug("Saved ephemeris to %s", output)

    return dfs, obsids
