from pathlib import Path

import numpy as np
from .ssoflux import iau_hg_mag, comet_mag
from .keteutils.propagate import calc_geometries
import pandas as pd

__all__ = ["calc_ephems"]


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

    orb = orb.loc[orb["desig"].isin(geoms["desig"])].reset_index()

    orb["G"] = orb["G"].fillna(gpar_default)
    _orb = orb[["H", "G", "M1", "M2", "K1", "K2", "PC"]].to_numpy()

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
    """
    dfs = []
    obsids = []
    _orb = orb[["desig", "H", "G", "M1", "M2", "K1", "K2", "PC"]].copy()
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

    return dfs, obsids
