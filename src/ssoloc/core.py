import numpy as np
from .ssoflux import iau_hg_mag, comet_mag
from .keteutils.propagate import calc_geometries
import pandas as pd

__all__ = ["calc_ephems"]


def _calc_ephem(orb, simulstates, gpar_default=0.15, sort_by=["vmag"]):
    """
    Parameters
    ----------
    orb : `~pandas.DataFrame`
        Orbit data with columns of orbital elements. It must be indexed by
        the designations of the objects. (i.e., `orb.set_index("desig")` should
        have been called before this function).

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
    orb_in_fov = orb.loc[geoms["desig"]].reset_index()

    vmags = iau_hg_mag(
        orb_in_fov["H"],
        geoms["alpha"],
        gpar=orb_in_fov["G"].fillna(gpar_default),
        robs=geoms["r_obs"],
        rhel=geoms["r_hel"],
    )
    tmags, nmags = comet_mag(
        m1=orb_in_fov["M1"],
        m2=orb_in_fov["M2"],
        k1=orb_in_fov["K1"],
        k2=orb_in_fov["K2"],
        pc=orb_in_fov["PC"],
        alpha__deg=geoms["alpha"],
        robs=geoms["r_obs"],
        rhel=geoms["r_hel"],
    )
    # either vmag or tmag/nmag, whichever is the brightest:
    geoms["vmag"] = np.nanmin([vmags, tmags, nmags], axis=0)

    geoms = pd.DataFrame.from_dict(geoms)

    if sort_by is not None:
        geoms = geoms.sort_values(sort_by).reset_index(drop=True)

    return geoms, orb_in_fov


def calc_ephems(
    orb, simulstates, gpar_default=0.15, sort_by=["vmag"], output=None, overwrite=False
):
    """Calculate ephemerides for the objects in the FOV.
    Run _calc_ephem for multiple SimultaneousStates"""
    dfs = []
    obsids = []
    # set index for quick "ordered" `.loc` lookup in `_calc_ephem`
    _orb = orb.set_index("desig")
    for idx, simulstates in enumerate(list(simulstates)):
        eph, _ = _calc_ephem(_orb, simulstates, gpar_default=gpar_default, sort_by=None)
        eph["obsindex"] = idx
        obsids.append(simulstates.fov.observer.desig)
        dfs.append(eph)
    dfs = pd.concat(dfs, ignore_index=True)

    if sort_by is not None:
        dfs = dfs.sort_values(["obsindex"] + sort_by).reset_index(drop=True)

    return dfs, obsids
