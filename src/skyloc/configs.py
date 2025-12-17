# Configuration file for SSOLoc

from pathlib import Path
import pandas as pd
from astropy.table import QTable
from astropy import units as u
import numpy as np
from io import StringIO


__all__ = [
    "PKG_PATH",
    "SOLSPECT",
    "solspect_wlcut",
    "MINIMUM_ORB_COLS",
    "HORIZONS_DEPOCHS",
    "KETE_SBDB_MINIMUM_FIELDS",
    "KETE_SBDB2KETECOLS",
    "cols2kete_sbdb",
    "SBDB_FIELDS",
    "SBDB_ALLOWED_SBCLASS",
    "IMPACTED",
]

PKG_PATH = Path(__file__).parent

# **************************************************************************************** #
#                                      Solar Spectrum                                      #
# **************************************************************************************** #
SOLSPECT = QTable.read(
    PKG_PATH / "data" / "astm_e490.csv",
    names=["wl", "flam"],
    units=(u.um, u.W / u.m**2 / u.um),
)


def update_solspects():
    """
    Update the SOLSPECT table to include additional columns.
    """
    global SOLSPECT
    flam = SOLSPECT["flam"].value
    wl = SOLSPECT["wl"].value
    fnu = 3.33564095e5 * flam * (wl * wl)  # Convert to fnu in MJy
    abmag = -2.5 * np.log10(fnu / 0.003631)  # Convert to AB magnitude
    SOLSPECT["fnu"] = fnu * u.MJy
    SOLSPECT["ABmag"] = abmag * u.mag


# Update the default SOLSPECT table with fnu and ABmag columns
update_solspects()


def solspect_wlcut(
    wl_min=0.6, wl_max=5.2, wl_knots=None, add_fnu_ab=True, update_global=True
):
    """Cut the SOLSPECT table to the specified wavelength range.

    Parameters
    ----------
    wl_min, wl_max : float
        Minimum and maximum wavelength in micrometers (um).

    wl_knots : array-like, optional
        If provided, the SOLSPECT table will be linearly interpolated to these
        wavelength knots. Must be in um.

    add_fnu_ab : bool, optional
        If `True`, add the `fnu` and `ABmag` columns to the SOLSPECT table
        after the `wl_knots` interpolation. Ignored if `wl_knots` is `None`.
        Default is `True`.

    update_global : bool, optional
        If `True`, update the global SOLSPECT variable. Default is `True`.
    """
    if update_global:
        global SOLSPECT
    _wl = SOLSPECT["wl"].value
    # update the global SOLSPECT variable
    SOLSPECT = SOLSPECT[(wl_min <= _wl) & (_wl <= wl_max)]

    if wl_knots is not None:
        _wl = SOLSPECT["wl"].value
        SOLSPECT = QTable(
            [
                wl_knots,
                np.interp(wl_knots, _wl, SOLSPECT["flam"].value),
            ],
            names=["wl", "flam"],
            units=(u.um, u.W / u.m**2 / u.um),
        )
        if add_fnu_ab:
            update_solspects()

    return SOLSPECT


# I think at least these are required for kete and skyloc
MINIMUM_ORB_COLS = [
    "desig",
    "ecc",
    "incl",
    "peri_dist",
    "peri_arg",
    "lon_node",
    "peri_time",
    "epoch",
]

# **************************************************************************************** #
#                                       JPL Horizons                                       #
# **************************************************************************************** #
HORIZONS_DEPOCHS = 50

# **************************************************************************************** #
#                                         JPL SBDB                                         #
# **************************************************************************************** #
# The following fields are default for kete, as of 2025 March.
# But... not all of them are actually used in kete (e.g., "name").
KETE_SBDB_MINIMUM_FIELDS = [
    "pdes",
    "name",
    "spkid",
    "orbit_id",
    "rms",
    "H",
    "G",
    "epoch",
    "e",
    "i",
    "q",
    "w",
    "tp",
    "om",
    "A1",
    "A2",
    "A3",
    "DT",
    "M1",
    "M2",
    "K1",
    "K2",
    "PC",
    "rot_per",
    "H_sigma",
]

KETE_SBDB2KETECOLS = {
    "e": "ecc",
    "i": "incl",
    "q": "peri_dist",
    "w": "peri_arg",
    "tp": "peri_time",
    "om": "lon_node",
    "pdes": "desig",
}


def cols2kete_sbdb(orb):
    """Convert SBDB column names to `kete` style column names.

    Parameters
    ----------
    orb : pandas.DataFrame
        The SBDB orbit table.

    Returns
    -------
    orb : pandas.DataFrame
        The SBDB orbit table with `kete` style column names.
        See `KETE_SBDB2KETECOLS` in `skyloc.configs`.

    """
    return orb.rename(columns=KETE_SBDB2KETECOLS)

# KETE_SINGLESBDB2KETECOLS = {
# _PARAM_MAP = {
#     "a1": "a1",
#     "a2": "a2",
#     "a3": "a3",
#     "aln": "alpha",
#     "nm": "m",
#     "r0": "r_0",
#     "nk": "k",
#     "nn": "n",
#     "dt": "dt",
#     "e": "eccentricity",
#     "q": "peri_dist",
#     "tp": "peri_time",
#     "node": "lon_of_ascending",
#     "peri": "peri_arg",
#     "i": "inclination",
# }


# TODO: Eventually some of these may be moved to astroquery.jplsbdb

# ignore: column is extremely unlikely to be used in practice (empty for most
#   objects, or redundant)
# [a/c]only : column is (scientifically) only for asteroids/comets
# simple: column is most likely to be used in practice
#   - exception: diameter/albedo are hard-coded in kete, so to be consistent,
#     they are included...
# lite: a slightly lighter than "all" but heavier than "simple" set.
# dtype: data type to convert to (i: int, f: float, s: str)
# fillna: fillna value when the column is missing or empty - not added yet.

#              i a c s l
#              g o o i i
#              n n n m t
#              o l l p e
#              r y y l
_SBDB_FIELDS = pd.read_csv(
    StringIO(
        """column,ignore,aonly,conly,simple,lite,dtype
spkid         ,0,0,0,1,1,i8
full_name     ,0,0,0,0,1,str
kind          ,0,1,1,1,1,str
pdes          ,0,0,0,1,1,str
name          ,0,0,0,0,1,str
prefix        ,0,0,1,1,1,str
neo           ,0,0,0,0,1,str
pha           ,0,1,0,0,1,str
sats          ,0,0,0,0,1,uint8
H             ,0,1,0,1,1,f8
G             ,0,1,0,1,1,f8
M1            ,0,0,1,1,1,f8
M2            ,0,0,1,1,1,f8
K1            ,0,0,1,1,1,f8
K2            ,0,0,1,1,1,f8
PC            ,0,0,1,1,1,f8
S0            ,1,0,0,0,0,str
S0_sigma      ,1,0,0,0,0,str
diameter      ,0,0,0,1,1,f8
extent        ,1,0,0,0,0,str
albedo        ,0,0,0,1,1,f8
rot_per       ,0,0,0,0,1,f8
pole          ,1,0,0,0,0,str
GM            ,1,0,0,0,0,f8
density       ,1,0,0,0,0,f8
BV            ,1,1,0,0,0,f8
UB            ,1,1,0,0,0,f8
IR            ,1,1,0,0,0,f8
spec_B        ,0,1,0,1,1,str
spec_T        ,0,1,0,1,1,str
H_sigma       ,0,1,0,0,0,f8
diameter_sigma,1,0,0,0,0,f8
orbit_id      ,0,0,0,1,1,str
epoch         ,0,0,0,1,1,f8
epoch_mjd     ,1,0,0,0,0,f8
epoch_cal     ,1,0,0,0,0,str
equinox       ,1,0,0,0,0,str
e             ,0,0,0,1,1,f8
a             ,0,0,0,0,0,f8
q             ,0,0,0,1,1,f8
i             ,0,0,0,1,1,f8
om            ,0,0,0,1,1,f8
w             ,0,0,0,1,1,f8
ma            ,0,0,0,0,0,f8
ad            ,0,0,0,0,0,f8
n             ,1,0,0,0,0,f8
tp            ,0,0,0,1,0,f8
tp_cal        ,1,0,0,0,0,str
per           ,0,0,0,0,0,f8
per_y         ,0,0,0,0,0,f8
moid          ,0,0,0,0,1,f8
moid_ld       ,1,0,0,0,0,f8
moid_jup      ,0,0,0,0,1,f8
t_jup         ,0,0,0,0,1,f8
sigma_e       ,0,0,0,0,1,f8
sigma_a       ,0,0,0,0,1,f8
sigma_q       ,0,0,0,0,1,f8
sigma_i       ,0,0,0,0,1,f8
sigma_om      ,0,0,0,0,1,f8
sigma_w       ,0,0,0,0,1,f8
sigma_ma      ,0,0,0,0,1,f8
sigma_ad      ,0,0,0,0,1,f8
sigma_n       ,1,0,0,0,1,f8
sigma_tp      ,0,0,0,0,1,f8
sigma_per     ,0,0,0,0,1,f8
class         ,0,0,0,1,1,str
source        ,0,0,0,0,0,str
soln_date     ,0,0,0,1,1,str
producer      ,1,0,0,0,0,str
data_arc      ,0,0,0,0,0,uint32
first_obs     ,0,0,0,0,1,str
last_obs      ,0,0,0,0,1,str
n_obs_used    ,0,0,0,0,1,uint32
n_del_obs_used,0,0,0,0,0,uint32
n_dop_obs_used,0,0,0,0,0,uint32
pe_used       ,0,0,0,0,0,str
sb_used       ,0,0,0,0,0,str
condition_code,0,0,0,1,1,str
rms           ,0,0,0,1,1,f8
two_body      ,0,0,0,1,1,str
A1            ,0,0,0,1,1,f8
A1_sigma      ,0,0,0,1,1,f8
A2            ,0,0,0,1,1,f8
A2_sigma      ,0,0,0,1,1,f8
A3            ,0,0,0,1,1,f8
A3_sigma      ,0,0,0,1,1,f8
DT            ,0,0,0,1,1,f8
DT_sigma      ,0,0,0,1,1,f8"""
    ),
    dtype={
        "column": str,
        "ignore": bool,
        "aonly": bool,
        "conly": bool,
        "simple": bool,
        "lite": bool,
        "dtype": str,
    },
)
# numpy's np.dtype(value):
_SBDB_FIELDS["dtype"] = _SBDB_FIELDS["dtype"].map(lambda x: np.dtype(x))
# _SBDB_FIELDS["dtype"] = _SBDB_FIELDS["dtype"].map({"i": int, "f": float, "s": str})
# For details of each column: https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html
# I found saving columns in, e.g., int32, does not really help reducing
# memory/storage usage for parquet.

SBDB_FIELDS = {}
# All fields in the _SBDB_FIELDS:
SBDB_FIELDS["*"] = {
    c.strip(): t for c, t in zip(_SBDB_FIELDS["column"], _SBDB_FIELDS["dtype"])
}
for _name, _query in zip(
    ["all", "ignore", "simple", "simple_ast", "simple_com", "all_ast", "all_com", "lite", "lite_ast", "lite_com"],
    [
        "~ignore",
        "ignore",
        "simple",
        "simple & ~conly",
        "simple & ~aonly",
        "~ignore & ~conly",
        "~ignore & ~aonly",
        "lite",
        "lite & ~conly",
        "lite & ~aonly",
    ],
):
    _df = _SBDB_FIELDS.query(_query)
    SBDB_FIELDS[_name] = {c.strip(): t for c, t in zip(_df["column"], _df["dtype"])}

"""
* `"*"`: Literally all fields
* `"all"`: `"*"` without the `ignore` columns.
*  `"ignore"`: Columns that are extremely unlikely to be used in practice.
*  `"simple"`: Columns that are most likely to be used in practice.
*  `″simple_ast"`: `"simple"` without the comet-related columns.
*  `″simple_com"`: `"simple"` without the asteroid-related columns.
*  `"all_ast"`: `"all"` without the comet-related columns.
*  `"all_com"`: `"all"` without the asteroid-related columns.
*  `"lite"`: A slightly lighter than "all" but heavier than "simple" set.
*  `"lite_ast"`: `"lite"` without the comet-related columns.
*  `"lite_com"`: `"lite"` without the asteroid-related columns.
"""

SBDB_ALLOWED_SBCLASS = [
    "IEO",
    "ATE",
    "APO",
    "AMO",
    "MCA",
    "IMB",
    "MBA",
    "OMB",
    "TJN",
    "AST",
    "CEN",
    "TNO",
    "PAA",
    "HYA",
    "ETc",
    "JFc",
    "JFC",
    "CTc",
    "HTC",
    "PAR",
    "HYP",
    "COM",
]

# **************************************************************************************** #
#                                     Impacted Objects                                     #
# **************************************************************************************** #
# impacted and permanently lost objects by 2024. see also
# https://en.wikipedia.org/wiki/Asteroid_impact_prediction#List_of_successfully_predicted_asteroid_impacts
IMPACTED = [
    # Asteroids
    "2008 TC3",  # On Earth at 2008-10-07T02:48:50.400+00:00
    "2014 AA",  # On Earth at 2014-01-02T02:30:35.086+00:00
    "2018 LA",  # On Earth at 2018-06-02T16:48:08.746+00:00
    "2019 MO",  # On Earth at 2019-06-22T21:29:24.866+00:00
    "2022 EB5",  # On Earth at 2022-03-11T21:25:49.509+00:00
    "2022 WJ1",  # On Earth at 2022-11-19T08:33:48.418+00:00
    "2023 CX1",  # On Earth at 2023-02-13T03:07:11.999+00:00
    "2024 BX1",  # On Earth at 2024-01-21T00:37:09.875+00:00
    "2024 RW1",  # On Earth at 2024-09-04T16:42:09.431+00:00
    "2024 UQ",  # On Earth at 2024-10-22T10:55:07.739+00:00
    "2024 XA1",  # 2024-12-03 16:14
    # Comets
    "1981 V1",  # On Sun at 1981-11-04T12:45:12.615+00:00
    "1997 T2",  # On Sun at 1997-10-04T07:39:44.815+00:00
    "2002 X14",  # On Sun at 2002-12-12T07:10:55.815+00:00
    "1989 N3",  # On Sun at 1989-07-08T18:30:44.615+00:00
    "2003 K9",  # On Sun at 2003-05-24T20:51:43.816+00:00
    "2003 L8",  # On Sun at 2003-06-16T02:08:31.815+00:00
    "2003 M1",  # On Sun at 2003-06-16T15:06:07.815+00:00
    "2003 M2",  # On Sun at 2003-06-18T21:34:55.815+00:00
    "2003 M3",  # On Sun at 2003-06-18T09:20:31.816+00:00
    "2001 M8",  # On Sun at 2001-06-27T12:56:31.816+00:00
    "2008 C7",  # On Sun at 2008-02-09T21:34:54.815+00:00
    "2007 M5",  # On Sun at 2007-06-25T12:42:06.815+00:00
    "2007 V4",  # On Sun at 2007-11-03T19:32:30.815+00:00
    "2008 D9",  # On Sun at 2008-03-01T02:43:04.416+00:00
    "2008 H3",  # On Sun at 2008-04-17T06:27:42.816+00:00
    "2007 X10",  # On Sun at 2007-12-14T10:46:54.816+00:00
    "2008 J14",  # On Sun at 2008-05-14T09:34:54.815+00:00
    "2005 L7",  # On Sun at 2005-06-07T08:51:43.816+00:00
    "2005 L9",  # On Sun at 2005-06-07T09:49:19.816+00:00
    "2005 W16",  # On Sun at 2005-11-29T04:03:43.815+00:00
    "2005 L12",  # On Sun at 2005-06-12T16:32:31.815+00:00
    "2005 X5",  # On Sun at 2005-12-09T10:03:43.815+00:00
    "2004 Q6",  # On Sun at 2004-08-26T21:20:31.816+00:00
    "2007 E4",  # On Sun at 2007-03-03T11:30:06.816+00:00
    "2005 Y9",  # On Sun at 2005-12-27T17:58:55.815+00:00
    "2006 A5",  # On Sun at 2006-01-05T15:49:18.816+00:00
    "2006 X7",  # On Sun at 2006-12-12T17:30:06.816+00:00
]
