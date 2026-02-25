"""General ephemeris parquet compression/decompression utilities.

This module provides functions for compacting ephemeris DataFrames with
scaled integer storage and HEALPix coordinate encoding, and for loading
such compacted files with automatic decompression.

Attributes
----------
EPH_DTYPES_BASE : dict
    Base ephemeris column compression specifications. Each entry maps a column
    name to a 4-tuple ``(factor, stored_dtype, navalue, desired_dtype)``:

    - **factor** (int): Multiplication factor applied before integer conversion.
    - **stored_dtype** (str): NumPy dtype for storage (e.g., ``'uint16'``).
    - **navalue** (int): Sentinel value for NaN or out-of-bounds entries.
    - **desired_dtype** (str): Output dtype when loading (e.g., ``'float64'``).

    **Precision notes:**

    - ``alpha``: ±5" precision, covers 0-180 deg phase angle
    - ``r_hel``, ``r_obs``: ±0.02 LD precision, covers 0-65.535 AU
    - ``racosdec_rate``, ``dec_rate``: ±0.5 mas/min precision, covers ±32.7675 "/min
    - ``sky_motion``: ±0.5 mas/min precision, 0-65.535 "/min
    - ``sky_motion_pa``: ±10" precision, covers -180 to +180 deg
    - ``vmag``: ±0.25 mmag precision, covers 0-32.76 mag

DEFAULT_COORD_MAP : dict
    Default coordinate column mapping for HEALPix encoding. Maps coordinate
    system names to ``(lon_col, lat_col)`` tuples:
    ``{"eqj2000": ("ra", "dec"), "helecl": ("hel_ecl_lon", "hel_ecl_lat"), ...}``.
"""

import re
from glob import glob
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from pqfilt._operators import apply_filter_operator, validate_operator

__all__ = [
    "EPH_DTYPES_BASE",
    "parse_hpcolname",
    "coo2hpcol",
    "hpcol2coo",
    "compact_ephem_parq_cols",
    "parse_compact_ephem",
    "load_compact_parq_ephem",
    "filter_ephem",
]

# EPH_DTYPES_BASE: Base ephemeris column compression specifications.
#
# This dictionary defines how floating-point ephemeris columns are compressed
# to integer types for efficient Parquet storage. Each entry maps a column
# name to a 4-tuple: (factor, stored_dtype, navalue, desired_dtype).
#
# Parameters:
#   factor : int
#       Multiplication factor applied before integer conversion.
#       Example: factor=1000 means 0.001 precision for float values.
#   stored_dtype : str
#       NumPy dtype string for storage (e.g., 'uint16', 'int16').
#   navalue : int
#       Sentinel value used for NaN or out-of-bounds entries.
#   desired_dtype : str
#       Output dtype when loading/decompressing (typically 'float64').
#
# Precision notes:
#   - alpha: ±5" precision, covers 0-180 deg phase angle
#   - r_hel, r_obs: 1.e-3au=±0.2 LD precision (~±75_000 km), covers 0-65.535 AU
#   - racosdec_rate, dec_rate: ±0.5 mas/min precision, covers ±32.7675 "/min
#   - sky_motion: ±0.5 mas/min precision, covers 0-65.535 "/min
#   - sky_motion_pa: ±10" precision, covers -180 to +180 deg
#   - vmag: ±0.25 mmag precision, covers 0-32.76 mag
# EPH_DTYPES_BASE = {
#     "alpha": (360, "uint16", 65535, "float64"),
#     "r_hel": (1000, "uint16", 0, "float64"),
#     "r_obs": (1000, "uint16", 0, "float64"),
#     "racosdec_rate": (1000, "int16", 0, "float64"),
#     "dec_rate": (1000, "int16", 0, "float64"),
#     "sky_motion": (1000, "uint16", 0, "float64"),
#     "sky_motion_pa": (180, "int16", 32767, "float64"),
#     "vmag": (2000, "uint16", 0, "float64"),
# }
EPH_DTYPES_BASE = {
    "alpha": (360, "uint16", 65535, "float64"),
    "r_hel": (1, "float32", 0, "float64"),
    "r_obs": (1, "float32", 0, "float64"),
    "racosdec_rate": (1, "float32", 0, "float64"),
    "dec_rate": (1, "float32", 0, "float64"),
    "sky_motion": (1, "float32", 0, "float64"),
    "sky_motion_pa": (180, "int16", 32767, "float64"),
    "vmag": (1, "float32", 0, "float64"),
}

# Default coordinate column mapping: coord_name -> (lon_col, lat_col)
DEFAULT_COORD_MAP = {
    "eqj2000": ("ra", "dec"),
    "helecl": ("hel_ecl_lon", "hel_ecl_lat"),
    "obsecl": ("obs_ecl_lon", "obs_ecl_lat"),
}


def parse_hpcolname(colname: str) -> tuple[str, str, int] | None:
    """Parse a HEALPix column name into its components.

    Expected format: ``{coord_name}_hpidx_{ring|nested}_{nside_expr}``
    where *nside_expr* is either a plain integer or ``base^exp``.

    Parameters
    ----------
    colname : str
        Column name to parse.

    Returns
    -------
    tuple[str, str, int] or None
        ``(coord_name, scheme, nside)`` if *colname* matches, else ``None``.
    """
    match = re.match(r"^(\w+)_hpidx_(ring|nested)_(.+)$", colname)
    if not match:
        return None
    coord_name, scheme, nside_expr = match.groups()
    if "^" in nside_expr:
        base, exp = nside_expr.split("^")
        nside = int(base) ** int(exp)
    else:
        nside = int(nside_expr)
    return coord_name, scheme, nside


def coo2hpcol(
    df: pd.DataFrame,
    coord_name: str,
    lon_col: str,
    lat_col: str,
    nside: int,
    scheme: str = "ring",
) -> pd.DataFrame:
    """Convert a lon/lat coordinate pair to a HEALPix index column.

    Adds a column ``{coord_name}_hpidx_{scheme}_{nside_str}`` (uint64) and
    drops the original *lon_col* and *lat_col*.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (modified in-place).
    coord_name : str
        Label for the coordinate system (e.g., ``'eqj2000'``).
    lon_col, lat_col : str
        Column names for longitude and latitude **in degrees**.
    nside : int
        HEALPix *nside* parameter.
    scheme : str, optional
        ``'ring'`` (default) or ``'nested'``.

    Returns
    -------
    pd.DataFrame
        DataFrame with the HEALPix index column added and lon/lat dropped.
    """
    if lon_col not in df.columns or lat_col not in df.columns:
        return df

    lon_rad = np.deg2rad(df[lon_col].to_numpy())
    lat_rad = np.deg2rad(df[lat_col].to_numpy())
    theta = np.pi / 2 - lat_rad  # colatitude
    hpidx = hp.ang2pix(nside, theta, lon_rad, nest=(scheme == "nested"))

    # Build nside string (use exponent notation if power of 2)
    log2_nside = np.log2(nside)
    nside_str = f"2^{int(log2_nside)}" if log2_nside == int(log2_nside) else str(nside)

    df[f"{coord_name}_hpidx_{scheme}_{nside_str}"] = hpidx.astype("uint64")
    df.drop(columns=[lon_col, lat_col], inplace=True)
    return df


def hpcol2coo(
    df: pd.DataFrame,
    hp_col: str,
    coord_name: str,
    lon_col: str,
    lat_col: str,
    update_df: bool = True,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Convert a HEALPix index column back to lon/lat coordinates.

    The scheme and nside are parsed from the column name via
    `parse_hpcolname`.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (modified in-place if ``update_df=True``).
    hp_col : str
        Name of the HEALPix index column.
    coord_name : str
        Label for the coordinate system (must match *hp_col*).
    lon_col, lat_col : str
        Output column names for longitude and latitude **in degrees**.
    update_df : bool, optional
        If ``True`` (default), also return *df* with
        *lon_col*/*lat_col* added and *hp_col* dropped.

    Returns
    -------
    lon_deg, lat_deg : np.ndarray
        Longitude and latitude arrays in degrees.
    df_out : pd.DataFrame
        Returned **only** when ``update_df=True``.  Same as *df* with
        coordinate columns added and HEALPix column dropped.

    Raises
    ------
    ValueError
        If *hp_col* cannot be parsed by `parse_hpcolname`.
    """
    parsed = parse_hpcolname(hp_col)
    if parsed is None:
        raise ValueError(f"Cannot parse HEALPix column name: '{hp_col}'")
    _, scheme, nside = parsed

    theta, phi = hp.pix2ang(
        nside, df[hp_col].to_numpy().astype("int64"), nest=(scheme == "nested")
    )
    lon_deg = np.rad2deg(phi)
    lat_deg = 90.0 - np.rad2deg(theta)

    if not update_df:
        return lon_deg, lat_deg

    df[lon_col] = lon_deg
    df[lat_col] = lat_deg
    df.drop(columns=[hp_col], inplace=True)
    return lon_deg, lat_deg, df


def compact_ephem_parq_cols(
    eph,
    dtypes=None,
    nside=2**29,
    coord_cols=None,
    scheme="ring",
    drop_cols=None,
):
    """Compacts ephemeris DataFrame columns for efficient parquet storage.

    This function applies two types of compression:
    1. Scales floating-point columns by a factor and stores as integer types.
    2. Converts coordinate pairs (lon/lat) to HEALPix indices.

    Parameters
    ----------
    eph : pd.DataFrame
        Ephemeris DataFrame with columns to compact.

    dtypes : dict, optional
        Dictionary mapping column names to (factor, storing_dtype, navalue, original_dtype).

          - factor: multiplication factor before storing
          - storing_dtype: numpy dtype string for storage (e.g., 'uint16', 'int16')
          - navalue: value to use for NaN and out-of-bounds entries
          - original_dtype: output dtype when loading (not used during compaction)

        If None, uses `EPH_DTYPES_BASE`.

    nside : int, optional
        HEALPix nside parameter. Default is 2**29 (~0.04 arcsec resolution).

    coord_cols : list of tuples, optional
        List of (coord_name, lon_col, lat_col) for coordinate columns to convert
        to HEALPix indices. If None, uses DEFAULT_COORD_MAP entries that exist
        in the DataFrame.

    scheme : str, optional
        HEALPix indexing scheme, 'ring' or 'nested'. Default is 'ring'.

    drop_cols : list of str, optional
        Columns to drop before compaction. Default is None.

    Returns
    -------
    pd.DataFrame
        Compacted DataFrame with scaled columns named '{col}*{factor}' and
        HEALPix columns named '{coord}_hpidx_{scheme}_{nside_str}'.
    """
    _eph = eph.copy()

    # Drop specified columns
    if drop_cols:
        for col in drop_cols:
            if col in _eph.columns:
                _eph = _eph.drop(columns=[col])

    # Compact factored columns
    if dtypes is None:
        dtypes = EPH_DTYPES_BASE
    elif not isinstance(dtypes, dict):
        raise TypeError(
            "dtypes must be a dict mapping column names to "
            "(factor<numeric>, storing_dtype, navalue<numeric>, original_dtype)"
        )

    for col, (factor, storing_dtype, navalue, _) in dtypes.items():
        if col not in _eph.columns:
            continue

        # Get dtype bounds
        dtype = np.dtype(storing_dtype)
        if np.issubdtype(dtype, np.integer):
            dtype_info = np.iinfo(dtype)
        elif np.issubdtype(dtype, np.floating):
            dtype_info = np.finfo(dtype)
        else:
            raise TypeError(f"Unsupported storing_dtype: {storing_dtype}")

        min_val, max_val = dtype_info.min, dtype_info.max
        col_data = _eph[col]

        # Calculate bounds in original units
        val_min_orig = min_val / factor
        val_max_orig = max_val / factor
        out_of_bounds = (col_data < val_min_orig) | (col_data > val_max_orig)

        scaled = col_data * factor
        scaled[out_of_bounds | pd.isna(col_data)] = navalue

        # Drop original column and create new compacted column
        _eph = _eph.drop(columns=[col])
        _eph[f"{col}*{factor}"] = scaled.astype(storing_dtype)

    # Determine coordinate columns to convert
    if coord_cols is None:
        coord_cols = []
        for coord_name, (lon_col, lat_col) in DEFAULT_COORD_MAP.items():
            if lon_col in _eph.columns and lat_col in _eph.columns:
                coord_cols.append((coord_name, lon_col, lat_col))

    # Convert coordinates to HEALPix indices
    for coord_name, lon_col, lat_col in coord_cols:
        _eph = coo2hpcol(_eph, coord_name, lon_col, lat_col, nside, scheme)

    return _eph


def parse_compact_ephem(
    df: pd.DataFrame,
    dtypes: dict | None = None,
    coord_map: dict | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Decompress a compacted ephemeris DataFrame.

    This reverses the transformation applied by `compact_ephem_parq_cols`
    on an already-loaded DataFrame:

    - Divides factored columns by their factors to recover original values.
    - Converts HEALPix index columns back to lon/lat coordinates.

    Use this when you already have a compacted DataFrame in memory (e.g.,
    read via ``pd.read_parquet`` or obtained from another source). For
    loading directly from a Parquet *file*, use `load_compact_parq_ephem`.

    Parameters
    ----------
    df : pd.DataFrame
        Compacted ephemeris DataFrame with factored columns (named
        ``'{col}*{factor}'``) and/or HEALPix index columns (named
        ``'{coord}_hpidx_{scheme}_{nside_expr}'``).

    dtypes : dict, optional
        Dictionary mapping original column names to
        ``(factor, stored_dtype, navalue, original_dtype)``.
        Must match the dtypes used during compaction.
        If None, uses `EPH_DTYPES_BASE`.

    coord_map : dict, optional
        Mapping from ``coord_name`` to ``(lon_col, lat_col)``.
        If None, uses `DEFAULT_COORD_MAP`.

    columns : list of str, optional
        Columns to keep in output. Use original names (e.g., ``'alpha'``,
        ``'ra'``). If None, all decompressed columns are returned.

    Returns
    -------
    pd.DataFrame
        Decompressed ephemeris DataFrame with original column names.

    Notes
    -----
    HEALPix parameters (scheme and nside) are automatically parsed from
    column names. Expected format:
    ``{coord_name}_hpidx_{ring|nested}_{nside_expr}``
    """
    if coord_map is None:
        coord_map = DEFAULT_COORD_MAP

    if dtypes is None:
        dtypes = EPH_DTYPES_BASE
    elif not isinstance(dtypes, dict):
        raise TypeError(
            "dtypes must be a dict mapping column names to "
            "(factor<numeric>, stored_dtype, navalue<numeric>, original_dtype)"
        )

    # Build mapping from original col -> (stored_col, factor, navalue, original_dtype)
    factor_map = {
        col: (f"{col}*{factor}", factor, navalue, original_dtype)
        for col, (factor, _, navalue, original_dtype) in dtypes.items()
    }

    df = df.copy()

    # Auto-detect HEALPix columns
    hp_col_info = {}
    for col in df.columns:
        if (parsed := parse_hpcolname(col)):
            hp_col_info[col] = parsed

    # Decompress factored columns
    cols2drop = []
    for orig_col, (stored_col, factor, navalue, original_dtype) in factor_map.items():
        if stored_col in df.columns:
            namask = df[stored_col] == navalue
            if factor == 1:
                df[orig_col] = df[stored_col].astype(original_dtype)
            else:
                vals = (df[stored_col] / factor).astype(original_dtype)
                try:
                    vals[namask] = np.nan
                except ValueError:
                    vals[namask] = 0  # If original_dtype is not float64, set to 0
                df[orig_col] = vals.astype(original_dtype)
            cols2drop.append(stored_col)

    df = df.drop(columns=cols2drop)

    # Decompress HEALPix coordinate columns
    for hp_col, (coord_name, _scheme, _nside) in hp_col_info.items():
        if hp_col in df.columns and coord_name in coord_map:
            lon_col, lat_col = coord_map[coord_name]
            _, _, df = hpcol2coo(df, hp_col, coord_name, lon_col, lat_col)

    # Reorder columns if user requested specific ones
    if columns is not None:
        available = [c for c in columns if c in df.columns]
        df = df[available]

    return df.reset_index(drop=True)


def load_compact_parq_ephem(
    fpath,
    dtypes=None,
    filters=None,
    columns=None,
    coord_map=None,
):
    """Loads a compacted ephemeris parquet file and decompresses it.

    This reverses the transformation applied by `compact_ephem_parq_cols`:
    - Divides factored columns by their factors to recover original values.
    - Converts HEALPix indices back to lon/lat coordinates.

    Parameters
    ----------
    fpath : str or Path
        Path to the compacted ephemeris Parquet file.

    dtypes : dict, optional
        Dictionary mapping original column names to (factor, stored_dtype, navalue, original_dtype).

          - factor: factor multiplied to the stored value for compaction
          - stored_dtype: (not used in this function)
          - navalue: value used instead of NaN for the stored column
          - original_dtype: The final desired dtype after decompression

        Must match the dtypes used during compaction.
        If None, uses `EPH_DTYPES_BASE`.

        .. warning::
            If original dtype is not 64-bit float, NaN values (i.e.,
            entries with compacted value equal to the navalue) will
            silently be set to 0, because `np.nan` cannot be used (e.g.,
            integer).

    filters : list, optional
        Filter expressions using original column names. Supports two formats:

        **AND format** (list of tuples)::

            filters=[('alpha', '>', 10), ('vmag', '<', 20)]

        **DNF format** (list of lists): OR between groups, AND within::

            filters=[[('alpha', '>', 10)], [('vmag', '>', 20)]]

        For factored columns, values are automatically scaled. Coordinate
        columns (stored as HEALPix) are **not supported** for filtering.

        Supported operators: '>', '>=', '<', '<=', '==', '!=', 'in', 'not in'.

    columns : list of str, optional
        Columns to load and return. Use original names (e.g., 'alpha', 'ra').

    coord_map : dict, optional
        Mapping from coord_name to (lon_col, lat_col). If None, uses DEFAULT_COORD_MAP.

    Returns
    -------
    pd.DataFrame
        Decompressed ephemeris DataFrame with original column names and values.

    Raises
    ------
    ValueError
        If a filter is specified for a coordinate column.

    Notes
    -----
    HEALPix parameters (scheme and nside) are automatically parsed from column
    names. Expected format: ``{coord_name}_hpidx_{ring|nested}_{nside_expr}``
    """
    if coord_map is None:
        coord_map = DEFAULT_COORD_MAP

    # Build set of coordinate column names for filter validation
    coord_cols = set()
    for lon_col, lat_col in coord_map.values():
        coord_cols.add(lon_col)
        coord_cols.add(lat_col)

    if dtypes is None:
        dtypes = EPH_DTYPES_BASE
    elif not isinstance(dtypes, dict):
        raise TypeError(
            "dtypes must be a dict mapping column names to "
            "(factor<numeric>, stored_dtype, navalue<numeric>, original_dtype)"
        )

    # Build mapping from original col -> (stored_col, factor, navalue, original_dtype)
    factor_map = {
        col: (f"{col}*{factor}", factor, navalue, original_dtype)
        for col, (factor, _, navalue, original_dtype) in dtypes.items()
    }

    def _transform_filter(filt):
        """Transform a single filter tuple to use stored column name."""
        col, op, val = filt
        if col in coord_cols:
            raise ValueError(
                f"Filtering on coordinate column '{col}' is not supported. "
                "Coordinate columns are stored as HEALPix indices and cannot be "
                "efficiently pre-filtered. Filter manually after loading."
            )
        elif col in factor_map:
            stored_col, factor, _, _ = factor_map[col]
            return (stored_col, op, val * factor)
        else:
            return (col, op, val)

    def _build_pa_expr(col, op, val):
        """Build a PyArrow expression from a filter tuple."""
        validate_operator(op)
        return apply_filter_operator(op, ds.field(col), val)

    # Normalize filters to DNF (list of lists)
    pre_filter_groups = []
    if filters:
        if isinstance(filters[0], list):  # (list of lists)
            for group in filters:
                if (pre_group := [_transform_filter(filt) for filt in group]):
                    pre_filter_groups.append(pre_group)
        elif (pre_group := [_transform_filter(filt) for filt in filters]):
            pre_filter_groups.append(pre_group)

    # Determine which stored columns to load
    dataset = ds.dataset(fpath, format="parquet")
    all_stored_cols = set(dataset.schema.names)

    # Auto-detect HEALPix columns
    hp_col_info = {}
    for col in all_stored_cols:
        if (parsed := parse_hpcolname(col)):
            hp_col_info[col] = parsed

    # Build reverse map: original_col -> hp_col
    orig_to_hp = {}
    for hp_col, (coord_name, _, _) in hp_col_info.items():
        if coord_name in coord_map:
            lon_col, lat_col = coord_map[coord_name]
            orig_to_hp[lon_col] = hp_col
            orig_to_hp[lat_col] = hp_col

    if columns is not None:
        load_stored_cols = set()
        for col in columns:
            if col in factor_map:
                stored_col, _, _, _ = factor_map[col]
                if stored_col in all_stored_cols:
                    load_stored_cols.add(stored_col)
            elif col in orig_to_hp:
                load_stored_cols.add(orig_to_hp[col])
            elif col in all_stored_cols:
                load_stored_cols.add(col)
        load_stored_cols = list(load_stored_cols)
    else:
        load_stored_cols = None

    # Build pyarrow filter expression from DNF groups
    pa_filter = None
    if pre_filter_groups:
        or_exprs = []
        for group in pre_filter_groups:
            and_exprs = [_build_pa_expr(col, op, val) for col, op, val in group]
            group_expr = and_exprs[0]
            for expr in and_exprs[1:]:
                group_expr = group_expr & expr
            or_exprs.append(group_expr)
        pa_filter = or_exprs[0]
        for expr in or_exprs[1:]:
            pa_filter = pa_filter | expr

    # Load data
    table = dataset.to_table(columns=load_stored_cols, filter=pa_filter)
    df = table.to_pandas()

    # Delegate decompression to parse_compact_ephem
    return parse_compact_ephem(df, dtypes=dtypes, coord_map=coord_map, columns=columns)


def filter_ephem(
    filelist,
    dtypes=None,
    filter_per_file=True,
    output=None,
    overwrite=False,
    columns=None,
    coord_map=None,
    **filter_kwargs,
):
    """Filter ephemeris parquet files and optionally save the result.

    This function loads compacted ephemeris Parquet files, applies filters
    based on column values, and returns the filtered DataFrame. It can process
    files individually or concatenate first, and optionally save the result.

    Parameters
    ----------
    filelist : str, Path, or list-like
        Files to filter. Can be:
        - Glob pattern string (e.g., 'data/*.parquet')
        - Single file path
        - List of file paths

    dtypes : dict, optional
        Dictionary mapping column names to (factor, stored_dtype, navalue, desired_dtype).
        If None, uses EPH_DTYPES_BASE.

    filter_per_file : bool, optional
        If True (default), apply filters to each file separately then concatenate.
        If False, load all files first, concatenate, then filter once.
        Use True for better memory efficiency with large datasets.

    output : str or Path, optional
        Path to save the filtered result as Parquet. If None, result is not saved.

    overwrite : bool, optional
        If True, overwrite existing output file. Default is False.

    columns : list of str, optional
        Columns to load and return. Use original column names.

    coord_map : dict, optional
        Mapping from coord_name to (lon_col, lat_col). If None, uses DEFAULT_COORD_MAP.

    **filter_kwargs
        Filter conditions as keyword arguments. Each kwarg becomes a filter:
        - Column name as key
        - Filter value as value (see formats below)

        **Supported filter formats:**

        **Comparison operators** (as tuples)::

            filter_ephem(files, vmag=('<', 20))
            filter_ephem(files, dec=('>', 30), ra=('<=', 180))

        **Equality/inequality**::

            filter_ephem(files, desig=123)  # Equal to
            filter_ephem(files, desig=('!=', 123))  # Not equal

        **Membership** (lists or tuples without operator)::

            filter_ephem(files, desig=[1, 2, 3])  # desig in [1, 2, 3]
            filter_ephem(files, desig=('not in', [1, 2, 3]))

        Supported operators: '>', '>=', '<', '<=', '==', '!=', 'in', 'not in'

    Returns
    -------
    pd.DataFrame
        Filtered ephemeris DataFrame with original column names and values.

    Raises
    ------
    FileNotFoundError
        If no files match the glob pattern or file list is empty.
    FileExistsError
        If output file exists and overwrite=False.
    ValueError
        If filter syntax is invalid or filters on coordinate columns.

    Examples
    --------
    Filter by magnitude and designation::

        df = filter_ephem('data/*.parquet', vmag=('<', 20), desig=[1, 2, 3])

    Filter and save to file::

        df = filter_ephem(
            'data/*.parquet',
            vmag=('<', 20),
            dec=('>', 30),
            output='filtered.parquet'
        )

    Load all files first, then filter::

        df = filter_ephem(
            ['file1.parquet', 'file2.parquet'],
            filter_per_file=False,
            alpha=('>', 10)
        )

    Notes
    -----
    - Coordinate columns (ra, dec, etc.) cannot be pre-filtered during file loading
      due to HEALPix encoding. To filter on coordinates, use filter_per_file=False
      or filter the returned DataFrame manually.
    - When filter_per_file=True, filters are applied during Parquet read for
      better memory efficiency.
    """
    if dtypes is None:
        dtypes = EPH_DTYPES_BASE

    if coord_map is None:
        coord_map = DEFAULT_COORD_MAP

    # Resolve file list
    if isinstance(filelist, (str, Path)):
        filelist_str = str(filelist)
        if any(c in filelist_str for c in ["*", "?", "[", "]"]):
            files = sorted(glob(filelist_str))
        else:
            files = [filelist_str]
    else:
        files = [str(f) for f in filelist]

    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {filelist}")

    # Parse filter_kwargs into pyarrow filter format
    filters = []
    for col, val in filter_kwargs.items():
        if isinstance(val, tuple):
            if len(val) == 2:
                op, filter_val = val
                if op in ("in", "not in"):
                    filters.append((col, op, filter_val))
                elif op in (">", ">=", "<", "<=", "==", "!="):
                    filters.append((col, op, filter_val))
                else:
                    validate_operator(op, col=col)
            else:
                raise ValueError(
                    f"Filter tuple for '{col}' must have 2 elements (op, value), got {len(val)}"
                )
        elif isinstance(val, (list, tuple)):
            filters.append((col, "in", val))
        else:
            filters.append((col, "==", val))

    # Load and filter data
    if filter_per_file:
        dfs = []
        for fpath in files:
            try:
                df = load_compact_parq_ephem(
                    fpath,
                    dtypes=dtypes,
                    filters=filters if filters else None,
                    columns=columns,
                    coord_map=coord_map,
                )
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                raise RuntimeError(f"Error loading {fpath}: {e}") from e

        if not dfs:
            result = pd.DataFrame()
        else:
            result = pd.concat(dfs, ignore_index=True)
    else:
        dfs = []
        for fpath in files:
            try:
                df = load_compact_parq_ephem(
                    fpath,
                    dtypes=dtypes,
                    filters=None,
                    columns=columns,
                    coord_map=coord_map,
                )
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                raise RuntimeError(f"Error loading {fpath}: {e}") from e

        if not dfs:
            result = pd.DataFrame()
        else:
            result = pd.concat(dfs, ignore_index=True)

            # Apply filters to concatenated DataFrame
            if filters:
                mask = pd.Series(True, index=result.index)
                for col, op, val in filters:
                    if col not in result.columns:
                        continue
                    mask &= apply_filter_operator(op, result[col], val)
                result = result[mask].reset_index(drop=True)

    # Save output if requested
    if output is not None:
        output_path = Path(output)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file '{output}' already exists. Use overwrite=True to replace."
            )
        if output_path.suffix.lower() == ".csv":
            result.to_csv(output, index=False)
        else:
            result.to_parquet(output, index=False)

    return result
