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
    - ``dra*cosdec/dt``, ``ddec/dt``: ±0.5 mas/min precision
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

from ._filter_utils import apply_filter_operator, validate_operator

__all__ = [
    "EPH_DTYPES_BASE",
    "compact_ephem_parq_cols",
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
#   - r_hel, r_obs: ±0.02 LD precision (~1500 km), covers 0-65.535 AU
#   - dra*cosdec/dt, ddec/dt: ±0.5 mas/min precision
#   - sky_motion: ±0.5 mas/min precision, 0-65.535 "/min
#   - sky_motion_pa: ±10" precision, covers -180 to +180 deg
#   - vmag: ±0.25 mmag precision, covers 0-32.76 mag
EPH_DTYPES_BASE = {
    "alpha": (360, "uint16", 65535, "float64"),
    "r_hel": (1000, "uint16", 0, "float64"),
    "r_obs": (1000, "uint16", 0, "float64"),
    "dra*cosdec/dt": (1000, "int16", 0, "float64"),
    "ddec/dt": (1000, "int16", 0, "float64"),
    "sky_motion": (1000, "uint16", 0, "float64"),
    "sky_motion_pa": (180, "int16", 32767, "float64"),
    "vmag": (2000, "uint16", 0, "float64"),
}

# Default coordinate column mapping: coord_name -> (lon_col, lat_col)
DEFAULT_COORD_MAP = {
    "eqj2000": ("ra", "dec"),
    "helecl": ("hel_ecl_lon", "hel_ecl_lat"),
    "obsecl": ("obs_ecl_lon", "obs_ecl_lat"),
}


def compact_ephem_parq_cols(
    eph,
    dtypes,
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

    dtypes : dict
        Dictionary mapping column names to (factor, stored_dtype, navalue, desired_dtype).
        - factor: multiplication factor before storing
        - stored_dtype: numpy dtype string for storage (e.g., 'uint16', 'int16')
        - navalue: value to use for NaN and out-of-bounds entries
        - desired_dtype: output dtype when loading (not used during compaction)

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
    for col, (factor, dtype, navalue, _) in dtypes.items():
        if col not in _eph.columns:
            continue

        # Get dtype bounds
        dtype_info = np.iinfo(dtype)
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
        _eph[f"{col}*{factor}"] = scaled.astype(dtype)

    # Build nside string for column name
    # If nside is a power of 2, use exponent notation
    log2_nside = np.log2(nside)
    if log2_nside == int(log2_nside):
        nside_str = f"2^{int(log2_nside)}"
    else:
        nside_str = str(nside)

    # Determine coordinate columns to convert
    if coord_cols is None:
        coord_cols = []
        for coord_name, (lon_col, lat_col) in DEFAULT_COORD_MAP.items():
            if lon_col in _eph.columns and lat_col in _eph.columns:
                coord_cols.append((coord_name, lon_col, lat_col))

    # Convert coordinates to HEALPix indices using healpy
    # healpy uses (theta, phi) in radians where theta=colatitude (0 at N. pole)
    nest = scheme == "nested"
    for coord_name, lon_col, lat_col in coord_cols:
        if lon_col not in _eph.columns or lat_col not in _eph.columns:
            continue
        lon_rad = np.deg2rad(_eph[lon_col].to_numpy())
        lat_rad = np.deg2rad(_eph[lat_col].to_numpy())
        theta = np.pi / 2 - lat_rad  # colatitude = 90° - latitude
        hpidx = hp.ang2pix(nside, theta, lon_rad, nest=nest)
        _eph[f"{coord_name}_hpidx_{scheme}_{nside_str}"] = hpidx.astype("uint64")
        # Drop original columns
        _eph = _eph.drop(columns=[lon_col, lat_col])

    return _eph


def load_compact_parq_ephem(
    fpath,
    dtypes,
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

    dtypes : dict
        Dictionary mapping original column names to (factor, stored_dtype, navalue, desired_dtype).
        Must match the dtypes used during compaction.

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

    # Build mapping from original col -> (stored_col, factor, navalue, desired_dtype)
    factor_map = {
        col: (f"{col}*{factor}", factor, navalue, desired_dtype)
        for col, (factor, _, navalue, desired_dtype) in dtypes.items()
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

    def _parse_healpix_col(colname):
        """Parse HEALPix column name to extract coord_name, scheme, nside."""
        pattern = r"^(\w+)_hpidx_(ring|nested)_(.+)$"
        match = re.match(pattern, colname)
        if not match:
            return None
        coord_name, scheme, nside_expr = match.groups()
        if "^" in nside_expr:
            base, exp = nside_expr.split("^")
            nside = int(base) ** int(exp)
        else:
            nside = int(nside_expr)
        return coord_name, scheme, nside

    # Normalize filters to DNF (list of lists)
    pre_filter_groups = []
    if filters:
        is_dnf = isinstance(filters[0], list)
        if is_dnf:
            for group in filters:
                pre_group = [_transform_filter(filt) for filt in group]
                if pre_group:
                    pre_filter_groups.append(pre_group)
        else:
            pre_group = [_transform_filter(filt) for filt in filters]
            if pre_group:
                pre_filter_groups.append(pre_group)

    # Determine which stored columns to load
    dataset = ds.dataset(fpath, format="parquet")
    all_stored_cols = set(dataset.schema.names)

    # Auto-detect HEALPix columns
    hp_col_info = {}
    for col in all_stored_cols:
        parsed = _parse_healpix_col(col)
        if parsed:
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

    # Decompress factored columns
    for orig_col, (stored_col, factor, navalue, desired_dtype) in factor_map.items():
        if stored_col in df.columns:
            if factor == 1:
                df[orig_col] = df[stored_col].astype(desired_dtype)
            else:
                vals = df[stored_col].astype("float64") / factor
                if navalue != 0:
                    vals = vals.replace(navalue / factor, np.nan)
                df[orig_col] = vals.astype(desired_dtype)
            df = df.drop(columns=[stored_col])

    # Decompress HEALPix coordinate columns using healpy
    hp_cols_processed = set()
    for hp_col, (coord_name, scheme, nside) in hp_col_info.items():
        if hp_col in df.columns and hp_col not in hp_cols_processed:
            nest = scheme == "nested"
            theta, phi = hp.pix2ang(
                nside, df[hp_col].to_numpy().astype("int64"), nest=nest
            )
            lon_deg = np.rad2deg(phi)
            lat_deg = 90.0 - np.rad2deg(theta)  # latitude = 90° - colatitude

            if coord_name in coord_map:
                lon_col, lat_col = coord_map[coord_name]
                df[lon_col] = lon_deg
                df[lat_col] = lat_deg

            df = df.drop(columns=[hp_col])
            hp_cols_processed.add(hp_col)

    # Reorder columns if user requested specific ones
    if columns is not None:
        available = [c for c in columns if c in df.columns]
        df = df[available]

    return df.reset_index(drop=True)


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
