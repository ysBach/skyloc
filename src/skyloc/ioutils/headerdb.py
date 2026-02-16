"""Header database utilities for FITS observation data stored in parquet/CSV.

This module provides base classes for working with FITS header databases,
including WCS key generation and header-based coordinate transformations.
"""

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "generate_sip_wcs_keys",
    "HeaderDBBase",
]


def generate_sip_wcs_keys(sip=3, pv=True, cd=True):
    """Generate list of FITS WCS header keys.

    Creates a list of standard WCS header keywords based on the requested
    features. Useful for extracting WCS-relevant columns from header databases.

    Parameters
    ----------
    sip : int or None, optional
        SIP polynomial order for distortion correction. If ``None``, no SIP
        keys are included. Default is ``3`` (includes A/B/AP/BP up to order 3).

    pv : bool or int, optional
        Include PVi_j projection parameters (used by TPV, ZPN, etc.).
        If int, specifies max j index per axis. If ``True``, uses max=10.
        Default is ``True``.

    cd : bool, optional
        Include CDi_j matrix keys (alternative to PC+CDELT representation).
        Default is ``True``.

    Returns
    -------
    list of str
        WCS header key names.

    Examples
    --------
    Basic usage in a subclass::

        from skyloc.ioutils.headerdb import HeaderDBBase, generate_sip_wcs_keys

        WCS_KEYS = generate_sip_wcs_keys(sip=3, pv=True, cd=True)

        class MyHeaderDB(HeaderDBBase):
            def __init__(self, fpath, ...):
                db = my_loader(fpath)
                super().__init__(db, fpath, wcs_keys=WCS_KEYS)

    """
    # Core WCS keys (always included)
    keys = [
        "WCSAXES",
        "CRPIX1", "CRPIX2",
        "CRVAL1", "CRVAL2",
        "CTYPE1", "CTYPE2",
        "CUNIT1", "CUNIT2",
        "CDELT1", "CDELT2",
        "PC1_1", "PC1_2", "PC2_1", "PC2_2",
        "LONPOLE", "LATPOLE",
        "RADESYS", "EQUINOX",
        "WCSNAME",
    ]

    # CD matrix (alternative to PC+CDELT)
    if cd:
        keys.extend(["CD1_1", "CD1_2", "CD2_1", "CD2_2"])

    # PV projection parameters
    if pv:
        max_j = pv if isinstance(pv, int) else 10
        for i in [1, 2]:
            for j in range(max_j + 1):
                keys.append(f"PV{i}_{j}")

    # SIP distortion coefficients
    if sip is not None and sip > 0:
        # Forward SIP (A, B)
        keys.extend(["A_ORDER", "B_ORDER"])
        for order in range(sip + 1):
            for p in range(order + 1):
                q = order - p
                keys.extend([f"A_{p}_{q}", f"B_{p}_{q}"])

        # Inverse SIP (AP, BP)
        keys.extend(["AP_ORDER", "BP_ORDER"])
        for order in range(sip + 1):
            for p in range(order + 1):
                q = order - p
                keys.extend([f"AP_{p}_{q}", f"BP_{p}_{q}"])

    return keys


class HeaderDBBase:
    """Base class for FITS header databases stored in parquet or CSV.

    Provides common functionality for loading and querying observation header
    databases. Subclass this for mission-specific implementations.

    Parameters
    ----------
    db : pd.DataFrame
        Pre-loaded header database DataFrame.

    fpath : str or Path, optional
        Original file path(s) for reference.

    wcs_keys : list of str, optional
        WCS header keys to use for WCS object creation.
        Default uses ``generate_sip_wcs_keys()``.

    Examples
    --------
    Direct initialization::

        import pandas as pd
        db = pd.read_parquet("headers.parq")
        hdb = HeaderDBBase(db, wcs_keys=generate_sip_wcs_keys(sip=3))

    Using ``from_file`` classmethod::

        hdb = HeaderDBBase.from_file("headers.parq", drop_colna=True)
    """

    def __init__(self, db, fpath=None, wcs_keys=None):
        self.db = db
        self.fpath = fpath
        self.wcs_keys = wcs_keys if wcs_keys is not None else generate_sip_wcs_keys()

    def __repr__(self):
        return f"{self.__class__.__name__}(fpath={self.fpath}, num_rows={len(self.db)})"

    def __len__(self):
        return len(self.db)

    def __getattr__(self, name):
        """Delegate attribute access to underlying DataFrame."""
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return getattr(self.db, name)
        except AttributeError:
            try:
                return self.db[name]
            except KeyError:
                raise AttributeError(
                    f"'{self.__class__.__name__}' has no attribute '{name}'"
                )

    @classmethod
    def from_file(cls, fpath, fmt="auto", drop_colna=True, drop_cols=None, **kwargs):
        """Load header database from parquet or CSV file(s).

        Parameters
        ----------
        fpath : str, Path, or list
            Path(s) to the header database file(s).

        fmt : {``"auto"``, ``"parquet"``, ``"csv"``}, optional
            File format. ``"auto"`` detects from file extension.
            Default is ``"auto"``.

        drop_colna : bool, optional
            If ``True``, drop columns that are completely NaN. Default is ``True``.

        drop_cols : list of str, optional
            Columns to drop from the DataFrame.

        **kwargs
            Additional keyword arguments passed to the constructor.

        Returns
        -------
        HeaderDBBase
            Loaded header database instance.
        """
        if isinstance(fpath, (str, Path)):
            fpath_list = [fpath]
        else:
            fpath_list = list(fpath)

        dfs = []
        for fp in fpath_list:
            fp = Path(fp)
            if not fp.exists():
                raise FileNotFoundError(f"File not found: {fp}")

            # Detect format
            if fmt == "auto":
                suffix = fp.suffix.lower()
                if suffix in (".parq", ".parquet"):
                    _fmt = "parquet"
                elif suffix in (".csv", ".tsv"):
                    _fmt = "csv"
                else:
                    raise ValueError(f"Cannot auto-detect format for: {fp}")
            else:
                _fmt = fmt

            # Load file
            if _fmt == "parquet":
                df = pd.read_parquet(fp)
            elif _fmt == "csv":
                df = pd.read_csv(fp)
            else:
                raise ValueError(f"Unsupported format: {_fmt}")

            dfs.append(df)

        db = pd.concat(dfs, ignore_index=True)

        # Drop specified columns
        if drop_cols:
            cols_to_drop = [c for c in drop_cols if c in db.columns]
            if cols_to_drop:
                db = db.drop(columns=cols_to_drop)

        # Drop all-NA columns
        if drop_colna:
            db = db.dropna(axis="columns", how="all")

        return cls(db, fpath=fpath, **kwargs)

    def wcs(
        self,
        wcs_class=None,
        id_col=None,
        group_col=None,
        file_col="filename",
        filter_ids=None,
        drop_wcs_keys=False,
    ):
        """Create WCS objects from header rows.

        Parameters
        ----------
        wcs_class : class, optional
            WCS class to instantiate from header dict. Default is
            ``astropy.wcs.WCS``. For TAN-SIP projections,
            ``skyloc.ioutils.FastTanSipWCS`` is significantly faster
            (~100x for initialization, ~10x for transformations).

        id_col : str or None, optional
            Column name for observation/frame ID. If ``None``, uses
            ``file_col`` as the ID. Default is ``None``.

        group_col : str or None, optional
            Column name for grouping (e.g., detector number). If ``None``,
            no grouping is applied. Default is ``None``.

        file_col : str, optional
            Column name for filename. Default is ``"filename"``.

        filter_ids : array-like, optional
            Subset of IDs to process. Only used when ``id_col`` is set.
            If ``None``, uses all rows.

        drop_wcs_keys : bool, optional
            If ``True``, drop WCS columns from ``self.db`` after processing.
            Default is ``False``.

        Returns
        -------
        dict
            Structure depends on parameters:

            - ``id_col=None, group_col=None``: ``{filename: wcs_obj}``
            - ``id_col=None, group_col=set``: ``{filename: {group: wcs_obj}}``
            - ``id_col=set, group_col=None``: ``{id: wcs_obj}``
            - ``id_col=set, group_col=set``: ``{id: {group: wcs_obj}}``

        Notes
        -----
        When ``id_col`` is ``None``, the filename column is used as the
        dictionary key, providing a simpler return structure without
        nested ``"wcs"``/``"file"`` dicts.
        """
        if wcs_class is None:
            from astropy.wcs import WCS
            wcs_class = WCS

        # Determine actual ID column
        use_file_as_id = id_col is None
        actual_id_col = file_col if use_file_as_id else id_col

        # Check column exists
        if actual_id_col not in self.db.columns:
            raise ValueError(f"ID column '{actual_id_col}' not found in database")

        # Filter rows
        _db = self.db
        if filter_ids is not None and not use_file_as_id:
            mask = _db[actual_id_col].isin(filter_ids)
            _db = _db.loc[mask]

        # Find available WCS columns
        available_keys = [k for k in self.wcs_keys if k in _db.columns]
        if not available_keys:
            return {}

        # Columns to load
        cols_needed = [actual_id_col]
        if group_col and group_col in _db.columns:
            cols_needed.append(group_col)
        cols_needed.extend(available_keys)
        cols_needed = list(dict.fromkeys(c for c in cols_needed if c in _db.columns))

        # Clean and convert
        df_clean = _db[cols_needed].where(pd.notna(_db[cols_needed]), None)
        row_dicts = df_clean.to_dict("records")

        # Build WCS dictionary
        wcs_dict = {}

        for row in row_dicts:
            obs_id = row.get(actual_id_col)
            if obs_id is None:
                continue

            group_val = row.get(group_col) if group_col else None

            # Build header dict
            hdr = {k: row[k] for k in available_keys if row.get(k) is not None}
            if not hdr:
                continue

            try:
                wcs_obj = wcs_class(hdr)
            except Exception:
                continue

            # Store based on grouping
            if group_col:
                wcs_dict.setdefault(obs_id, {})
                wcs_dict[obs_id][group_val] = wcs_obj
            else:
                wcs_dict[obs_id] = wcs_obj

        # Optionally drop WCS columns
        if drop_wcs_keys:
            cols_to_drop = [c for c in available_keys if c in self.db.columns]
            if cols_to_drop:
                self.db = self.db.drop(columns=cols_to_drop)

        return wcs_dict
