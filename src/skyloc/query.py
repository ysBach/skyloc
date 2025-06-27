from pathlib import Path
import pandas as pd

from .jplsbdb import SBDBQuery, sanitize_sbdb
from .configs import KETE_SBDB_MINIMUM_FIELDS, KETE_SBDB2KETECOLS

__all__ = [
    "fetch_orb",
]


def fetch_orb(
    output="orb.parq",
    update_output=False,
    server="jplsbdb",
    fields="simple",
    drop_impacted=True,
    drop_unreliable=True,
    kind2bools=True,
    engine="auto",
    compression="snappy",
    filters=None,
):
    """Fetch the orbit data from API and save it to a parquet file.

    Parameters
    ----------
    output : path-like, optional
        The path to the output file. If `None`, it will be saved to the
        default cache path (see `kete.cache.cache_path()`) with filename
        ``"horizons_orbits.parq"``.
        Default is ``"orb.parq"``.

    update_output : bool, float, optional
        If `True`, update the entries with newer "soln_date" and remove
        entries that are not in the latest "pdes" list (deleted objects).
        The `fields` must contain **both** ``"pdes"`` and ``"soln_date"``
        fields. To completely update the output file, remove the existing file.
        If `float`, it will check last-modified time of the file and update the
        entries if the file is older than the given number of days. Ironically,
        setting `update_output` to `False` will be equivalent to setting it to
        ``0``.
        Default is `False`.

    server : str, optional
        The server to query. Currently, only "jplsbdb" is supported. Hopefully,
        in the future, it will be possible to query other servers like
        "iaumpc".
        Default is "jplsbdb".

    kind2bools : bool, optional
        If `True`, convert JPL SBDB's "kind" field (one of ``{"an", "au", "cn",
        "cu"}`` for asteroids and comets, numbered and unnumbered objects) to
        boolean mask columns of ``"is_comet"`` and ``"has_number"``. The
        original "kind" field is dropped.
        Default is `True`.

    fields : list, optional
        List of fields to download from API. If `None`, it will
        download the minimum fields (see `configs.KETE_SBDB_MINIMUM_FIELDS`).
        If `fields` is superset of already downloaded file, it will likely
        raise an error. Recommended to delete the existing file.
        Default is "simple" (see `~configs.SBDB_FIELDS`).

    drop_impacted : bool, optional
        If `True`, drop impacted objects based on the `IMPACTED` list.
        Default is `True`.

    drop_unreliable : bool, optional
        If `True`, drop unreliable objects based on the following criteria:
        - no magnitude-related information
        - prefix is "D" or "X" (disappeared or lost comets)
        - solution date is NaN (no reliable orbit)
        - two body is "T" (two body assumed orbit - unreliable)
        Default is `True`.

    engine, compression : str, optional
        The engine and compression to use for saving the parquet file
        (see `pandas.DataFrame.to_parquet` for details).

    filters : list-like, optional
        A list of filters to apply when reading the parquet file. This is
        passed to `pandas.read_parquet`. If `None`, no filters are applied.
        Example: ``filters=[("kind", "in", ("an", "cn")), ("condition_code",
        "==", "0")]`` to read only numbered asteroids and comets with the
        highest quality orbits.
        Default is `None`.

    Returns
    -------
    orb : `~pandas.DataFrame`
        The orbit data with columns of orbital elements.

    m_ng : `~pandas.Series`
        A boolean mask indicating which objects have non-gravitational
        parameters (i.e., `A1`, `A2`, `A3`, `DT` are not all zero).
        This is useful to filter out objects that do not have non-gravitational
        parameters, which are typically comets or asteroids with non-gravitational
        perturbations.

    Notes
    -----
    If, for some reasons, the user wants to save without dropping impacted or
    unreliable objects, they can just run `fetch_orb` with
    `drop_impacted=False`, etc, and then afterwards, load them by
    `fetch_orb(update_output=False)`.
    """
    if output is None:
        from kete.cache import cache_path

        output = Path(cache_path()) / "horizons_orbits.parq"
    else:
        output = Path(output)

    if output.exists():
        if not isinstance(update_output, bool):
            import time

            # delta-time in days
            dt = (time.time() - output.stat().st_mtime) / 86400
            update_output = dt > update_output

    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    if server == "jplsbdb":
        orb = _fetch_orb_sbdb(
            output=output,
            update_output=update_output,
            fields=fields,
            drop_impacted=drop_impacted,
            drop_unreliable=drop_unreliable,
            kind2bools=kind2bools,
            engine=engine,
            compression=compression,
            filters=filters,
        )
        m_ng = (
            (orb["A1"] != 0.0)
            | (orb["A2"] != 0.0)
            | (orb["A3"] != 0.0)
            | (orb["DT"] != 0.0)
        )
    else:
        raise ValueError(
            f"Server {server} is not supported. Only 'jplsbdb' is available."
        )

    return orb, m_ng


def _fetch_orb_sbdb(
    output=None,
    update_output=False,
    fields=None,
    drop_impacted=True,
    drop_unreliable=True,
    kind2bools=True,
    engine="auto",
    compression="snappy",
    filters=None,
):
    float_cols = list(KETE_SBDB2KETECOLS.values())[:-1] + [
        "epoch",
        "H",
        "G",
        "M1",
        "M2",
        "K1",
        "K2",
        "PC",
        "A1",
        "A2",
        "A3",
        "DT",
    ]

    def _postproc(orb, kind2bools):
        # Drop unreliable or impacted objects
        orb = sanitize_sbdb(
            orb, drop_impacted=drop_impacted, drop_unreliable=drop_unreliable
        )
        # Update column names to match the `kete` convention
        orb = orb.rename(columns=KETE_SBDB2KETECOLS)

        # Fill non-grav model params with 0.0 for convenience (when using
        # NonGravModel.new_comet)
        for col in ["A1", "A2", "A3", "DT"]:
            try:
                orb[col] = orb[col].fillna(0.0)
            except KeyError:
                continue

        if kind2bools:
            _str = orb["kind"].str
            orb["is_comet"] = _str.startswith("c")
            orb["has_number"] = _str.endswith("n")
            orb.drop(columns=["kind"], inplace=True)

        return orb

    if fields is None:
        fields = KETE_SBDB_MINIMUM_FIELDS
    elif isinstance(fields, list):
        fields = tuple(fields)  # Ensure fields is a tuple for consistency
    # else: (str), leave as is

    if not output.exists():
        # Default setting similar to kete.horizons.fetch_known_orbit_data:
        q = SBDBQuery(fields=fields, full_prec=True, sb_xfrag=True)
        orb = q.query()

        orb = _postproc(orb, kind2bools=kind2bools)

        orb.to_parquet(str(output), engine=engine, compression=compression)
        # TODO: Do we need full freedom for all the args for `to_parquet`?

    # File exists, but the user wants to update it
    elif update_output:
        orb = pd.read_parquet(str(output), engine=engine)
        # Read the original parquet file
        latest_date = orb["soln_date"].max()[:10]  # YYYY-MM-DD in PST
        # === Query 1: pdes - to check any deleted entries
        _pdes = SBDBQuery(fields=["pdes"], full_prec=False, sb_xfrag=True).query()
        # Drop entries in orb that are not in the latest pdes list (deleted objects)
        orb = orb[orb["desig"].isin(_pdes["pdes"])]

        # === Query 2: all fields - use "soln_date" to check any new entries
        q_new = SBDBQuery(
            fields=fields,
            full_prec=True,
            sb_xfrag=True,
            sb_cdata='{"AND":["soln_date|GE|' + latest_date + '"]}',
        )

        # If q_new.fields is not equal to orb.columns (DataFrame), except for
        # kete-mapping columns, raise an error:
        cols_new = set(q_new.fields) - set(KETE_SBDB2KETECOLS.keys())
        cols_old = set(orb.columns) - set(KETE_SBDB2KETECOLS.values())
        if cols_new != cols_old:
            raise ValueError(
                "The fields in the new query do not match the existing DataFrame columns.\n"
                + f"Former: {cols_old}\n Latter: {cols_new}\n"
                + "Please ensure the fields are consistent."
            )

        new_orb = q_new.query()
        new_orb = _postproc(new_orb, kind2bools=kind2bools)

        # === Append new entries to the existing DataFrame
        orb = pd.concat([orb, new_orb], ignore_index=True).drop_duplicates(
            subset="desig", keep="last"
        )

        # Save the updated DataFrame to the parquet file (overwriting the old one)
        orb.to_parquet(str(output), engine=engine, compression=compression)

    orb = pd.read_parquet(str(output), engine=engine, filters=filters)

    # Convert the columns to the appropriate dtypes
    for col in float_cols:
        if col in orb.columns:
            orb[col] = orb[col].astype(float)

    return orb
