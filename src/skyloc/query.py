from pathlib import Path
from collections.abc import Iterable, Mapping
import logging
import time
from warnings import warn

import numpy as np
import pandas as pd

from .configs import KETE_SBDB2KETECOLS, KETE_SBDB_MINIMUM_FIELDS
from .jplsbdb import SBDBQuery, sanitize_sbdb, cols2bools_sbdb

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_orb",
    "query_orb_by_desig",
]


def _normalize_desigs(desigs):
    """Normalize designation input into a flat, unique list of strings."""
    flat = []

    def _walk(value):
        if value is None:
            return

        if isinstance(value, (str, bytes, np.str_)):
            s = str(value).strip()
            if s:
                flat.append(s)
            return

        if isinstance(value, Mapping):
            s = str(value).strip()
            if s:
                flat.append(s)
            return

        if isinstance(value, np.ndarray):
            for v in value.tolist():
                _walk(v)
            return

        if isinstance(value, Iterable):
            for v in value:
                _walk(v)
            return

        s = str(value).strip()
        if s:
            flat.append(s)

    _walk(desigs)

    if not flat:
        return []

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(flat))


def fetch_orb(
    output="orb.parq",
    update_output=False,
    server="jplsbdb",
    fields="simple",
    drop_impacted=True,
    drop_unreliable=True,
    kind2bools=True,
    neo2bool=True,
    pha2bool=True,
    twobody2bool=True,
    engine="auto",
    compression="snappy",
    filters=None,
    strict_column_match=False,
):
    """Fetch the orbit data from API and save it to a parquet file.

    **Note**: kete is only required when `output=None`. In that case, install with:
    ``pip install skyloc[kete]``

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
        Default is ``"jplsbdb"``.

    kind2bools, neo2bool, pha2bool, twobody2bool : bool, optional
        Whether to convert certain string columns to boolean columns. See
        `jplsbdb.cols2bools_sbdb` for details.
        Default is `True` for all.

    fields : list, optional
        List of fields to download from API. If `None`, it will
        download the minimum fields (see `configs.KETE_SBDB_MINIMUM_FIELDS`).
        If `fields` is superset of already downloaded file, it will likely
        raise an error. Recommended to delete the existing file.
        Default is ``"simple"`` (see `~configs.SBDB_FIELDS`).

    drop_impacted, drop_unreliable : bool, optional
        Whether to drop objects that are marked as impacted or unreliable
        in the SBDB data. See `jplsbdb.sanitize_sbdb` for details.
        Defaults are `True` for both.

    engine, compression : str, optional
        The engine and compression to use for saving the parquet file
        (see `pandas.DataFrame.to_parquet` for details).

    filters : str, list-like, or ExprNode, optional
        A list of filters to apply when reading the parquet file. This is
        passed either to `pqfilt.read` (if a string or ExprNode) or natively
        to `pandas.read_parquet` (if a list).

        If string: Uses the `pqfilt` expression syntax.
        Example: ``filters="kind in 'an','cn' & condition_code == 0"``

        If list: Uses PyArrow/Pandas native filter syntax.
        Example: ``filters=[("kind", "in", ("an", "cn")), ("condition_code",
        "==", "0")]``

        If `None` (default), no filters are applied to read all data.

    strict_column_match : bool, optional
        (not matured yet) If `True`, when updating the existing parquet file,
        check if the fields in the new query match the columns in the existing
        DataFrame, except for the columns that are mapped by
        `KETE_SBDB2KETECOLS`. This may be erroneous if the user changes the
        `fields` parameter using `kind2bools`, because it removes `"kind"` and
        adds `"is_comet"` and `"has_number"`...
        Default is `False`.

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
        from .keteutils._kete_import import require_kete

        require_kete()
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
            neo2bool=neo2bool,
            pha2bool=pha2bool,
            twobody2bool=twobody2bool,
            engine=engine,
            compression=compression,
            filters=filters,
            strict_column_match=strict_column_match,
        )
        m_ng = (
            ((orb["A1"] != 0.0) & ~pd.isna(orb["A1"]))
            | ((orb["A2"] != 0.0) & ~pd.isna(orb["A2"]))
            | ((orb["A3"] != 0.0) & ~pd.isna(orb["A3"]))
            | ((orb["DT"] != 0.0) & ~pd.isna(orb["DT"]))
        )
    else:
        raise ValueError(
            f"Server {server} is not supported. Only 'jplsbdb' is available."
        )

    logger.info("Loaded %d objects (%d columns)", orb.shape[0], orb.shape[1])

    # Expensive stats only at DEBUG level
    if logger.isEnabledFor(10):  # logging.DEBUG = 10
        n_ng = np.sum(m_ng)
        logger.debug("Latest soln_date [US/Pacific]: %s", orb["soln_date"].max())
        logger.debug("Grav: %d | Non-grav: %d", orb.shape[0] - n_ng, n_ng)
        try:
            n_com = np.sum(orb["is_comet"])
        except KeyError:
            n_com = np.sum(orb["kind"].str.startswith("c"))
        logger.debug("Comets: %d | Non-comet: %d", n_com, orb.shape[0] - n_com)

    return orb, m_ng


def query_orb_by_desig(
    desigs,
    drop_impacted=True,
    drop_unreliable=True,
    kind2bools=True,
    neo2bool=True,
    pha2bool=True,
    twobody2bool=True,
    strict=False,
    max_workers=1,
    max_retries=2,
    retry_delay=0.25,
):
    """Query SBDB for one/few designation strings.

    This uses the single-object SBDB search endpoint (`id_type="search"`) for
    each requested designation and concatenates the results.

    Parameters
    ----------
    desigs : str or iterable
        One or a few designations/search strings to query. Nested iterables
        (including NumPy arrays) are flattened.

    strict : bool, optional
        If `True`, raise `ValueError` when any requested designation is not
        found. If `False` (default), missing designations are skipped.

    max_workers : int, optional
        Number of worker threads for querying multiple designations.
        Use values > 1 to query several designations concurrently.

    max_retries : int, optional
        Number of retries for transient query failures (not for true not-found
        responses). Default is 2.

    retry_delay : float, optional
        Initial delay in seconds between retries. Exponential backoff is used.

    Returns
    -------
    orb : `~pandas.DataFrame`
        Orbit table in a kete/skyloc-compatible schema.
    """
    desigs = _normalize_desigs(desigs)
    if not desigs:
        raise ValueError("`desigs` must contain at least one non-empty designation.")

    rows = []
    missing = []
    failed = []

    def _query_one(d):
        last_err = None
        for i_try in range(max_retries + 1):
            status, row, err = _query_single_orb_by_search(d)
            if status != "error":
                return status, row, err
            last_err = err
            if i_try < max_retries:
                time.sleep(retry_delay * (2**i_try))
        return "error", None, last_err

    if max_workers > 1 and len(desigs) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_query_one, desigs))
        for d, (status, row, err) in zip(desigs, results):
            if status == "ok":
                rows.append(row)
            elif status == "not_found":
                missing.append(d)
            else:
                failed.append((d, err))
    else:
        for d in desigs:
            status, row, err = _query_one(d)
            if status == "ok":
                rows.append(row)
            elif status == "not_found":
                missing.append(d)
            else:
                failed.append((d, err))

    if missing:
        msg = f"Could not find {len(missing)} designation(s): {missing}"
        logger.warning(msg)

    if failed:
        failed_desigs = [d for d, _ in failed]
        logger.warning(
            "Query failed for %d designation(s): %s",
            len(failed_desigs),
            failed_desigs,
        )
        logger.debug("Failure details: %s", failed)

    if strict and (missing or failed):
        parts = []
        if missing:
            parts.append(f"not found={missing}")
        if failed:
            parts.append(f"failed={[d for d, _ in failed]}")
        raise ValueError("Failed to query all requested designations: " + "; ".join(parts))

    if not rows:
        return pd.DataFrame(columns=["desig"])

    orb = pd.DataFrame(rows)
    orb = cols2bools_sbdb(
        orb,
        kind2bools=kind2bools,
        neo2bool=neo2bool,
        pha2bool=pha2bool,
        twobody2bool=twobody2bool,
    )
    orb = sanitize_sbdb(
        orb,
        drop_impacted=drop_impacted,
        drop_unreliable=drop_unreliable,
    )
    orb = orb.drop_duplicates(subset="desig", keep="last")
    return orb.reset_index(drop=True)


def _query_single_orb_by_search(
    desig,
):
    """Single-object query by search string via astroquery.

    Returns
    -------
    status : {"ok", "not_found", "error"}
    row : dict or None
    err : str or None
    """
    try:
        from astroquery.jplsbdb import SBDB
    except Exception as e:
        warn(f"astroquery.jplsbdb unavailable for single-object fallback: {e}")
        return "error", None, str(e)

    try:
        obj = SBDB.query(
            desig,
            id_type="search",
            full_precision=True,
            phys=True,
            covariance="mat",
        )
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if (
            "not found" in low
            or "no match" in low
            or "no matches" in low
            or "does not match" in low
            or "matched 0 objects" in low
        ):
            return "not_found", None, msg
        return "error", None, msg

    def _to_float(value, unit=None):
        if value is None:
            return np.nan
        try:
            if hasattr(value, "to_value"):
                if unit is None:
                    return float(value.to_value())
                try:
                    return float(value.to_value(unit))
                except Exception:
                    # Some fields are already plain floats even when a unit is requested.
                    pass
            return float(value)
        except Exception:
            try:
                return float(value.value)
            except Exception:
                return np.nan

    objinfo = obj.get("object", {})
    phys = obj.get("phys_par", {})
    orbit = obj.get("orbit", {})
    elems = orbit.get("elements", {})
    mpars = orbit.get("model_pars") or {}

    row = {
        "desig": objinfo.get("des", desig),
        "kind": objinfo.get("kind"),
        "prefix": objinfo.get("prefix"),
        "neo": objinfo.get("neo"),
        "pha": objinfo.get("pha"),
        "H": _to_float(phys.get("H")),
        "G": _to_float(phys.get("G")),
        "M1": _to_float(phys.get("M1")),
        "M2": _to_float(phys.get("M2")),
        "K1": _to_float(phys.get("K1")),
        "K2": _to_float(phys.get("K2")),
        "PC": _to_float(phys.get("PC")),
        "diameter": _to_float(phys.get("diameter")),
        "albedo": _to_float(phys.get("albedo")),
        "spec_B": phys.get("spec_B"),
        "spec_T": phys.get("spec_T"),
        "orbit_id": orbit.get("orbit_id"),
        "epoch": _to_float(orbit.get("epoch"), unit="d"),
        "ecc": _to_float(elems.get("e")),
        "peri_dist": _to_float(elems.get("q"), unit="au"),
        "incl": _to_float(elems.get("i"), unit="deg"),
        "lon_node": _to_float(elems.get("om"), unit="deg"),
        "peri_arg": _to_float(elems.get("w"), unit="deg"),
        "peri_time": _to_float(elems.get("tp"), unit="d"),
        "class": orbit.get("class") or objinfo.get("class"),
        "soln_date": orbit.get("soln_date"),
        "condition_code": orbit.get("condition_code"),
        "rms": _to_float(orbit.get("rms")),
        "two_body": orbit.get("two_body"),
        "A1": _to_float(mpars.get("A1"), unit="au / d2"),
        "A2": _to_float(mpars.get("A2"), unit="au / d2"),
        "A3": _to_float(mpars.get("A3"), unit="au / d2"),
        "DT": _to_float(mpars.get("DT"), unit="d"),
    }
    return "ok", row, None


def _fetch_orb_sbdb(
    output=None,
    update_output=False,
    fields=None,
    drop_impacted=True,
    drop_unreliable=True,
    kind2bools=True,
    neo2bool=True,
    pha2bool=True,
    twobody2bool=True,
    cols2kete=True,
    engine="auto",
    compression="snappy",
    strict_column_match=False,
    filters=None,
):
    """Fetch the orbit data from JPL SBDB and save it to a parquet file.

    Parameters
    ----------
    """
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

    if fields is None:
        fields = KETE_SBDB_MINIMUM_FIELDS
    elif isinstance(fields, list):
        fields = tuple(fields)  # Ensure fields is a tuple for consistency
    # else: (str), leave as is

    if not output.exists():
        # Default setting similar to kete.horizons.fetch_known_orbit_data:
        q = SBDBQuery(fields=fields, full_prec=True, sb_xfrag=True)
        orb = q.query(
            kind2bools=kind2bools,
            neo2bool=neo2bool,
            pha2bool=pha2bool,
            twobody2bool=twobody2bool,
            drop_impacted=drop_impacted,
            drop_unreliable=drop_unreliable,
            cols2kete=cols2kete,
        )

        # Convert the columns to the appropriate dtypes
        for col in float_cols:
            if col in orb.columns:
                orb[col] = orb[col].astype(float)

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

        # Just in case, apply post-processing to original orb:
        orb = cols2bools_sbdb(
            orb,
            kind2bools=kind2bools,
            neo2bool=neo2bool,
            pha2bool=pha2bool,
            twobody2bool=twobody2bool,
        )
        orb = sanitize_sbdb(
            orb, drop_impacted=drop_impacted, drop_unreliable=drop_unreliable
        )
        if cols2kete:
            orb = orb.rename(columns=KETE_SBDB2KETECOLS)

        if strict_column_match:
            # If q_new.fields is not equal to orb.columns (DataFrame), except for
            # kete-mapping columns, raise an error:
            cols_new = set(q_new.fields) - set(KETE_SBDB2KETECOLS.keys())
            cols_old = set(orb.columns) - set(KETE_SBDB2KETECOLS.values())
            if cols_new != cols_old:
                raise ValueError(
                    "The fields in the new query do not match the existing DataFrame "
                    + f"columns.\n Former: {cols_old}\n Latter: {cols_new}\n"
                    + "Please ensure the fields are consistent."
                )

        new_orb = q_new.query(
            kind2bools=kind2bools,
            neo2bool=neo2bool,
            pha2bool=pha2bool,
            twobody2bool=twobody2bool,
            drop_impacted=drop_impacted,
            drop_unreliable=drop_unreliable,
            cols2kete=True,
        )

        # === Append new entries to the existing DataFrame
        orb = pd.concat([orb, new_orb], ignore_index=True).drop_duplicates(
            subset="desig", keep="last"
        )

        # After concat, object-dtype boolean columns may contain mixed
        # bool (True/False) and str ("True"/"False"/"T"/"Y") values.
        # pd.Series.astype(bool) is WRONG here because bool("False") == True.
        # Instead, map known truthy string representations explicitly.
        _truthy = {True, "True", "T", "Y", "true", "t", "y", 1}
        for col in ["two_body", "neo", "pha", "is_comet", "has_number"]:
            if col in orb.columns and orb[col].dtype == object:
                orb[col] = orb[col].isin(_truthy)

        # Save the updated DataFrame to the parquet file (overwriting the old one)
        orb.to_parquet(str(output), engine=engine, compression=compression)

    # Finally, read and return the DataFrame for consistency
    if isinstance(filters, str) or type(filters).__name__ in ("FilterExpr", "AndExpr", "OrExpr"):
        import pqfilt
        orb = pqfilt.read(str(output), filters=filters)
    else:
        # Backward compatible pyarrow filter list
        orb = pd.read_parquet(str(output), engine=engine, filters=filters)

    # Convert the columns to the appropriate dtypes
    for col in float_cols:
        if col in orb.columns:
            orb[col] = orb[col].astype(float)

    return orb
