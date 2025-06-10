# SBDB orbital information DB batch query class

from functools import lru_cache
from warnings import warn
import requests
import pandas as pd
import numpy as np

from .configs import SBDB_FIELDS, SBDB_ALLOWED_SBCLASS, IMPACTED

__all__ = [
    "SBDB_FIELDS",
    "SBDBQuery",
    "sanitize_sbdb",
]


@lru_cache()
def _query_cached(base_url, params_tuple):
    """Query with caching."""
    res = requests.get(base_url, params=dict(params_tuple), timeout=240)
    res.raise_for_status()

    # if not res.ok:
    #     raise ValueError(f"Query failed: {res.text}")

    return res.json()


class SBDBQuery:
    """Class to handle SBDB queries.
    It is based on SBDB Query API Version 1.0 (Aug 2021), which is the most
    recent version as of 2024 Sept.
    TODO: When this class gets mature enough, consider adding it to astroquery.
    """

    def __init__(
        self,
        info=None,
        fields="pdes",
        sort=None,
        limit=None,
        limit_from=None,
        full_prec=False,
        sb_ns=None,
        sb_kind=None,
        sb_group=None,
        sb_class=None,
        sb_sat=None,
        sb_xfrag=None,
        sb_defs=None,
        sb_cdata=None,
    ):
        """Get SBDB query URL for small bodies.

        Parameters
        ----------
        info : {"count", "field", "all"}, optional
            When ``"count"`` is selected, return the number of objects
            available in the SBDB. When ``"field"`` is selected, return all
            available output fields. If ``"all"`` is selected, output count and
            field results. See mode ``I`` section in the link below for
            details.
            https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html
            If provided, mode ``I`` will be used, and all other parameters
            except `sp_defs` will completely be ignored.
            Default is `None`.

        fields : str (comma-separated) or list of str, optional
            List of fields to be output. If no fields are specified, only the
            count (number of records matching found) is output. Field names are
            **case-sensitive**.

            Some convenient options from ``SBDB_FIELDS`` are available ::

              * `"*"`: Literally all fields
              * `"all"`: `"*"` without the `ignore` columns.
              *  `"ignore"`: Columns that are extremely unlikely to be used in
                 practice.
              *  `"simple"`: Columns that are most likely to be used in
                 practice.
              *  `″simple_ast"`: `"simple"` without the comet-related columns.
              *  `″simple_com"`: `"simple"` without the asteroid-related
                 columns.
              *  `"all_ast"`: `"all"` without the comet-related columns.
              *  `"all_com"`: `"all"` without the asteroid-related columns.

            Default is ``"pdes"``.

        sort : str, optional
            Sort results by the specified field(s). Up to three fields can be
            specified, separated by commas (``,``) and descending direction can
            be specified by prefixing the field name with minus (``-``)
            (ascending is the default).

        limit : int, optional
            Limit data to the first `limit` results (where `limit` is the
            specified number and must be an integer value greater than zero).
            Default is `100`, so if you want all results, set `limit` to
            `None`.

        limit_from : int, optional
            Limit data starting from the specified record (where zero is the
            first record). Useful for “paging” through large datasets. Requires
            `limit`. **CAUTION**: it is possible for the underlying database to
            change between API calls.

        full_prec : bool, int optional
            Output data in full precision (normally, data are returned in
            reduced precision for display purposes). Default is `False`.

        sb_ns : str, optional
            Numbered status: restrict query results to either numbered
            (``"n"``) or unnumbered (``"u"``) small-bodies.

        sb_kind : str, optional
            Limit results to either asteroids-only (``"a"``) or comets-only
            (``"c"``).

        sb_group : str, optional
            Limit results to NEOs-only (``"neo"``) or PHAs-only (``"pha"``).

        sb_class : str (comma-separated) or list of str, optional
            Limit results to small-bodies with orbits of the specified class
            (or classes). Allowable values are valid 3-character orbit-class
            codes (see section below on orbit classes). If specifying more than
            one class, separate entities with a comma (e.g., ``"TJN,CEN"``) or
            provide a list of str (e.g., ``["TJN", "CEN"]``). **Codes are
            case-sensitive.**
            See "Available SBDB Orbit Classes" section at the link for details:
            https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html

        sb_sat : bool, optional
            Limit results to small-bodies with at least one known satellite.

        sb_xfrag : bool, optional
            Exclude all comet fragments (if any) from results.

        sb_defs : {"class", "field", "all"}, optional
            Return SBDB definitions and data related to available filter
            constraints. These data are typically only useful in supporting
            webpage apps. See "mode ``I``" section in the link for details.
            https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html
            If provided, mode ``I`` will be used, and all other parameters
            except `info` will completely be ignored.

        sb_cdata : str, optional
            Custom field constraints (``"sb-cdata"`` field). Maximum length is
            2048 characters when converted to the URI encoded string.
            See this link for details:
            https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html#constraints


        Notes
        -----
        This will get matured/generated in the future similar to
        `astroquery.jplsbdb`. This is at a primitive stage to query all
        information specifically for "all objects"
        """
        self.base_url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api?"

        params = {}

        infomode = info is not None or sb_defs is not None

        if infomode:
            # Use info mode, ignore any other things
            if info not in ["count", "field", "all"]:
                raise ValueError("`info` must be 'count', 'field', or 'all'")
            if sb_defs not in ["class", "field", "all"]:
                raise ValueError("`sb_defs` must be 'class', 'field', or 'all'")
            params["info"] = info
            params["sb-defs"] = sb_defs

        else:
            try:
                self.fields = list(SBDB_FIELDS[fields].keys())
                params["fields"] = ",".join(self.fields)
                if isinstance(fields, str):
                    if fields.endswith("ast"):
                        sb_kind = "a"
                    elif fields.endswith("com"):
                        sb_kind = "c"
            except (TypeError, KeyError):
                if not isinstance(fields, str):  # if list
                    try:
                        self.fields = fields
                        fields = ",".join(fields)
                    except TypeError:
                        raise TypeError(
                            f"`fields` must be str or list of str, got {type(fields)}"
                        )
                params["fields"] = fields
                self.fields = fields.split(",")

            param_specs = {
                "sort": (sort, str),
                "limit": (limit, int),
                "limit-from": (limit_from, int),
                "sb-ns": (sb_ns, lambda x: x in ["n", "u"]),
                "sb-kind": (sb_kind, lambda x: x in ["a", "c"]),
                "sb-group": (sb_group, lambda x: x in ["neo", "pha"]),
                "sb-defs": (sb_defs, str),
                "sb-cdata": (sb_cdata, str),
                "sb-sat": (sb_sat, bool),
                "sb-xfrag": (sb_xfrag, bool),
            }
            # Validate and assign
            for key, (val, typ_or_validator) in param_specs.items():
                if val is not None:
                    if callable(typ_or_validator):
                        if not typ_or_validator(val):
                            raise ValueError(f"`{key}` has invalid value: {val}")
                    elif not isinstance(val, typ_or_validator):
                        raise TypeError(f"`{key}` must be {typ_or_validator.__name__}")
                    params[key] = int(val) if isinstance(val, bool) else val

            if sb_class is not None:
                if isinstance(sb_class, str):
                    _classes = sb_class.split(",")
                else:
                    _classes = sb_class
                    sb_class = ",".join(sb_class)

                for _class in _classes:
                    if _class not in SBDB_ALLOWED_SBCLASS:
                        raise ValueError(
                            f"The element {_class} in `sb_class` is not in {SBDB_ALLOWED_SBCLASS}."
                        )

                params["sb-class"] = sb_class

            params["full-prec"] = int(full_prec)

        self._params = params

    def query(self):
        """Query SBDB."""
        data = _query_cached(self.base_url, tuple(sorted(self._params.items())))

        if (ver := data["signature"]["version"]) != "1.0":
            warn(f"Only ver 1.0 is guaranteed but got {ver}")

        failed_cols = []
        try:
            self.orb = pd.DataFrame(data["data"], columns=data["fields"])
            # For safety, convert all columns to the type in SBDB_FIELDS
            for c in self.orb.columns:
                self.orb[c] = self.orb[c].astype(SBDB_FIELDS["*"][c])

        except Exception:
            failed_cols.append(c)

        if failed_cols:
            # The user should not see this warning unless there is a bug in the code.
            warn(f"Failed to convert columns {failed_cols} to SBDB_FIELDS types.")

        return self.orb


def sanitize_sbdb(orb, drop_unreliable=True, drop_impacted=True):
    """Drop unreliable or impacted objects from the SBDB orbit table.

    Parameters
    ----------
    orb : pandas.DataFrame
        The SBDB orbit table to be sanitized.

    drop_impacted : bool, optional
        If `True`, drop impacted objects based on `IMPACTED` list.
        Default is `True`.

    drop_unreliable : bool, optional
        If `True`, drop unreliable objects based on the following criteria ..::

          - no magnitude-related information
          - prefix is "D" or "X" (disappeared or lost comets)
          - solution date is NaN/"None" (no reliable orbit)
          - two body is "T" (two body assumed orbit - unreliable)

        Default is `True`.
    """
    mask = None
    if drop_unreliable:
        magfields = ["M1", "M2", "K1", "K2", "PC", "H", "G"]
        nocols = []
        for col in magfields + ["prefix", "soln_date", "two_body"]:
            if col not in orb.columns:
                nocols.append(col)

        if nocols:
            warn(f"Fields {nocols} unavailable - skipping drop_unreliable")
        else:
            # absolutely no magnitude-related information
            mask = np.all(
                pd.isna(orb[magfields].apply(pd.to_numeric, errors="coerce")), axis=1
            )
            # other unreliable fields
            mask |= (
                orb["prefix"].isin(["D", "X"])
                | pd.isna(orb["soln_date"])
                | (orb["soln_date"].str.len() < 10)
                | (orb["two_body"] == "T")
            )
    # soln_date len is for those shorter than "yyyy-mm-dd"; ex: "None"
    if drop_impacted:
        if "pdes" not in orb.columns:
            warn(
                "Field 'pdes' unavailable - skipping drop_impacted",
            )
        else:
            _mask = orb["pdes"].isin(IMPACTED)
            mask = _mask if mask is None else mask | _mask

    if mask is not None:
        orb = orb[~mask].reset_index(drop=True)

    return orb
