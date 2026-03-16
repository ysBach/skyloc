"""Command-line interface entry points for skyloc.

Provides Click-based CLI tools registered as console scripts:

- ``query-horizons`` — query JPL Horizons and save ephemerides.
- ``filter-ephem`` — filter compacted ephemeris Parquet files.
- ``ephem2plotly`` — render ephemeris data as interactive HTML plots.
"""

from .filter_ephem import filter_ephem_cli
from .query_horizons import query_horizons_cli
from .plot_ephem import ephem2plotly_cli

__all__ = ["filter_ephem_cli", "query_horizons_cli", "ephem2plotly_cli"]
