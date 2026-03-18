Command-Line Tools
==================

skyloc provides three CLI tools registered as console-script entry points.

.. contents:: Tools
   :local:
   :depth: 1

query-horizons
--------------

Query JPL Horizons for a target and save the ephemeris as a Parquet file.

.. code-block:: bash

   # Basic usage: Ceres ephemeris for one year
   query-horizons "Ceres" -o ceres.parq \
       --start 2025-01-01 --stop 2026-01-01

   # With interactive HTML plot
   query-horizons "3200" -o phaethon.parq \
       --start 2025-01-01 --stop 2026-01-01 \
       --plot phaethon.html

   # Discrete epochs (TDB)
   query-horizons "Ceres" -o ceres.parq \
       --epochs "2460600.5,2460601.5" --in-tdb

Options
^^^^^^^

``--start``
   Start time (ISO format). Defaults to now.

``--stop``
   Stop time (ISO format). Defaults to now.

``--step``
   Step size (Horizons syntax, e.g. ``1d``, ``6h``). Default: ``1d``.

``--epochs``
   Comma-separated JD values for discrete-epoch mode.

``--location``
   Observer location code. Default: ``500@399`` (geocentre).

``--compact / --no-compact``
   Compact output using scaled integers + HEALPix. Default: ``--compact``.

``--plot PATH``
   Also save an interactive HTML plot.

``--in-tdb``
   Interpret epoch times as TDB instead of UTC.

filter-ephem
------------

Filter compacted ephemeris Parquet files using ``pqfilt`` expression syntax.

.. code-block:: bash

   # Filter by magnitude
   filter-ephem data/*.parquet -f "vmag < 20" -o filtered.parquet

   # Multiple AND-ed filters
   filter-ephem data/*.parquet \
       -f "vmag < 20" -f "dec > 30" -o out.parquet

   # Select specific columns
   filter-ephem data/*.parquet \
       -f "vmag < 20" --columns desig,ra,dec,vmag -o out.parquet

Options
^^^^^^^

``-f / --filter EXPR``
   Filter expression (repeatable; multiple are AND-combined).

``-o / --output PATH``
   Output file path (required).

``--overwrite``
   Overwrite output if it already exists.

``--columns COLS``
   Comma-separated list of columns to include.

``--per-file / --no-per-file``
   Filter each file separately (default) or load all first.

ephem2plotly
------------

Render an ephemeris Parquet file as an interactive HTML plot.

.. code-block:: bash

   ephem2plotly eph.parq -o eph.html
   ephem2plotly eph.parq -o eph.html --width 1600 --height 900
   ephem2plotly eph.parq -o eph.html --title "3200 Phaethon"

Options
^^^^^^^

``-o / --output PATH``
   Output HTML file path (required).

``--width / --height``
   Figure dimensions in pixels.

``--title TEXT``
   HTML document title.

``--no-webgl``
   Use standard SVG traces instead of WebGL.
