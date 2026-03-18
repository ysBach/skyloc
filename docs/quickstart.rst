Quickstart
==========

This page provides a brief overview of the most common skyloc workflows.

Locating objects in a field of view
-----------------------------------

The core workflow uses :class:`~skyloc.core.Locator` to find Solar System
objects that fall within an image's field of view:

.. code-block:: python

   from skyloc import Locator

   loc = Locator(
       orb="orb_sbdb.parq",        # orbital-element Parquet file
       jd_tdb=2460600.5,           # epoch (TDB Julian date)
       fov_ra=180.0,               # FOV centre RA [deg]
       fov_dec=30.0,               # FOV centre Dec [deg]
       fov_radius=0.3,             # search radius [deg]
   )
   eph = loc.run()                 # returns a DataFrame of matched objects

Querying JPL SBDB
-----------------

Fetch orbital elements for small bodies using
:func:`~skyloc.query.fetch_orb`:

.. code-block:: python

   from skyloc import fetch_orb

   orb = fetch_orb(output="orb_sbdb.parq")

Querying JPL Horizons
---------------------

Generate ephemerides via the Horizons system:

.. code-block:: python

   from skyloc.jplhorizons import horizons_quick

   eph_raw, eph_df = horizons_quick(
       objid="Ceres",
       epochs={"start": "2025-01-01", "stop": "2026-01-01", "step": "1d"},
       location="500@399",
   )

Compacting ephemeris files
--------------------------

Store ephemeris DataFrames efficiently using scaled-integer Parquet
compression:

.. code-block:: python

   from skyloc.ioutils.ephemeris import compact_ephem_parq_cols

   compact_df = compact_ephem_parq_cols(eph_df)
   compact_df.to_parquet("eph_compact.parq")

Fast WCS transformations
-------------------------

Use :class:`~skyloc.ioutils.wcs.FastTanSipWCS` for Numba-accelerated
WCS when processing many headers:

.. code-block:: python

   from skyloc.ioutils import FastTanSipWCS

   wcs = FastTanSipWCS(header_dict)
   ra, dec = wcs.all_pix2world(x, y, 0)
