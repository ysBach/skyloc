"""Query JPL Horizons ephemeris CLI tool.

Queries JPL Horizons via :func:`~skyloc.jplhorizons.horizons_quick`,
saves the result as a Parquet file (optionally compacted), and
optionally plots an interactive HTML.
"""

import sys

import click


@click.command(name="query-horizons")
@click.argument("objid")
@click.option(
    "-o", "--output", required=True, type=click.Path(),
    help="Output Parquet file path.",
)
@click.option(
    "--start", default=None,
    help=(
        "Start time (ISO format, e.g. '2025-01-01'). "
        "If omitted, uses the current time."
    ),
)
@click.option(
    "--stop", default=None,
    help=(
        "Stop time (ISO format, e.g. '2026-01-01'). "
        "If omitted, uses the current time."
    ),
)
@click.option(
    "--step", default="1d",
    help="Step size in Horizons syntax, e.g. '1d', '6h', '30m'. Default: '1d'.",
)
@click.option(
    "--epochs", "epoch_list", default=None,
    help=(
        "Comma-separated JD values to query at specific epochs instead of "
        "start/stop/step. Example: '2460600.5,2460601.5'."
    ),
)
@click.option(
    "--location", default="500@399",
    help="Observer location code. Default: '500@399' (geocenter, Earth).",
)
@click.option(
    "--in-tdb", is_flag=True, default=False,
    help=(
        "Interpret epoch times as TDB (Barycentric Dynamical Time). "
        "Default is False, i.e., epochs are in UTC."
    ),
)
@click.option(
    "--compact/--no-compact", default=True,
    help=(
        "Compact the output Parquet (scaled int + HEALPix). "
        "Default: --compact."
    ),
)
@click.option(
    "--plot", "plot_path", default=None, type=click.Path(),
    help="If given, also save an interactive HTML plot to this path.",
)
def query_horizons_cli(
    objid, output, start, stop, step, epoch_list,
    location, in_tdb, compact, plot_path,
):
    """Query JPL Horizons ephemeris and save to Parquet.

    OBJID: Target object identifier (name, number, or designation).

    Examples::

        query-horizons "Ceres" -o ceres.parq --start 2025-01-01 --stop 2026-01-01

        query-horizons "3200" -o eph.parq --start 2025-01-01 --stop 2026-01-01 --step 1d --location "500@399"

        query-horizons "3200" -o eph.parq --start 2025-01-01 --stop 2026-01-01 --plot 3200.html

        query-horizons "Ceres" -o ceres.parq --epochs "2460600.5,2460601.5" --in-tdb
    """
    from skyloc.jplhorizons import horizons_quick

    # ---- build epochs argument ----
    if epoch_list is not None:
        # Discrete JD list mode
        try:
            epochs = [float(x.strip()) for x in epoch_list.split(",")]
        except ValueError as exc:
            click.echo(f"Error: cannot parse --epochs: {exc}", err=True)
            sys.exit(1)
    else:
        # Range mode: build dict for astroquery
        if start is None or stop is None:
            from datetime import datetime, timezone

            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            if start is None:
                start = now_iso
            if stop is None:
                stop = now_iso
        epochs = {"start": start, "stop": stop, "step": step}

    # ---- query ----
    try:
        eph_raw, eph_df = horizons_quick(
            objid=objid,
            epochs=epochs,
            location=location,
            in_tdb=in_tdb,
            auto_choose_recordnum=True,
        )
    except Exception as exc:
        click.echo(f"Error querying Horizons: {exc}", err=True)
        sys.exit(1)

    click.echo(
        f"Queried {len(eph_df)} epochs for '{objid}' @ {location}", err=True
    )

    # ---- extract extra columns from raw Horizons output ----
    from astropy import units as u

    _extra_cols = {
        "elong": ("elong", u.deg),
        "lunar_elong": ("lunar_elong", u.deg),
        "gal_lat": ("GlxLat", u.deg),
    }
    for dest, (src, unit) in _extra_cols.items():
        if src in eph_raw.colnames:
            try:
                eph_df[dest] = eph_raw[src].to(unit).value
            except Exception:
                pass  # column may be masked or missing values

    # ---- optionally compact ----
    if compact:
        from skyloc.ioutils.ephemeris import compact_ephem_parq_cols, EPH_DTYPES_BASE

        dtypes = EPH_DTYPES_BASE | {
            "elong": (360, "uint16", 65535, "float64"),
            "lunar_elong": (360, "uint16", 65535, "float64"),
            "gal_lat": (180, "int16", 32767, "float64"),
        }
        eph_df = compact_ephem_parq_cols(eph_df, dtypes=dtypes)
        click.echo("Compacted columns for storage.", err=True)

    # ---- save parquet ----
    eph_df.to_parquet(output, engine="auto", compression="snappy")
    click.echo(f"Saved {output}  ({len(eph_df)} rows)", err=True)

    # ---- optionally plot ----
    if plot_path is not None:
        try:
            _plot_from_df(eph_raw, eph_df, plot_path, compact=compact)
            click.echo(f"Saved interactive HTML -> {plot_path}", err=True)
        except ImportError:
            click.echo(
                "Error: plotly is required for --plot. "
                "Install with: pip install plotly",
                err=True,
            )
            sys.exit(1)


# ---- Default plot configuration ----
# Matches the typical ephemeris workflow (initial_plot_kwargs pattern).


def _default_axis_configs(df):
    """Build default axis_configs from available columns in *df*.

    Returns a dict suitable for ``plot_ephemeris_plotly(axis_configs=...)``.
    Only includes axes whose columns exist in *df*.
    """
    configs = {}

    # Magnitude (left axis)
    # Prefer Tmag for comets (if vmag is absent), otherwise vmag
    if "Tmag" in df.columns and "vmag" not in df.columns:
        mag_col, mag_name = "Tmag", "Tmag"
    elif "vmag" in df.columns:
        mag_col, mag_name = "vmag", "Vmag"
    else:
        mag_col = None
    if mag_col is not None:
        configs["mag"] = dict(
            cols=(mag_col,),
            side="left",
            title="mag",
            color="red",
            lw=2,
            dash="solid",
            names=(mag_name,),
            required=True,
            invert=True,
            showgrid=True,
        )

    # Sky motion (left axis)
    if "sky_motion" in df.columns:
        configs["sky"] = dict(
            cols=("sky_motion",),
            side="left",
            title="Sky motion [arcsec/min]",
            color="#00CC96",
            lw=2,
            dash="solid",
            names=("sky_motion",),
            required=False,
            showgrid=False,
            unit="arcsec/min",
            initially_hidden=False,
        )

    # Angles (right axis)
    angle_cols = tuple(c for c in ("alpha", "elong", "lunar_elong") if c in df.columns)
    if angle_cols:
        n = len(angle_cols)
        configs["angle"] = dict(
            cols=angle_cols,
            side="right",
            title="Angle [deg]",
            range=(0, 150),
            color="#AB63FA",
            lw=tuple([2, 1, 2][:n]),
            dash=tuple(["solid", "solid", "dot"][:n]),
            names=angle_cols,
            unit="deg",
            initially_hidden=tuple(
                [False, True, True][:n]
            ),
        )

    # Distances (right axis)
    dist_cols = tuple(c for c in ("r_hel", "r_obs") if c in df.columns)
    if dist_cols:
        n = len(dist_cols)
        configs["dist"] = dict(
            cols=dist_cols,
            side="right",
            title="Distance [au]",
            color="blue",
            lw=2,
            dash=tuple(["solid", "dot"][:n]),
            names=dist_cols,
            unit="au",
        )

    # Latitudes (right axis)
    lat_cols = tuple(c for c in ("dec", "gal_lat") if c in df.columns)
    if lat_cols:
        n = len(lat_cols)
        configs["lat"] = dict(
            cols=lat_cols,
            side="right",
            title="Latitude [deg]",
            range=(-90, 90),
            color="#FFA15A",
            lw=tuple([1, 1][:n]),
            dash=tuple(["dot", "solid"][:n]),
            names=tuple(["Dec", "Gal.Lat."][:n]),
            unit="deg",
        )

    return configs


def _default_shade_configs(df):
    """Build default shade_configs from available columns in *df*."""
    configs = {}
    if "elong" in df.columns:
        configs["elong_window"] = dict(
            col="elong",
            range=(80, 120),
            fillcolor="rgba(128,128,128,0.18)",
            line_width=0,
            layer="below",
            xref="x",
            yref="paper",
            y0=0.0,
            y1=1.0,
            initially_hidden=True,
        )
    return configs


_DEFAULT_LAYOUT_KWARGS = dict(
    use_webgl=True,
    show_xgrid=False,
    xgrid_month_step=6,
    domain_right=0.8,
    domain_left=0.08,
    left_positions=(0, 0.07),
    left_margin=20,
    right_positions=(0.83, 0.91, 0.99),
    right_margin=40,
    legend_fontsize=16,
    tickfont_size=18,
    title_standoff=8,
    fig_width=1200,
    fig_height=800,
)


def _plot_from_df(eph_raw, eph_df, plot_path, *, compact):
    """Generate default interactive HTML plot from ephemeris DataFrame."""
    from skyloc.plotutils.plot_ephem import plot_ephemeris_plotly
    from skyloc.ioutils.ephemeris import EPH_DTYPES_BASE

    # Decompress if compact
    if compact:
        from skyloc.ioutils.ephemeris import parse_compact_ephem

        dtypes = EPH_DTYPES_BASE | {
            "elong": (360, "uint16", 65535, "float64"),
            "lunar_elong": (360, "uint16", 65535, "float64"),
            "gal_lat": (180, "int16", 32767, "float64"),
        }
        df = parse_compact_ephem(eph_df, dtypes=dtypes)
    else:
        df = eph_df

    # Build times
    from astropy.time import Time

    if "jd_utc" in df.columns:
        times = Time(df["jd_utc"].to_numpy(), format="jd", scale="utc")
    elif "datetime_jd" in df.columns:
        times = Time(df["datetime_jd"].to_numpy(), format="jd", scale="utc")
    else:
        import numpy as np
        times = np.arange(len(df))

    axis_configs = _default_axis_configs(df)
    shade_configs = _default_shade_configs(df)

    if not axis_configs:
        import numpy as np
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                axis_configs[col] = dict(
                    cols=(col,),
                    side="left" if not axis_configs else "right",
                    title=col,
                )
                if len(axis_configs) >= 5:
                    break

    plot_ephemeris_plotly(
        times=times,
        dataframe=df,
        axis_configs=axis_configs,
        shade_configs=shade_configs if shade_configs else None,
        output_html=plot_path,
        **_DEFAULT_LAYOUT_KWARGS,
    )


if __name__ == "__main__":
    query_horizons_cli()
