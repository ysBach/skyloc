"""Plot ephemeris parquet file as interactive HTML via Plotly.

Loads a (possibly compacted) ephemeris Parquet file, auto-detects columns,
and produces an interactive HTML using
:func:`~skyloc.plotutils.plot_ephem.plot_ephemeris_plotly`.
"""

import sys

import click


@click.command(name="ephem2plotly")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o", "--output", required=True, type=click.Path(),
    help="Output HTML file path.",
)
@click.option("--width", default=1200, type=int, help="Figure width in pixels.")
@click.option("--height", default=800, type=int, help="Figure height in pixels.")
@click.option("--title", default="Interactive ephemeris", help="HTML document title.")
@click.option(
    "--no-webgl", is_flag=True, default=False,
    help="Disable WebGL (use standard SVG traces instead).",
)
def ephem2plotly_cli(input_file, output, width, height, title, no_webgl):
    """Plot an ephemeris Parquet file as interactive HTML.

    INPUT_FILE: Path to the ephemeris Parquet file (plain or compacted).

    Auto-detects available columns and creates a sensible default multi-axis
    plot. Compacted files (with ``*factor`` or ``_hpidx_`` columns) are
    automatically decompressed.

    Examples::

        ephem2plotly eph_3200.parq -o 3200.html
        ephem2plotly eph_3200.parq -o 3200.html --width 1600 --height 900
        ephem2plotly eph_3200.parq -o 3200.html --title "3200 Phaethon"
    """
    try:
        import numpy as np
        import pandas as pd
        from astropy.time import Time

        from skyloc.ioutils.ephemeris import parse_compact_ephem
        from skyloc.plotutils.plot_ephem import plot_ephemeris_plotly
    except ImportError as exc:
        click.echo(f"Error: missing dependency: {exc}", err=True)
        sys.exit(1)

    # ---- load parquet ----
    df = pd.read_parquet(input_file)

    # ---- detect and decompress compacted columns ----
    has_compacted = any(
        "*" in col or "_hpidx_" in col for col in df.columns
    )
    if has_compacted:
        from skyloc.ioutils.ephemeris import EPH_DTYPES_BASE

        dtypes = EPH_DTYPES_BASE | {
            "elong": (360, "uint16", 65535, "float64"),
            "lunar_elong": (360, "uint16", 65535, "float64"),
            "gal_lat": (180, "int16", 32767, "float64"),
        }
        df = parse_compact_ephem(df, dtypes=dtypes)
        click.echo("Detected compacted format; decompressed.", err=True)

    # ---- build times ----
    if "jd_utc" in df.columns:
        times = Time(df["jd_utc"].to_numpy(), format="jd", scale="utc")
    elif "datetime_jd" in df.columns:
        times = Time(df["datetime_jd"].to_numpy(), format="jd", scale="utc")
    else:
        click.echo(
            "Warning: no 'jd_utc' or 'datetime_jd' column found; "
            "using integer index as x-axis.",
            err=True,
        )
        times = np.arange(len(df))

    # ---- build default axis_configs from available columns ----
    axis_configs = _build_default_axis_configs(df)

    if not axis_configs:
        click.echo("Error: no plottable numeric columns found.", err=True)
        sys.exit(1)

    # ---- plot ----
    try:
        plot_ephemeris_plotly(
            times=times,
            dataframe=df,
            axis_configs=axis_configs,
            use_webgl=not no_webgl,
            fig_width=width,
            fig_height=height,
            output_html=output,
        )
    except Exception as exc:
        click.echo(f"Error generating plot: {exc}", err=True)
        sys.exit(1)

    click.echo(
        f"Saved interactive HTML ({len(df)} points) -> {output}", err=True
    )


def _build_default_axis_configs(df):
    """Build default axis_configs dict from available DataFrame columns.

    Returns a dict suitable for ``plot_ephemeris_plotly(axis_configs=...)``.
    Only includes axes whose columns exist in *df*.
    """
    import numpy as np

    configs = {}

    if "dec" in df.columns:
        configs["dec"] = dict(
            cols=("dec",), side="left", title="Dec [deg]",
            color="#1f77b4",
        )

    if "vmag" in df.columns:
        configs["vmag"] = dict(
            cols=("vmag",), side="right", title="V mag",
            color="#d62728", invert=True,
        )

    r_cols = tuple(c for c in ("r_hel", "r_obs") if c in df.columns)
    if r_cols:
        configs["dist"] = dict(
            cols=r_cols, side="right", title="Distance [AU]",
            color=("#2ca02c", "#ff7f0e")[:len(r_cols)],
            dash=("solid", "dot")[:len(r_cols)],
            names=("r_hel", "r_obs")[:len(r_cols)],
        )

    if "alpha" in df.columns:
        configs["alpha"] = dict(
            cols=("alpha",), side="right", title="Phase angle [deg]",
            color="#9467bd",
        )

    if "ra" in df.columns:
        configs["ra"] = dict(
            cols=("ra",), side="right", title="RA [deg]",
            color="#8c564b",
        )

    rate_cols = tuple(
        c for c in ("racosdec_rate", "dec_rate") if c in df.columns
    )
    if rate_cols:
        configs["rates"] = dict(
            cols=rate_cols, side="right", title="Rate [\"/min]",
            color=("#e377c2", "#bcbd22")[:len(rate_cols)],
            dash=("solid", "dot")[:len(rate_cols)],
            names=rate_cols,
        )

    if "sky_motion" in df.columns:
        configs["sky_motion"] = dict(
            cols=("sky_motion",), side="right", title="Sky motion [\"/min]",
            color="#17becf",
        )

    # Fallback: if nothing matched, grab first few numeric columns
    if not configs:
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                configs[col] = dict(
                    cols=(col,),
                    side="left" if not configs else "right",
                    title=col,
                )
                if len(configs) >= 5:
                    break

    return configs


if __name__ == "__main__":
    ephem2plotly_cli()
