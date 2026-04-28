"""Calculate local ephemerides from queried orbital elements."""

import sys

import click
import numpy as np

from ._ephem_common import epoch_jds_from_range
from .query_horizons import _plot_from_df
from skyloc.keteutils._kete_import import kete, require_kete
from skyloc.keteutils.spice import is_spk_loaded
from skyloc.utils import utc2tdb


def _parse_objids(objid):
    objids = [x.strip() for x in objid.split(",") if x.strip()]
    if not objids:
        raise ValueError("OBJID must contain at least one object identifier.")
    return objids


def _build_epoch_jds_tdb(start, stop, step, epoch_list, *, in_tdb):
    if epoch_list is not None:
        values = [float(x.strip()) for x in epoch_list.split(",")]
        return np.asarray(values) if in_tdb else utc2tdb(values).jd

    if start is None or stop is None:
        from datetime import datetime, timezone

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        start = now_iso if start is None else start
        stop = now_iso if stop is None else stop

    _, epoch_jds = epoch_jds_from_range(start, stop, step, in_tdb=in_tdb)
    return np.asarray(epoch_jds) if in_tdb else utc2tdb(epoch_jds).jd


def _observer_state(location, jd_tdb, index):
    state_desig = f"{location}:{index}"
    jd_tdb = float(jd_tdb)

    is_loaded, spice_name = is_spk_loaded(location, convert_to="name")
    if is_loaded:
        state = kete.spice.get_state(spice_name, jd_tdb, "sun", kete.Frames.Ecliptic)
        is_spice_location = True
    elif len(location) == 3:
        state = kete.spice.mpc_code_to_ecliptic(location, jd_tdb)
        is_spice_location = False
    else:
        raise ValueError(
            f"Unsupported location '{location}'. Use a SPICE-resolvable target/name "
            "or a three-character MPC observatory code."
        )

    observer = kete.State(
        desig=state_desig,
        jd=jd_tdb,
        pos=state.pos,
        vel=state.vel,
        center_id=state.center_id,
    )
    return observer, is_spice_location


def _make_all_sky_fovs(location, epoch_jds_tdb):
    require_kete()
    pointing = kete.Vector.from_lat_lon(0.0, 0.0)
    fovs = []
    for i, jd_tdb in enumerate(epoch_jds_tdb):
        observer, is_spice_location = _observer_state(location, jd_tdb, i)
        if is_spice_location:
            fov = kete.fov.OmniDirectionalFOV(observer=observer)
        else:
            fov = kete.fov.ConeFOV(
                observer=observer,
                pointing=pointing,
                angle=180.0,
            )
        fovs.append(fov)
    return fovs


def _calculate_local_ephemeris(
    objids,
    epoch_jds_tdb,
    location,
    *,
    include_asteroids=True,
    non_gravs=True,
):
    from skyloc.core import SSOLocator
    from skyloc.query import query_orb_by_desig

    orb = query_orb_by_desig(objids, strict=True)
    if orb.empty:
        raise ValueError("No orbital elements were found for the requested object(s).")

    epoch_jds_tdb = np.asarray(epoch_jds_tdb)
    fovs = _make_all_sky_fovs(location, epoch_jds_tdb)
    jd0 = float(np.mean(epoch_jds_tdb))
    dt_limit = float(np.max(np.abs(epoch_jds_tdb - jd0))) + 1.0e-9

    loc = SSOLocator(
        fovs=fovs,
        orb=orb,
        non_gravs=non_gravs,
        drop_major_asteroids=False,
    )
    loc.propagate_n_body(
        jd0=jd0,
        suppress_errors=True,
        include_asteroids=include_asteroids,
    )
    loc.fov_state_check(
        dt_limit=dt_limit,
        include_asteroids=include_asteroids,
        use_spice_for_loaded=False,
    )
    loc.calc_ephems(compact=False, add_obsid=False, drop_obsindex=True, add_jds=True)

    if loc.eph is None or len(loc.eph) == 0:
        raise ValueError("No ephemeris rows were generated.")

    eph = loc.eph
    sort_cols = [c for c in ("jd_tdb", "desig") if c in eph.columns]
    if sort_cols:
        eph = eph.sort_values(sort_cols).reset_index(drop=True)
    return eph


@click.command(name="calc-ephem")
@click.argument("objid")
@click.option(
    "-o", "--output", required=True, type=click.Path(),
    help="Output Parquet file path.",
)
@click.option(
    "--start",
    default=None,
    help=(
        "Start time (ISO format, e.g. '2025-01-01'). "
        "If omitted, uses the current time."
    ),
)
@click.option(
    "--stop",
    default=None,
    help=(
        "Stop time (ISO format, e.g. '2026-01-01'). "
        "If omitted, uses the current time."
    ),
)
@click.option(
    "--step",
    default="1d",
    help="Step size in fixed Horizons syntax, e.g. '1d', '6h', '30m'. Default: '1d'.",
)
@click.option(
    "--epochs",
    "epoch_list",
    default=None,
    help=(
        "Comma-separated JD values to calculate at specific epochs instead of "
        "start/stop/step. Example: '2460600.5,2460601.5'."
    ),
)
@click.option(
    "--location",
    default="500@399",
    help=(
        "Observer location. Supports SPICE-resolvable names/codes and "
        "three-character MPC observatory codes. Default: '500@399'."
    ),
)
@click.option(
    "--in-tdb",
    is_flag=True,
    default=False,
    help=(
        "Interpret epoch times as TDB (Barycentric Dynamical Time). "
        "Default is False, i.e., epochs are in UTC."
    ),
)
@click.option(
    "--compact/--no-compact",
    default=True,
    help=(
        "Compact the output Parquet (scaled int + HEALPix). "
        "Default: --compact."
    ),
)
@click.option(
    "--plot",
    "plot_path",
    default=None,
    type=click.Path(),
    help="If given, also save an interactive HTML plot to this path.",
)
@click.option(
    "--include-asteroids/--no-include-asteroids",
    default=True,
    show_default=True,
    help="Include massive asteroid perturbers in local propagation.",
)
@click.option(
    "--use-nongravs/--no-use-nongravs",
    "non_gravs",
    default=True,
    show_default=True,
    help="Use non-gravitational terms when available in the queried orbit data.",
)
def calc_ephem_cli(
    objid,
    output,
    start,
    stop,
    step,
    epoch_list,
    location,
    in_tdb,
    compact,
    plot_path,
    include_asteroids,
    non_gravs,
):
    """Calculate local ephemerides and save to Parquet.

    OBJID is a target object identifier, or comma-separated identifiers.
    """
    try:
        objids = _parse_objids(objid)
        epoch_jds_tdb = _build_epoch_jds_tdb(
            start,
            stop,
            step,
            epoch_list,
            in_tdb=in_tdb,
        )
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(
        f"Calculating {len(epoch_jds_tdb)} epochs for {len(objids)} object(s) @ {location}.",
        err=True,
    )

    try:
        eph_df = _calculate_local_ephemeris(
            objids,
            epoch_jds_tdb,
            location,
            include_asteroids=include_asteroids,
            non_gravs=non_gravs,
        )
    except Exception as exc:
        click.echo(f"Error calculating ephemeris: {exc}", err=True)
        sys.exit(1)

    if compact:
        from skyloc.ioutils.ephemeris import EPH_DTYPES_BASE, compact_ephem_parq_cols

        eph_df = compact_ephem_parq_cols(eph_df, dtypes=EPH_DTYPES_BASE)
        click.echo("Compacted columns for storage.", err=True)

    eph_df.to_parquet(output, engine="auto", compression="snappy")
    click.echo(f"Saved {output}  ({len(eph_df)} rows)", err=True)

    if plot_path is not None:
        try:
            _plot_from_df(None, eph_df, plot_path, compact=compact)
            click.echo(f"Saved interactive HTML -> {plot_path}", err=True)
        except ImportError:
            click.echo(
                "Error: plotly is required for --plot. "
                "Install with: pip install plotly",
                err=True,
            )
            sys.exit(1)


if __name__ == "__main__":
    calc_ephem_cli()
