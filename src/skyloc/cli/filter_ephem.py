"""Filter ephemeris CLI tool."""

import sys

import click

from ..ioutils.ephemeris import filter_ephem, EPH_DTYPES_BASE
from ..ioutils._filter_utils import to_numeric_if_possible


STRING_COLUMNS = ["desig"]


def parse_filter_value(ctx, param, value):
    """Click callback to parse filter value string.

    Supports:
    - Comparison: ">30", "<=180", "==5"
    - Comma-separated for 'in': "1,2,3"
    - Plain values for equality
    """
    if value is None:
        return None

    value_str = value.strip()

    # Determine if we should convert to numeric
    # param.name corresponds to the option name (e.g., 'desig')
    convert = True
    if param and param.name in STRING_COLUMNS:
        convert = False

    converter = to_numeric_if_possible if convert else lambda x: x

    for op in [">=", "<=", "!=", "==", ">", "<"]:
        if value_str.startswith(op):
            val_part = value_str[len(op) :].strip()
            return (op, converter(val_part))

    if "," in value_str:
        parts = [p.strip() for p in value_str.split(",")]
        values = [converter(p) for p in parts]
        return ("in", values)

    return converter(value_str)


KNOWN_FILTER_COLUMNS = list(EPH_DTYPES_BASE.keys()) + [
    "desig",
    "jd",
    "ra",
    "dec",
    "hel_ecl_lon",
    "hel_ecl_lat",
    "obs_ecl_lon",
    "obs_ecl_lat",
]

COLUMN_TO_OPTION = {}
for col in KNOWN_FILTER_COLUMNS:
    option_name = col.replace("*", "_").replace("/", "_")
    COLUMN_TO_OPTION[option_name] = col


@click.command(name="filter-ephem")
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=False))
@click.option(
    "-o", "--output", required=True, type=click.Path(), help="Output file path (parquet or csv)"
)
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it exists")
@click.option("--columns", help="Comma-separated list of columns to include in output")
@click.option(
    "--filter-per-file/--no-filter-per-file",
    default=True,
    help="Filter each file separately (default) vs load all first",
)
def filter_ephem_cli(
    files, output, overwrite, columns, filter_per_file, **filter_kwargs
):
    """Filter ephemeris parquet files (output to .parq or .csv).

    FILES: Parquet files or glob patterns to filter

    Examples:

      # Filter by magnitude
      filter-ephem data/*.parquet --vmag "<20" -o filtered.parquet

      # Filter by designation (multiple values)
      filter-ephem data/*.parquet --desig "1,2,3" -o filtered.parquet

      # Multiple filters
      filter-ephem data/*.parquet --vmag "<20" --dec ">30" -o filtered.parquet

      # Select specific columns
      filter-ephem data/*.parquet --vmag "<20" --columns desig,ra,dec,vmag -o out.parquet
    """
    columns_list = None
    if columns:
        columns_list = [c.strip() for c in columns.split(",")]

    filelist = files[0] if len(files) == 1 else list(files)

    filter_kwargs_actual = {}
    for opt_name, col_name in COLUMN_TO_OPTION.items():
        if opt_name in filter_kwargs and filter_kwargs[opt_name] is not None:
            filter_kwargs_actual[col_name] = filter_kwargs[opt_name]

    try:
        result = filter_ephem(
            filelist=filelist,
            filter_per_file=filter_per_file,
            output=output,
            overwrite=overwrite,
            columns=columns_list,
            **filter_kwargs_actual,
        )

        click.echo(f"Filtered {len(result)} rows from input files")
        click.echo(f"Output saved to: {output}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except FileExistsError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error filtering ephemeris: {e}", err=True)
        sys.exit(1)


for opt_name, col_name in COLUMN_TO_OPTION.items():
    filter_ephem_cli = click.option(
        f"--{opt_name}",
        callback=parse_filter_value,
        help=f'Filter {col_name} (e.g., ">30", "1,2,3")',
    )(filter_ephem_cli)


if __name__ == "__main__":
    filter_ephem_cli()
