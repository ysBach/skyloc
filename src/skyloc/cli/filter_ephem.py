"""Filter ephemeris CLI tool.

Uses ``pqfilt`` expression syntax for generic filter parsing,
then delegates to :func:`~skyloc.ioutils.ephemeris.filter_ephem`
for domain-specific decompression and filtering.
"""

import sys

import click

from ..ioutils.ephemeris import filter_ephem
from pqfilt._parser import parse_expression, FilterExpr, AndExpr, OrExpr


def _ast_to_filter_kwargs(node):
    """Convert a parsed pqfilt AST to filter_ephem **kwargs.

    Only flat AND of FilterExpr nodes can be converted to kwargs.
    For complex OR expressions, raises an error directing the user
    to use the Python API.

    Parameters
    ----------
    node : FilterExpr, AndExpr, or OrExpr
        Parsed expression from pqfilt.

    Returns
    -------
    dict
        Keyword arguments for ``filter_ephem()``.

    Raises
    ------
    ValueError
        If the expression contains OR logic (not supported by filter_ephem kwargs).
    """
    if isinstance(node, FilterExpr):
        return {node.col: (node.op, node.val)}

    if isinstance(node, AndExpr):
        kwargs = {}
        for child in node.children:
            if isinstance(child, FilterExpr):
                kwargs[child.col] = (child.op, child.val)
            else:
                raise ValueError(
                    "filter-ephem CLI only supports AND-combined simple filters. "
                    "For OR logic, use the Python API: "
                    "filter_ephem(files, ...) with pqfilt expressions."
                )
        return kwargs

    if isinstance(node, OrExpr):
        raise ValueError(
            "filter-ephem CLI does not support OR expressions. "
            "For OR logic, use the Python API."
        )

    raise TypeError(f"Unexpected node type: {type(node)}")


@click.command(name="filter-ephem")
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=False))
@click.option(
    "-f",
    "--filter",
    "filter_exprs",
    multiple=True,
    help=(
        'Filter expression, e.g. "vmag < 20", "desig in 1,2,3". '
        "Multiple -f flags are AND-ed together."
    ),
)
@click.option(
    "-o", "--output", required=True, type=click.Path(),
    help="Output file path (.parquet or .csv)",
)
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it exists")
@click.option("--columns", help="Comma-separated list of columns to include in output")
@click.option(
    "--per-file/--no-per-file",
    default=True,
    help="Filter each file separately (default) vs load all first",
)
def filter_ephem_cli(files, filter_exprs, output, overwrite, columns, per_file):
    """Filter ephemeris parquet files (output to .parq or .csv).

    FILES: Parquet files or glob patterns to filter.

    Uses pqfilt expression syntax for filters.

    Examples::

        filter-ephem data/*.parquet -f "vmag < 20" -o filtered.parquet
        filter-ephem data/*.parquet -f "vmag < 20" -f "dec > 30" -o out.parquet
        filter-ephem data/*.parquet -f "desig in 1,2,3" -o filtered.parquet
        filter-ephem data/*.parquet -f "vmag < 20" --columns desig,ra,dec,vmag -o out.parquet
    """
    columns_list = None
    if columns:
        columns_list = [c.strip() for c in columns.split(",")]

    filelist = files[0] if len(files) == 1 else list(files)

    # Parse and combine filter expressions (AND)
    filter_kwargs = {}
    if filter_exprs:
        if len(filter_exprs) == 1:
            combined = filter_exprs[0]
        else:
            combined = " & ".join(f"({expr})" for expr in filter_exprs)

        try:
            ast = parse_expression(combined)
            filter_kwargs = _ast_to_filter_kwargs(ast)
        except ValueError as e:
            click.echo(f"Filter error: {e}", err=True)
            sys.exit(1)

    try:
        result = filter_ephem(
            filelist=filelist,
            filter_per_file=per_file,
            output=output,
            overwrite=overwrite,
            columns=columns_list,
            **filter_kwargs,
        )

        click.echo(f"Filtered {len(result)} rows -> {output}", err=True)

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except FileExistsError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error filtering ephemeris: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    filter_ephem_cli()
