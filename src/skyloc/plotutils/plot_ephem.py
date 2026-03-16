"""Interactive multi-axis ephemeris plotting with Plotly.

Provides :func:`plot_ephemeris_plotly` for creating richly configurable,
multi-y-axis time-series plots of ephemeris data, and
:func:`save_plotly_html` for exporting them as self-contained HTML files
with interactive axis selectors and shade controls.

Requires the optional ``plotly`` dependency (``pip install plotly``).
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

try:
    import plotly
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:
    plotly = None


__all__ = ["plot_ephemeris_plotly", "save_plotly_html"]


def _require_plotly():
    """Raise an informative error when Plotly is not installed."""
    if plotly is None:
        raise ImportError(
            "plotly is required for plotting/export. Install it with "
            "`pip install plotly` or add it to your project dependencies."
        )


def plot_ephemeris_plotly(
    times,
    dataframe,
    axis_configs,
    shade_configs=None,
    use_webgl=True,
    show_xgrid=True,
    xgrid_month_step=6,
    domain_left=0.0,
    domain_right=0.78,
    left_positions=None,
    right_positions=(0.82, 0.89, 0.96),
    left_margin=80,
    right_margin=40,
    tickfont_size=14,
    legend_fontsize=12,
    title_standoff=8,
    fig_width=1200,
    fig_height=800,
    output_html=None,
):
    """
    Create an interactive multi-axis ephemeris plot with dynamic axis selection.

    Parameters
    ----------
    times
        Time array (e.g., ``astropy.time.Time`` or datetime-like sequence).
    dataframe
        Ephemeris table-like object with column access (e.g., Astropy Table or
        dict-like object).
    axis_configs : dict or sequence of dict
        Axis definitions. Each axis accepts keys such as:

        - ``cols`` or ``columns``: tuple/list of column names.
        - ``side``: ``"left"`` or ``"right"`` (default: ``"right"``).
        - ``title`` or ``label``: y-axis title text.
        - ``range`` or ``ylim``: y-axis range tuple.
        - ``color`` or ``colors``: trace color(s), scalar or per-column.
        - ``lw``/``width``/``line_width``: line width(s), scalar or per-column.
        - ``dash`` or ``dashes``: line dash style(s), scalar or per-column.
        - ``names`` or ``trace_names``: trace labels, scalar or per-column.
        - ``unit`` or ``units``: unit label(s) used in default hover templates.
        - ``hovertemplate`` or ``hovertemplates``: custom hover template(s).
        - ``showgrid``: whether to show y-grid for this axis.
        - ``position``: explicit axis position for overlay axes.
        - ``required``: if ``True``, raise when all requested columns are
          missing or non-finite.
        - ``invert``: invert this axis (equivalent to reversed autorange).
        - ``initially_hidden``: if ``True``, traces start as ``legendonly``.
          Can be a single bool (applies to all traces) or a per-column
          sequence of bools.  The axis itself is hidden only when **all** its
          traces are initially hidden.  Users can re-enable them via the
          interactive HTML controls.

        Example
        -------
        ``axis_configs=dict(
            mag=dict(cols=("vmag",), side="left", title="mag", invert=True),
            axis1=dict(
                cols=("sslat", "selat"),
                title="Latitude [deg]",
                color=("#FFA15A", "#FFA15A"),
                lw=(1, 1),
                dash=("solid", "dot"),
                side="right",
            ),
        )``

    shade_configs : dict or sequence of dict, optional
        Shading definitions. Each shade accepts keys such as:

        - ``col``/``column`` or ``cols``/``columns``: one or multiple columns.
        - ``range``/``between`` or ``ranges``: one range for all columns, or
          per-column ranges.
        - ``fillcolor``/``color`` or ``fillcolors``/``colors``.
        - ``opacity``/``alpha`` or ``opacities``/``alphas``.
        - ``y0``/``y1`` or ``y0s``/``y1s``.
        - ``line_width``/``lw``/``width`` and ``line_color``.
        - ``hatch``/``hatches`` with optional
          ``hatch_color``, ``hatch_width``, ``hatch_step``.
        - ``layer``: ``"below"`` or ``"above"``.
        - ``xref``/``yref``: Plotly refs (defaults: ``"x"``/``"paper"``).
        - ``enabled``: disable specific shade entry when ``False``.

        Example
        -------
        ``shade_configs=dict(
            daylight=dict(
                cols=("elong", "alpha"),
                ranges=((80, 120), (0, 20)),
                fillcolors=("lightgray", "#9ecae1"),
                alphas=(0.18, 0.25),
                hatches=(None, "/"),
            ),
        )``
    use_webgl : bool, optional
        If ``True`` (default), use ``plotly.graph_objects.Scattergl`` for line
        traces. If ``False``, use standard ``Scatter`` traces.
    show_xgrid : bool, optional
        Whether to draw x-axis grid lines.
    xgrid_month_step : int or None, optional
        Month interval used for x-axis major ticks when ``show_xgrid=True``.
        Passed as Plotly ``dtick="M{step}"`` when not ``None``.
    domain_left : float, optional
        Left edge of the x-axis domain in paper coordinates (default ``0.0``,
        clamped to ``[0.0, 0.3]``). Extra left-side y-axis positions are
        clamped to ``<= max(domain_left - 0.02, 0)``; with the default of
        ``0.0`` this means all auto-positioned left axes collapse to
        position ``0.0``. Set ``domain_left`` to e.g. ``0.12`` when using
        extra left axes so that positions have room.
    domain_right : float, optional
        Right edge of the x-axis domain in paper coordinates (default
        ``0.78``, clamped to ``[0.2, 0.92]`` when right axes exist). Extra
        right-side y-axis positions are clamped to
        ``>= min(domain_right + 0.02, 0.99)``.
    left_positions : sequence of float or None, optional
        Preferred positions for extra left-side y-axes (overlay axes with
        ``side="left"`` other than the primary). Values are clamped to
        ``<= max(domain_left - 0.02, 0)`` and enforced to have at least
        ``0.03`` spacing between neighbours.  If there are more left axes
        than entries, subsequent positions are auto-generated at ``-0.07``
        intervals.  Explicit ``position`` in an axis config overrides this
        but is still subject to clamping.
    right_positions : sequence of float, optional
        Preferred positions for right-side y-axes in paper coordinates.
        Values are clamped to ``>= min(domain_right + 0.02, 0.99)`` and
        enforced to have at least ``0.03`` spacing.  If there are more
        right axes than entries, subsequent positions are auto-generated at
        ``+0.07`` intervals.  Explicit ``position`` in an axis config
        overrides this but is still subject to clamping.
    left_margin : int, optional
        Left layout margin in pixels.
    right_margin : int, optional
        Right layout margin in pixels.
    tickfont_size : int, optional
        Tick label font size for x/y axes.
    legend_fontsize : int, optional
        Legend text font size (default ``12``).
    title_standoff : int, optional
        Y-axis title standoff in pixels.
    fig_width : int, optional
        Figure width in pixels.
    fig_height : int, optional
        Figure height in pixels.
    output_html : str or pathlib.Path, optional
        If provided, the function automatically calls ``save_plotly_html`` with
        this path before returning the figure.

    Notes
    -----
    - Only axes with at least one finite column are drawn.
    - Empty/missing columns are skipped automatically unless ``required=True``.
    - Hatch patterns on shade regions generate individual SVG line shapes per
      stripe per contiguous interval.  For dense data with many shaded
      intervals, this can produce a large number of shapes and slow down
      browser rendering.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure object.

    Raises
    ------
    ImportError
        If Plotly is not installed.
    TypeError
        If ``axis_configs`` or ``shade_configs`` has an invalid container type.
    ValueError
        If required axis/shade definitions are invalid (e.g., malformed ranges
        or required columns with no finite data).
    """
    _require_plotly()

    def _times_to_datetime64(tin):
        if hasattr(tin, "to_datetime"):
            return np.array(
                tin.to_datetime(leap_second_strict="silent"),
                dtype="datetime64[ns]",
            )
        return np.array(tin, dtype="datetime64[ns]")

    def _col_has_finite(colname):
        if colname is None:
            return False
        if colname not in dataframe:
            return False
        y = np.asarray(dataframe[colname], dtype=float)
        return np.isfinite(y).any()

    def _get_y(colname):
        return np.asarray(dataframe[colname], dtype=float)

    def _as_tuple(value):
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        try:
            return tuple(value)
        except TypeError:
            return (value,)

    def _broadcast(value, n_items, default):
        if n_items <= 0:
            return []
        if value is None:
            return [default] * n_items
        if isinstance(value, str):
            seq = [value]
        else:
            try:
                seq = list(value)
            except TypeError:
                seq = [value]
        if not seq:
            seq = [default]
        if len(seq) < n_items:
            seq.extend([seq[-1]] * (n_items - len(seq)))
        return seq[:n_items]

    def _pick(raw_cfg, *keys, default=None):
        for key in keys:
            if key in raw_cfg:
                return raw_cfg[key]
        return default

    def _is_number(value):
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def _clamp_unit_interval(value):
        if value is None:
            return None
        out = float(value)
        return float(min(max(out, 0.0), 1.0))

    def _normalize_range_pair(range_value, *, name):
        vals = _as_tuple(range_value)
        if len(vals) != 2 or (not _is_number(vals[0])) or (not _is_number(vals[1])):
            raise ValueError(f"{name} must be a 2-element numeric range.")
        low, high = float(vals[0]), float(vals[1])
        if low > high:
            low, high = high, low
        return (low, high)

    def _normalize_range_list(range_value, n_items, *, name):
        if range_value is None:
            raise ValueError(f"{name} must define `range`/`between`/`ranges`.")
        as_tuple = _as_tuple(range_value)

        if len(as_tuple) == 2 and _is_number(as_tuple[0]) and _is_number(as_tuple[1]):
            base = _normalize_range_pair(as_tuple, name=name)
            return [base] * n_items

        try:
            seq = list(range_value)
        except TypeError as exc:
            raise ValueError(f"{name} ranges are invalid.") from exc

        if not seq:
            raise ValueError(f"{name} ranges are empty.")

        parsed = [_normalize_range_pair(item, name=name) for item in seq]
        if len(parsed) < n_items:
            parsed.extend([parsed[-1]] * (n_items - len(parsed)))
        return parsed[:n_items]

    def _default_hover(trace_name, unit_text):
        suffix = f" {unit_text}" if unit_text else ""
        return f"{trace_name}=%{{y:.3f}}{suffix}<extra></extra>"

    def _normalize_axis_config(axis_name, raw_cfg):
        if not isinstance(raw_cfg, Mapping):
            raise TypeError(
                f"axis config {axis_name!r} must be a mapping, got {type(raw_cfg)!r}"
            )

        cols = tuple(
            col
            for col in _as_tuple(_pick(raw_cfg, "cols", "columns", default=()))
            if col is not None
        )
        required = bool(raw_cfg.get("required", False))
        if not cols:
            if required:
                raise ValueError(f"Axis {axis_name!r} has no columns but is required.")
            return None

        valid_indices = [i for i, col in enumerate(cols) if _col_has_finite(col)]
        if not valid_indices:
            if required:
                raise ValueError(
                    f"Axis {axis_name!r} has no finite data in columns {cols!r}."
                )
            return None

        side = str(_pick(raw_cfg, "side", default="right")).lower()
        if side not in {"left", "right"}:
            raise ValueError(
                f"Axis {axis_name!r} has invalid side {side!r}. Use 'left' or 'right'."
            )

        axis_label = str(_pick(raw_cfg, "label", "title", default=axis_name))
        axis_range = _pick(raw_cfg, "range", "ylim", default=None)
        axis_position = _pick(raw_cfg, "position", default=None)
        axis_showgrid = raw_cfg.get("showgrid", side == "left")
        axis_linewidth = float(_pick(raw_cfg, "axis_linewidth", default=2))
        axis_title_color = _pick(raw_cfg, "title_color", default=None)
        axis_tick_color = _pick(raw_cfg, "tick_color", default=None)
        axis_line_color = _pick(raw_cfg, "linecolor", default=None)
        axis_invert = bool(raw_cfg.get("invert", False))

        n_cols = len(cols)
        hidden_full = _broadcast(
            _pick(raw_cfg, "initially_hidden", default=None),
            n_cols,
            False,
        )
        names_full = _broadcast(
            _pick(raw_cfg, "names", "trace_names", default=None),
            n_cols,
            None,
        )
        colors_full = _broadcast(
            _pick(raw_cfg, "colors", "color", default=None),
            n_cols,
            "#444444",
        )
        widths_full = _broadcast(
            _pick(
                raw_cfg,
                "lw",
                "line_width",
                "line_widths",
                "width",
                "widths",
                default=None,
            ),
            n_cols,
            2,
        )
        dashes_full = _broadcast(
            _pick(raw_cfg, "dash", "dashes", default=None),
            n_cols,
            "solid",
        )
        units_full = _broadcast(
            _pick(raw_cfg, "unit", "units", default=None),
            n_cols,
            "",
        )
        hover_full = _broadcast(
            _pick(raw_cfg, "hovertemplate", "hovertemplates", default=None),
            n_cols,
            None,
        )

        traces = []
        for idx in valid_indices:
            trace_name = names_full[idx]
            if trace_name is None:
                trace_name = str(cols[idx])
            unit_text = units_full[idx]
            hovertemplate = hover_full[idx]
            if hovertemplate is None:
                hovertemplate = _default_hover(str(trace_name), str(unit_text))

            traces.append(
                dict(
                    col=cols[idx],
                    name=str(trace_name),
                    color=str(colors_full[idx]),
                    width=float(widths_full[idx]),
                    dash=str(dashes_full[idx]),
                    hovertemplate=str(hovertemplate),
                    initially_hidden=bool(hidden_full[idx]),
                )
            )

        base_axis_color = str(_pick(raw_cfg, "axis_color", default=traces[0]["color"]))
        if axis_line_color is None:
            axis_line_color = base_axis_color
        if axis_title_color is None:
            axis_title_color = base_axis_color
        if axis_tick_color is None:
            axis_tick_color = base_axis_color

        return dict(
            key=str(axis_name),
            label=axis_label,
            side=side,
            range=axis_range,
            position=axis_position,
            showgrid=bool(axis_showgrid),
            linecolor=str(axis_line_color),
            linewidth=axis_linewidth,
            title_color=str(axis_title_color),
            tick_color=str(axis_tick_color),
            invert=axis_invert,
            traces=traces,
        )

    def _normalize_shade_config(shade_name, raw_cfg):
        if not isinstance(raw_cfg, Mapping):
            raise TypeError(
                f"shade config {shade_name!r} must be a mapping, got {type(raw_cfg)!r}"
            )

        if not bool(raw_cfg.get("enabled", True)):
            return None

        cols = tuple(
            col
            for col in _as_tuple(
                _pick(raw_cfg, "cols", "columns", "col", "column", "shade_col", default=())
            )
            if col is not None
        )
        required = bool(raw_cfg.get("required", False))
        if not cols:
            if required:
                raise ValueError(f"Shade {shade_name!r} has no columns but is required.")
            return None

        n_cols = len(cols)
        ranges_full = _normalize_range_list(
            _pick(raw_cfg, "ranges", "range", "between", "elong_range", default=None),
            n_cols,
            name=f"Shade {shade_name!r}",
        )
        fillcolors_full = _broadcast(
            _pick(raw_cfg, "fillcolors", "colors", "fillcolor", "color", default=None),
            n_cols,
            "rgba(128,128,128,0.18)",
        )
        opacities_full = _broadcast(
            _pick(raw_cfg, "opacities", "alphas", "opacity", "alpha", default=None),
            n_cols,
            None,
        )
        line_widths_full = _broadcast(
            _pick(raw_cfg, "line_width", "line_widths", "lw", "width", "widths", default=0.0),
            n_cols,
            0.0,
        )
        line_colors_full = _broadcast(
            _pick(raw_cfg, "line_colors", "line_color", "linecolors", "linecolor", default=None),
            n_cols,
            None,
        )
        y0_full = _broadcast(_pick(raw_cfg, "y0s", "y0", default=0.0), n_cols, 0.0)
        y1_full = _broadcast(_pick(raw_cfg, "y1s", "y1", default=1.0), n_cols, 1.0)
        hatches_full = _broadcast(
            _pick(raw_cfg, "hatches", "hatch", "patterns", "pattern", default=None),
            n_cols,
            None,
        )
        hatch_colors_full = _broadcast(
            _pick(raw_cfg, "hatch_colors", "hatch_color", default=None),
            n_cols,
            None,
        )
        hatch_widths_full = _broadcast(
            _pick(raw_cfg, "hatch_widths", "hatch_width", default=1.0),
            n_cols,
            1.0,
        )
        hatch_steps_full = _broadcast(
            _pick(raw_cfg, "hatch_steps", "hatch_step", "hatch_spacing", default=0.08),
            n_cols,
            0.08,
        )
        hatch_opacities_full = _broadcast(
            _pick(raw_cfg, "hatch_opacities", "hatch_opacity", "hatch_alpha", default=None),
            n_cols,
            None,
        )

        layer = str(_pick(raw_cfg, "layer", default="below"))
        xref = str(_pick(raw_cfg, "xref", default="x"))
        yref = str(_pick(raw_cfg, "yref", default="paper"))

        bands = []
        for i, col in enumerate(cols):
            low, high = ranges_full[i]
            bands.append(
                dict(
                    col=str(col),
                    low=float(low),
                    high=float(high),
                    fillcolor=str(fillcolors_full[i]),
                    opacity=_clamp_unit_interval(opacities_full[i]),
                    line_width=float(line_widths_full[i]),
                    line_color=None
                    if line_colors_full[i] is None
                    else str(line_colors_full[i]),
                    y0=float(y0_full[i]),
                    y1=float(y1_full[i]),
                    hatch=None if hatches_full[i] is None else str(hatches_full[i]),
                    hatch_color=None
                    if hatch_colors_full[i] is None
                    else str(hatch_colors_full[i]),
                    hatch_width=float(hatch_widths_full[i]),
                    hatch_step=float(hatch_steps_full[i]),
                    hatch_opacity=_clamp_unit_interval(hatch_opacities_full[i]),
                )
            )

        valid_bands = [band for band in bands if _col_has_finite(band["col"])]
        if not valid_bands:
            if required:
                raise ValueError(
                    f"Shade {shade_name!r} has no finite data in columns {cols!r}."
                )
            return None

        return dict(
            key=str(shade_name),
            layer=layer,
            xref=xref,
            yref=yref,
            initially_hidden=bool(raw_cfg.get("initially_hidden", False)),
            bands=valid_bands,
        )

    def _hatch_shapes_for_interval(
        *,
        x0,
        x1,
        y0,
        y1,
        xref,
        yref,
        layer,
        hatch,
        hatch_color,
        hatch_width,
        hatch_opacity,
        hatch_step,
    ):
        pattern = "" if hatch is None else str(hatch).strip().lower()
        if pattern in {"", "none", "off", "false", "0"}:
            return []

        try:
            x0_i = int(np.datetime64(x0, "ns").astype("int64"))
            x1_i = int(np.datetime64(x1, "ns").astype("int64"))
        except Exception:
            return []
        if x1_i <= x0_i:
            return []

        step = float(min(max(hatch_step, 0.01), 0.5))
        color = hatch_color
        width = float(max(hatch_width, 0.1))
        fracs = np.arange(0.0, 1.0 + step * 0.5, step)

        def _x_at(frac):
            return np.datetime64(int(x0_i + frac * (x1_i - x0_i)), "ns")

        def _line(xa, ya, xb, yb):
            line_shape = dict(
                type="line",
                xref=xref,
                yref=yref,
                x0=xa,
                y0=ya,
                x1=xb,
                y1=yb,
                line=dict(color=color, width=width),
                layer=layer,
            )
            if hatch_opacity is not None:
                line_shape["opacity"] = hatch_opacity
            return line_shape

        shapes_out = []
        add_vertical = pattern in {"|", "v", "vertical", "vert", "+", "grid"}
        add_horizontal = pattern in {"-", "h", "horizontal", "horz", "+", "grid"}
        add_diag_up = pattern in {"/", "diag", "diag_up", "x"}
        add_diag_down = pattern in {"\\", "diag_down", "x"}

        if add_vertical:
            for f in fracs:
                x = _x_at(f)
                shapes_out.append(_line(x, y0, x, y1))

        if add_horizontal:
            for f in fracs:
                y = y0 + f * (y1 - y0)
                shapes_out.append(_line(x0, y, x1, y))

        diag_offsets = np.arange(-1.0, 1.0 + step * 0.5, step)
        if add_diag_up:
            for d in diag_offsets:
                f0 = max(0.0, d)
                f1 = min(1.0, 1.0 + d)
                shapes_out.append(_line(_x_at(f0), y0, _x_at(f1), y1))

        if add_diag_down:
            for d in diag_offsets:
                f0 = max(0.0, d)
                f1 = min(1.0, 1.0 + d)
                shapes_out.append(_line(_x_at(f0), y1, _x_at(f1), y0))

        return shapes_out

    t = _times_to_datetime64(times)
    Trace = go.Scattergl if use_webgl else go.Scatter

    if isinstance(axis_configs, Mapping):
        axis_entries = list(axis_configs.items())
    elif isinstance(axis_configs, Sequence) and not isinstance(
        axis_configs, (str, bytes)
    ):
        axis_entries = []
        for i, cfg in enumerate(axis_configs):
            if not isinstance(cfg, Mapping):
                raise TypeError(
                    "axis_configs sequence entries must be mappings. "
                    f"Entry {i} is {type(cfg)!r}."
                )
            axis_name = cfg.get("key", cfg.get("id", cfg.get("name", f"axis{i + 1}")))
            axis_entries.append((axis_name, cfg))
    else:
        raise TypeError("axis_configs must be a mapping or sequence of mappings.")

    active_axes = []
    for axis_name, raw_cfg in axis_entries:
        normalized = _normalize_axis_config(str(axis_name), raw_cfg)
        if normalized is not None:
            active_axes.append(normalized)

    if not active_axes:
        raise ValueError(
            "No plottable axes remain after filtering missing/non-finite columns."
        )

    primary_idx = next(
        (i for i, axis in enumerate(active_axes) if axis["side"] == "left"),
        0,
    )
    if primary_idx != 0:
        active_axes.insert(0, active_axes.pop(primary_idx))
    active_axes[0]["side"] = "left"

    for i, axis in enumerate(active_axes):
        axis["axis_id"] = "y" if i == 0 else f"y{i + 1}"
        axis["layout_key"] = "yaxis" if i == 0 else f"yaxis{i + 1}"

    right_axes = [axis for axis in active_axes[1:] if axis["side"] == "right"]
    left_axes = [axis for axis in active_axes[1:] if axis["side"] == "left"]
    max_domain_right = 0.92 if right_axes else 1.0
    domain_right = float(min(max(domain_right, 0.2), max_domain_right))
    domain_left = float(max(min(domain_left, 0.3), 0.0))

    raw_right_positions = [
        float(p) for p in _as_tuple(right_positions) if p is not None
    ]
    min_pos = min(domain_right + 0.02, 0.99)
    prev_right_pos = min_pos - 0.03
    for i, axis in enumerate(right_axes):
        if axis["position"] is not None:
            pos = float(axis["position"])
        elif i < len(raw_right_positions):
            pos = raw_right_positions[i]
        elif raw_right_positions:
            pos = raw_right_positions[-1] + 0.07 * (i - len(raw_right_positions) + 1)
        else:
            pos = min_pos + 0.07 * i

        pos = float(min(max(pos, 0.0), 1.0))
        if pos < min_pos:
            pos = min_pos
        if pos <= prev_right_pos + 0.03:
            pos = min(prev_right_pos + 0.07, 0.99)
        axis["position"] = pos
        prev_right_pos = pos

    # Position extra left-side axes
    raw_left_positions = [
        float(p) for p in _as_tuple(left_positions) if p is not None
    ] if left_positions is not None else []
    max_left_pos = max(domain_left - 0.02, 0.0)
    prev_left_pos = max_left_pos + 0.03
    for i, axis in enumerate(left_axes):
        if axis["position"] is not None:
            pos = float(axis["position"])
        elif i < len(raw_left_positions):
            pos = raw_left_positions[i]
        elif raw_left_positions:
            pos = raw_left_positions[-1] - 0.07 * (i - len(raw_left_positions) + 1)
        else:
            pos = max_left_pos - 0.07 * i

        pos = float(min(max(pos, 0.0), 1.0))
        if pos > max_left_pos:
            pos = max_left_pos
        if pos >= prev_left_pos - 0.03:
            pos = max(prev_left_pos - 0.07, 0.0)
        axis["position"] = pos
        prev_left_pos = pos

    fig = go.Figure()
    for axis in active_axes:
        for trace_cfg in axis["traces"]:
            trace_vis = "legendonly" if trace_cfg["initially_hidden"] else True
            fig.add_trace(
                Trace(
                    x=t,
                    y=_get_y(trace_cfg["col"]),
                    mode="lines",
                    name=trace_cfg["name"],
                    visible=trace_vis,
                    line=dict(
                        color=trace_cfg["color"],
                        width=trace_cfg["width"],
                        dash=trace_cfg["dash"],
                    ),
                    yaxis=axis["axis_id"],
                    hovertemplate=trace_cfg["hovertemplate"],
                )
            )

    shade_source = [] if shade_configs is None else shade_configs

    if isinstance(shade_source, Mapping):
        shade_entries = list(shade_source.items())
    elif isinstance(shade_source, Sequence) and not isinstance(
        shade_source, (str, bytes)
    ):
        shade_entries = []
        for i, cfg in enumerate(shade_source):
            if not isinstance(cfg, Mapping):
                raise TypeError(
                    "shade_configs sequence entries must be mappings. "
                    f"Entry {i} is {type(cfg)!r}."
                )
            shade_name = cfg.get("key", cfg.get("id", cfg.get("name", f"shade{i + 1}")))
            shade_entries.append((shade_name, cfg))
    else:
        raise TypeError("shade_configs must be a mapping or sequence of mappings.")

    normalized_shades = []
    for shade_name, raw_cfg in shade_entries:
        normalized = _normalize_shade_config(str(shade_name), raw_cfg)
        if normalized is not None:
            normalized_shades.append(normalized)

    shapes = []
    for shade_cfg in normalized_shades:
        for band in shade_cfg["bands"]:
            shade_vals = _get_y(band["col"])
            shade_mask = (shade_vals >= band["low"]) & (shade_vals <= band["high"])
            if not np.any(shade_mask):
                continue

            idx = np.flatnonzero(shade_mask)
            breaks = np.where(np.diff(idx) > 1)[0]
            starts = np.r_[idx[0], idx[breaks + 1]]
            ends = np.r_[idx[breaks], idx[-1]]

            for s, e in zip(starts, ends):
                line_dict = dict(width=band["line_width"])
                if band["line_color"] is not None:
                    line_dict["color"] = band["line_color"]

                rect_shape = dict(
                    type="rect",
                    xref=shade_cfg["xref"],
                    yref=shade_cfg["yref"],
                    x0=t[s],
                    x1=t[e],
                    y0=band["y0"],
                    y1=band["y1"],
                    fillcolor=band["fillcolor"],
                    line=line_dict,
                    layer=shade_cfg["layer"],
                )
                if band["opacity"] is not None:
                    rect_shape["opacity"] = band["opacity"]
                shapes.append(rect_shape)

                hatch_color = band["hatch_color"]
                if hatch_color is None:
                    hatch_color = band["line_color"] or "rgba(80,80,80,0.45)"

                shapes.extend(
                    _hatch_shapes_for_interval(
                        x0=t[s],
                        x1=t[e],
                        y0=band["y0"],
                        y1=band["y1"],
                        xref=shade_cfg["xref"],
                        yref=shade_cfg["yref"],
                        layer=shade_cfg["layer"],
                        hatch=band["hatch"],
                        hatch_color=hatch_color,
                        hatch_width=band["hatch_width"],
                        hatch_opacity=band["hatch_opacity"],
                        hatch_step=band["hatch_step"],
                    )
                )

    if shapes:
        fig.update_layout(shapes=shapes)

    xaxis = dict(
        title="Time",
        domain=[domain_left, domain_right],
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        showline=True,
        linecolor="rgba(0,0,0,0.65)",
        linewidth=1,
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        showticklabels=True,
        tickfont=dict(size=tickfont_size),
        showgrid=bool(show_xgrid),
        gridcolor="rgba(0,0,0,0.08)",
        gridwidth=1,
        tickformatstops=[
            dict(dtickrange=[None, 1000], value="%H:%M:%S.%L\n%Y-%m-%d"),
            dict(dtickrange=[1000, 60000], value="%H:%M:%S\n%Y-%m-%d"),
            dict(dtickrange=[60000, 86400000], value="%H:%M\n%Y-%m-%d"),
            dict(dtickrange=[86400000, 2592000000], value="%Y-%m-%d"),
            dict(dtickrange=[2592000000, 31536000000], value="%Y-%m"),
            dict(dtickrange=[31536000000, None], value="%Y"),
        ],
        hoverformat="%Y-%m-%d %H:%M",
        tickangle=-45,
    )
    if show_xgrid and (xgrid_month_step is not None):
        xaxis["dtick"] = f"M{xgrid_month_step}"

    common = dict(ticks="outside")

    def ytitle(text, color):
        return dict(text=text, font=dict(color=color, size=tickfont_size))

    fig.update_layout(
        width=fig_width,
        height=fig_height,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=legend_fontsize),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(l=left_margin, r=right_margin, t=60, b=60),
        xaxis=xaxis,
    )

    layout_updates = {}
    for i, axis in enumerate(active_axes):
        axis_layout = dict(
            title=ytitle(axis["label"], axis["title_color"]),
            range=list(axis["range"]) if axis["range"] is not None else None,
            showgrid=axis["showgrid"],
            gridcolor="rgba(0,0,0,0.12)",
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=axis["linecolor"],
            linewidth=axis["linewidth"],
            tickfont=dict(color=axis["tick_color"], size=tickfont_size),
            title_standoff=title_standoff,
            **common,
        )

        all_hidden = all(tc["initially_hidden"] for tc in axis["traces"])
        if all_hidden:
            axis_layout["visible"] = False

        if i > 0:
            axis_layout.update(
                overlaying="y",
                side=axis["side"],
                anchor="free",
                position=float(axis["position"]),
            )

        layout_updates[axis["layout_key"]] = axis_layout

    fig.update_layout(**layout_updates)

    for axis in active_axes:
        if not axis["invert"]:
            continue
        layout_axis = getattr(fig.layout, axis["layout_key"])
        if axis["range"] is not None:
            layout_axis.autorange = False
            layout_axis.range = list(axis["range"])
        else:
            layout_axis.autorange = "reversed"

    # Embed shade metadata + column data for interactive HTML controls
    shade_col_data = {}
    shade_band_meta = []
    for shade_cfg in normalized_shades:
        for band in shade_cfg["bands"]:
            col = band["col"]
            if col not in shade_col_data:
                vals = _get_y(col)
                shade_col_data[col] = [
                    float(v) if np.isfinite(v) else None for v in vals
                ]
            shade_band_meta.append(
                dict(
                    col=band["col"],
                    low=band["low"],
                    high=band["high"],
                    fillcolor=band["fillcolor"],
                    opacity=band["opacity"],
                    line_width=band["line_width"],
                    line_color=band["line_color"],
                    y0=band["y0"],
                    y1=band["y1"],
                    hatch=band["hatch"],
                    hatch_color=band["hatch_color"],
                    hatch_width=band["hatch_width"],
                    hatch_step=band["hatch_step"],
                    hatch_opacity=band["hatch_opacity"],
                    layer=shade_cfg["layer"],
                    xref=shade_cfg["xref"],
                    yref=shade_cfg["yref"],
                    key=shade_cfg["key"],
                    initially_hidden=shade_cfg["initially_hidden"],
                )
            )

    t_ns = [int(v) for v in np.array(t, dtype="datetime64[ns]").astype("int64")]

    all_numeric_cols = []
    _iter_cols = (
        dataframe.columns
        if hasattr(dataframe, "columns")
        else dataframe.keys()
    )
    for cname in _iter_cols:
        try:
            arr = np.asarray(dataframe[cname], dtype=float)
            if np.isfinite(arr).any():
                all_numeric_cols.append(str(cname))
                if cname not in shade_col_data:
                    shade_col_data[cname] = [
                        float(v) if np.isfinite(v) else None for v in arr
                    ]
        except (TypeError, ValueError):
            continue

    existing_meta = fig.layout.meta if fig.layout.meta is not None else {}
    if not isinstance(existing_meta, dict):
        existing_meta = {}
    existing_meta["ephem_shade_data"] = dict(
        bands=shade_band_meta,
        columns=shade_col_data,
        column_names=all_numeric_cols,
        time_ns=t_ns,
    )
    fig.update_layout(meta=existing_meta)

    if output_html is not None:
        save_plotly_html(fig, output_html)

    return fig


def save_plotly_html(
    fig,
    path,
    title="Interactive ephemeris",
    standalone=True,
    scroll_zoom=True,
    add_trace_controls=True,
    add_axis_controls=True,
    add_layout_controls=True,
    add_shade_controls=True,
):
    """
    Save Plotly figure as standalone HTML with optional interactive controls.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to export.
    path : str or pathlib.Path
        Output HTML file path.
    title : str, optional
        HTML document title.
    standalone : bool, optional
        If ``True``, embed Plotly JS in the output file. If ``False``, use
        CDN-based Plotly JS.
    scroll_zoom : bool, optional
        Whether wheel/pinch zoom is enabled in the rendered figure.
    add_trace_controls : bool, optional
        Include per-trace controls (visibility, legend name, color, width,
        dash).
    add_axis_controls : bool, optional
        Include per-y-axis controls (visibility, grid toggle, position,
        range/autorange).
    add_shade_controls : bool, optional
        Include per-shade-band controls (enabled, column, range, fill
        color, opacity, hatch pattern). Requires that shade metadata was
        embedded by ``plot_ephemeris_plotly``.
    add_layout_controls : bool, optional
        Include layout-level controls (legend font size, axis font size,
        x-axis grid toggle).

    Returns
    -------
    pathlib.Path
        Path to the written HTML file.

    Raises
    ------
    ImportError
        If Plotly is not installed.
    """
    _require_plotly()
    path = Path(path)

    if not add_trace_controls and not add_axis_controls and not add_layout_controls and not add_shade_controls:
        fig.write_html(
            path,
            include_plotlyjs=(True if standalone else "cdn"),
            full_html=True,
            config={
                "scrollZoom": bool(scroll_zoom),
                "displayModeBar": True,
                "responsive": True,
            },
        )
        return path

    fig_json = fig.to_plotly_json()
    div_id = "plotly-ephem-fig"

    def _axis_sort_key(axis_id):
        if axis_id == "y":
            return 1
        try:
            return int(axis_id[1:])
        except (TypeError, ValueError):
            return 10_000

    def _axis_id_from_layout_key(layout_key):
        if layout_key == "yaxis":
            return "y"
        if layout_key.startswith("yaxis") and layout_key[5:].isdigit():
            return f"y{layout_key[5:]}"
        return None

    def _axis_label(layout_axis, axis_id):
        if not isinstance(layout_axis, Mapping):
            return axis_id
        title = layout_axis.get("title")
        if isinstance(title, Mapping):
            text = title.get("text")
            if text:
                return str(text)
        if isinstance(title, str) and title:
            return title
        return axis_id

    def _axis_range_info(layout_axis):
        if not isinstance(layout_axis, Mapping):
            return (None, None, True)
        axis_range = layout_axis.get("range")
        range_min = None
        range_max = None
        if isinstance(axis_range, Sequence) and not isinstance(
            axis_range, (str, bytes)
        ):
            if len(axis_range) >= 2:
                range_min = axis_range[0]
                range_max = axis_range[1]
        autorange = layout_axis.get("autorange", True)
        if autorange == "reversed":
            autorange = False
        return (range_min, range_max, bool(autorange))

    layout = fig_json.get("layout", {})
    meta = layout.get("meta", {}) if isinstance(layout, Mapping) else {}

    # Extract shade data for interactive shade controls
    shade_data = {}
    if isinstance(meta, Mapping):
        shade_data = meta.get("ephem_shade_data", {})
    shade_bands_js = shade_data.get("bands", []) if isinstance(shade_data, Mapping) else []
    shade_columns_js = shade_data.get("columns", {}) if isinstance(shade_data, Mapping) else {}
    shade_col_names_js = shade_data.get("column_names", []) if isinstance(shade_data, Mapping) else []
    shade_time_ns_js = shade_data.get("time_ns", []) if isinstance(shade_data, Mapping) else []
    raw_axis_meta = []
    if isinstance(meta, Mapping):
        raw_axis_meta = meta.get("ephem_axis_controls", [])

    x_axis_control = dict(showgrid=False)
    if isinstance(layout, Mapping):
        xaxis_layout = layout.get("xaxis", {})
        if isinstance(xaxis_layout, Mapping):
            x_axis_control["showgrid"] = bool(xaxis_layout.get("showgrid", False))

    axis_controls = []
    seen_axes = set()

    if isinstance(raw_axis_meta, Sequence) and not isinstance(
        raw_axis_meta, (str, bytes)
    ):
        for axis_info in raw_axis_meta:
            if not isinstance(axis_info, Mapping):
                continue
            axis_id = axis_info.get("axis_id")
            layout_key = axis_info.get("layout_key")
            label = axis_info.get("label", axis_id)
            if not axis_id or not layout_key:
                continue
            if axis_id in seen_axes:
                continue
            layout_axis = layout.get(layout_key, {})
            range_min, range_max, autorange = _axis_range_info(layout_axis)
            axis_controls.append(
                dict(
                    axis_id=axis_id,
                    layout_key=layout_key,
                    label=str(label),
                    visible=bool(layout_axis.get("visible", True)),
                    position=layout_axis.get("position", None),
                    side=layout_axis.get("side", "left"),
                    movable=(str(layout_key) != "yaxis"),
                    showgrid=bool(layout_axis.get("showgrid", False)),
                    range_min=range_min,
                    range_max=range_max,
                    autorange=autorange,
                )
            )
            seen_axes.add(axis_id)

    if isinstance(layout, Mapping):
        for layout_key, layout_axis in layout.items():
            axis_id = _axis_id_from_layout_key(str(layout_key))
            if axis_id is None or axis_id in seen_axes:
                continue
            range_min, range_max, autorange = _axis_range_info(layout_axis)
            axis_controls.append(
                dict(
                    axis_id=axis_id,
                    layout_key=str(layout_key),
                    label=_axis_label(layout_axis, axis_id),
                    visible=(
                        bool(layout_axis.get("visible", True))
                        if isinstance(layout_axis, Mapping)
                        else True
                    ),
                    position=(
                        layout_axis.get("position", None)
                        if isinstance(layout_axis, Mapping)
                        else None
                    ),
                    side=(
                        layout_axis.get("side", "left")
                        if isinstance(layout_axis, Mapping)
                        else "left"
                    ),
                    movable=(str(layout_key) != "yaxis"),
                    showgrid=(
                        bool(layout_axis.get("showgrid", False))
                        if isinstance(layout_axis, Mapping)
                        else False
                    ),
                    range_min=range_min,
                    range_max=range_max,
                    autorange=autorange,
                )
            )
            seen_axes.add(axis_id)

    axis_controls.sort(key=lambda item: _axis_sort_key(item["axis_id"]))

    legend_font_size = 14
    label_font_size = 14
    if isinstance(layout, Mapping):
        legend = layout.get("legend", {})
        if isinstance(legend, Mapping):
            legend_font = legend.get("font", {})
            if isinstance(legend_font, Mapping) and legend_font.get("size") is not None:
                legend_font_size = legend_font.get("size")

        yaxis_layout = layout.get("yaxis", {})
        if isinstance(yaxis_layout, Mapping):
            y_tick_font = yaxis_layout.get("tickfont", {})
            if isinstance(y_tick_font, Mapping) and y_tick_font.get("size") is not None:
                label_font_size = y_tick_font.get("size")

    # Build trace metadata
    trace_controls = []
    for i, tr in enumerate(fig_json["data"]):
        line = tr.get("line", {})
        name = tr.get("name", f"trace_{i}")
        axis_id = tr.get("yaxis", "y")

        visible = tr.get("visible", True)
        if visible == "legendonly":
            visible = False

        trace_controls.append(
            {
                "index": i,
                "name": name,
                "axis": axis_id,
                "visible": bool(visible),
                "color": line.get("color", "#000000"),
                "width": line.get("width", 2),
                "dash": line.get("dash", "solid"),
            }
        )

    plot_html = pio.to_html(
        fig,
        include_plotlyjs=(True if standalone else "cdn"),
        full_html=False,
        div_id=div_id,
        config={
            "scrollZoom": bool(scroll_zoom),
            "displayModeBar": True,
            "responsive": True,
        },
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
  }}
  .container {{
    display: flex;
    flex-direction: row;
    gap: 12px;
    padding: 12px;
    box-sizing: border-box;
  }}
  .controls {{
    width: 380px;
    min-width: 320px;
    max-height: 95vh;
    overflow-y: auto;
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 12px;
    box-sizing: border-box;
    background: #fafafa;
  }}
  .plot-wrapper {{
    flex: 1;
    min-width: 0;
    position: relative;
  }}
  .resize-handle {{
    position: absolute;
    right: 0;
    bottom: 0;
    width: 18px;
    height: 18px;
    cursor: nwse-resize;
    background: linear-gradient(135deg, transparent 50%, #aaa 50%, #aaa 55%, transparent 55%, transparent 70%, #aaa 70%, #aaa 75%, transparent 75%);
    z-index: 10;
    border-radius: 0 0 4px 0;
  }}
  .trace-block {{
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 8px;
    margin-bottom: 6px;
    background: white;
  }}
  .axis-block {{
    border: 1px solid #bbb;
    border-radius: 6px;
    padding: 6px 8px;
    margin-bottom: 6px;
    background: white;
  }}
  .trace-title {{
    font-weight: bold;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
  }}
  .row label {{
    min-width: 70px;
    font-size: 13px;
  }}
  .inline-row {{
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 3px 0;
    flex-wrap: wrap;
  }}
  .inline-row label {{
    font-size: 13px;
    white-space: nowrap;
  }}
  .compact-row {{
    display: flex;
    align-items: center;
    gap: 4px;
    margin: 3px 0;
  }}
  .compact-row input[type="color"] {{ width: 28px; height: 24px; padding: 0; border: 1px solid #ccc; }}
  .compact-row input[type="number"] {{ width: 52px; }}
  .compact-row select {{ max-width: 90px; }}
  select, input[type="color"], input[type="number"], input[type="text"] {{
    flex: 1;
  }}
  .small {{
    font-size: 12px;
    color: #666;
    margin-bottom: 6px;
  }}
  details > summary {{
    cursor: pointer;
    font-size: 16px;
    font-weight: bold;
    padding: 4px 0;
    user-select: none;
  }}
  details {{ margin-bottom: 8px; }}
  .splitter {{
    width: 6px;
    cursor: col-resize;
    background: #ddd;
    border-radius: 3px;
    flex-shrink: 0;
    transition: background 0.15s;
  }}
  .splitter:hover, .splitter.active {{
    background: #999;
  }}
</style>
</head>
<body>
<div class="container">
  <div class="controls">
    {"<details open><summary>Layout</summary><div class='small'>Adjust legend and label font sizes.</div><div id='layout-controls'></div></details>" if add_layout_controls else ""}
    {"<details open><summary>Axes</summary><div class='small'>Toggle y-axes, y-grid, positions, ranges.</div><div id='axis-controls'></div></details>" if add_axis_controls else ""}
    {"<details><summary>Traces</summary><div class='small'>Visibility, color, width, dash.</div><div id='trace-controls'></div></details>" if add_trace_controls else ""}
    {"<details><summary>Shades</summary><div class='small'>Toggle shading, column, range, color, opacity, hatch.</div><div id='shade-controls'></div></details>" if add_shade_controls else ""}
  </div>
  <div class="splitter" id="sidebar-splitter"></div>
  <div class="plot-wrapper">
    {plot_html}
    <div class="resize-handle" id="resize-handle"></div>
  </div>
</div>

<script>
const TRACE = {json.dumps(trace_controls)};
const AXES = {json.dumps(axis_controls)};
const X_AXIS = {json.dumps(x_axis_control)};
const PLOT_ID = "{div_id}";
const ADD_TRACE_CONTROLS = {str(bool(add_trace_controls)).lower()};
const ADD_AXIS_CONTROLS = {str(bool(add_axis_controls)).lower()};
const ADD_LAYOUT_CONTROLS = {str(bool(add_layout_controls)).lower()};
const ADD_SHADE_CONTROLS = {str(bool(add_shade_controls)).lower()};
const DEFAULT_LEGEND_FONT_SIZE = {json.dumps(legend_font_size)};
const DEFAULT_LABEL_FONT_SIZE = {json.dumps(label_font_size)};
const SHADE_BANDS = {json.dumps(shade_bands_js)};
const SHADE_COLUMNS = {json.dumps(shade_columns_js)};
const SHADE_COL_NAMES = {json.dumps(shade_col_names_js)};
const SHADE_TIME_NS = {json.dumps(shade_time_ns_js)};

function restyleTrace(i, prop, val) {{
  const plot = document.getElementById(PLOT_ID);
  const update = {{}};
  update[prop] = val;
  Plotly.restyle(plot, update, [i]);
}}

function relayout(update) {{
  const plot = document.getElementById(PLOT_ID);
  Plotly.relayout(plot, update);
}}

function normalizeColor(c) {{
  if (typeof c !== "string") return "#000000";
  const s = c.trim().toLowerCase();

  if (/^#[0-9a-f]{{6}}$/.test(s)) return s;
  if (/^#[0-9a-f]{{3}}$/.test(s)) {{
    return "#" + s[1] + s[1] + s[2] + s[2] + s[3] + s[3];
  }}

  const named = {{
    red: "#ff0000",
    blue: "#0000ff",
    green: "#008000",
    black: "#000000",
    orange: "#ffa500",
    purple: "#800080",
    gray: "#808080",
    grey: "#808080",
    white: "#ffffff",
    yellow: "#ffff00",
    darkblue: "#00008b",
    darkgreen: "#006400",
    darkorange: "#ff8c00",
    magenta: "#ff00ff",
    cyan: "#00ffff"
  }};

  return named[s] || "#000000";
}}

function makeControlRow(labelText, inputEl) {{
  const row = document.createElement("div");
  row.className = "row";

  const label = document.createElement("label");
  label.textContent = labelText;

  row.appendChild(label);
  row.appendChild(inputEl);
  return row;
}}

function toggleAxis(axis, state) {{
  const axisInfo = AXES.find(a => a.axis_id === axis);
  if (!axisInfo) return;
  const layoutUpdate = {{}};
  layoutUpdate[axisInfo.layout_key + ".visible"] = state;
  relayout(layoutUpdate);

  TRACE.forEach(t => {{
    if (t.axis === axis) {{
      const plot = document.getElementById(PLOT_ID);
      Plotly.restyle(plot, {{ visible: state }}, [t.index]);
    }}
  }});
}}

function setAxisPosition(axisId, rawPos) {{
  const axisInfo = AXES.find(a => a.axis_id === axisId);
  if (!axisInfo || !axisInfo.movable) return;

  const pos = parseFloat(rawPos);
  if (Number.isNaN(pos)) return;
  const bounded = Math.min(1.0, Math.max(0.0, pos));
  axisInfo.position = bounded;

  const update = {{}};
  update[axisInfo.layout_key + ".position"] = bounded;
  update[axisInfo.layout_key + ".anchor"] = "free";
  relayout(update);
}}

function setAxisRange(axisId, rawMin, rawMax) {{
  const axisInfo = AXES.find(a => a.axis_id === axisId);
  if (!axisInfo) return;

  const minVal = parseFloat(rawMin);
  const maxVal = parseFloat(rawMax);
  if (Number.isNaN(minVal) || Number.isNaN(maxVal)) return;

  axisInfo.range_min = minVal;
  axisInfo.range_max = maxVal;
  axisInfo.autorange = false;

  const update = {{}};
  update[axisInfo.layout_key + ".autorange"] = false;
  update[axisInfo.layout_key + ".range"] = [minVal, maxVal];
  relayout(update);
}}

function setAxisAutorange(axisId, enabled) {{
  const axisInfo = AXES.find(a => a.axis_id === axisId);
  if (!axisInfo) return;

  axisInfo.autorange = !!enabled;
  const update = {{}};
  update[axisInfo.layout_key + ".autorange"] = !!enabled;
  relayout(update);
}}

function setAxisGrid(axisId, enabled) {{
  const axisInfo = AXES.find(a => a.axis_id === axisId);
  if (!axisInfo) return;

  axisInfo.showgrid = !!enabled;
  const update = {{}};
  update[axisInfo.layout_key + ".showgrid"] = !!enabled;
  relayout(update);
}}

function setXGrid(enabled) {{
  X_AXIS.showgrid = !!enabled;
  relayout({{"xaxis.showgrid": !!enabled}});
}}

function setLegendFontSize(rawSize) {{
  const size = parseFloat(rawSize);
  if (Number.isNaN(size)) return;
  relayout({{"legend.font.size": size}});
}}

function setLabelFontSize(rawSize) {{
  const size = parseFloat(rawSize);
  if (Number.isNaN(size)) return;

  const update = {{
    "xaxis.title.font.size": size,
    "xaxis.tickfont.size": size,
  }};

  AXES.forEach(axis => {{
    update[axis.layout_key + ".title.font.size"] = size;
    update[axis.layout_key + ".tickfont.size"] = size;
  }});

  relayout(update);
}}

function buildLayoutControls() {{
  if (!ADD_LAYOUT_CONTROLS) return;

  const div = document.getElementById("layout-controls");
  if (!div) return;

  const legendSize = document.createElement("input");
  legendSize.type = "number";
  legendSize.min = "6";
  legendSize.max = "40";
  legendSize.step = "1";
  legendSize.value = DEFAULT_LEGEND_FONT_SIZE;
  legendSize.addEventListener("input", () => {{
    setLegendFontSize(legendSize.value);
  }});
  div.appendChild(makeControlRow("legend font", legendSize));

  const labelSize = document.createElement("input");
  labelSize.type = "number";
  labelSize.min = "6";
  labelSize.max = "40";
  labelSize.step = "1";
  labelSize.value = DEFAULT_LABEL_FONT_SIZE;
  labelSize.addEventListener("input", () => {{
    setLabelFontSize(labelSize.value);
  }});
  div.appendChild(makeControlRow("axis font", labelSize));

  const xGridInput = document.createElement("input");
  xGridInput.type = "checkbox";
  xGridInput.checked = !!X_AXIS.showgrid;
  xGridInput.addEventListener("change", () => {{
    setXGrid(xGridInput.checked);
  }});
  div.appendChild(makeControlRow("x grid", xGridInput));

  // Time format dropdown
  const ISO_STOPS = [
    {{dtickrange: [null, 1000], value: "%H:%M:%S.%L\\n%Y-%m-%d"}},
    {{dtickrange: [1000, 60000], value: "%H:%M:%S\\n%Y-%m-%d"}},
    {{dtickrange: [60000, 86400000], value: "%H:%M\\n%Y-%m-%d"}},
    {{dtickrange: [86400000, 2592000000], value: "%Y-%m-%d"}},
    {{dtickrange: [2592000000, 31536000000], value: "%Y-%m"}},
    {{dtickrange: [31536000000, null], value: "%Y"}},
  ];
  const DISABLED_STOPS = ISO_STOPS.map(s => ({{...s, enabled: false}}));
  const timeFmtSelect = document.createElement("select");
  [
    ["ISO auto", "__iso_auto__"],
    ["Plotly auto", "__plotly__"],
    ["%Y-%m-%d", "%Y-%m-%d"],
    ["%d %b %Y", "%d %b %Y"],
    ["%b %Y", "%b %Y"],
  ].forEach(([label, fmt]) => {{
    const opt = document.createElement("option");
    opt.value = fmt;
    opt.textContent = label;
    if (fmt === "__iso_auto__") opt.selected = true;
    timeFmtSelect.appendChild(opt);
  }});
  timeFmtSelect.addEventListener("change", () => {{
    const val = timeFmtSelect.value;
    if (val === "__iso_auto__") {{
      relayout({{"xaxis.tickformat": null, "xaxis.tickformatstops": ISO_STOPS, "xaxis.hoverformat": "%Y-%m-%d %H:%M"}});
    }} else if (val === "__plotly__") {{
      relayout({{"xaxis.tickformat": null, "xaxis.tickformatstops": DISABLED_STOPS, "xaxis.hoverformat": "%Y-%m-%d %H:%M"}});
    }} else {{
      relayout({{"xaxis.tickformat": val, "xaxis.tickformatstops": DISABLED_STOPS, "xaxis.hoverformat": val}});
    }}
  }});
  div.appendChild(makeControlRow("time fmt", timeFmtSelect));
}}

function buildAxisControls() {{
  if (!ADD_AXIS_CONTROLS) return;

  const div = document.getElementById("axis-controls");
  if (!div) return;

  AXES.forEach(axis => {{
    const block = document.createElement("div");
    block.className = "axis-block";

    const row = document.createElement("div");
    row.className = "row";

    const chk = document.createElement("input");
    chk.type = "checkbox";
    chk.checked = !!axis.visible;
    chk.onchange = () => toggleAxis(axis.axis_id, chk.checked);

    const label = document.createElement("label");
    label.textContent = axis.label + " (" + axis.axis_id + ", " + axis.side + ")";

    row.appendChild(chk);
    row.appendChild(label);
    block.appendChild(row);

    if (axis.movable) {{
      const posInput = document.createElement("input");
      posInput.type = "number";
      posInput.min = "0";
      posInput.max = "1";
      posInput.step = "0.01";
      if (axis.position !== null && axis.position !== undefined) {{
        posInput.value = axis.position;
      }}
      posInput.addEventListener("input", () => {{
        setAxisPosition(axis.axis_id, posInput.value);
      }});
      block.appendChild(makeControlRow("position", posInput));
    }}

    const minInput = document.createElement("input");
    minInput.type = "number";
    minInput.step = "any";
    if (axis.range_min !== null && axis.range_min !== undefined) {{
      minInput.value = axis.range_min;
    }}

    const maxInput = document.createElement("input");
    maxInput.type = "number";
    maxInput.step = "any";
    if (axis.range_max !== null && axis.range_max !== undefined) {{
      maxInput.value = axis.range_max;
    }}

    axis._minInput = minInput;
    axis._maxInput = maxInput;

    const resetBtn = document.createElement("button");
    resetBtn.textContent = "auto range";
    resetBtn.style.cssText = "font-size:12px; padding:1px 8px; cursor:pointer;";
    resetBtn.addEventListener("click", () => {{
      setAxisAutorange(axis.axis_id, true);
      // Read back the auto-computed range after a short delay
      setTimeout(() => {{
        const plot = document.getElementById(PLOT_ID);
        const layoutAxis = plot.layout[axis.layout_key];
        if (layoutAxis && layoutAxis.range) {{
          minInput.value = layoutAxis.range[0];
          maxInput.value = layoutAxis.range[1];
        }}
      }}, 100);
    }});

    const updateRange = () => {{
      setAxisRange(axis.axis_id, minInput.value, maxInput.value);
    }};
    minInput.addEventListener("change", updateRange);
    maxInput.addEventListener("change", updateRange);
    block.appendChild(makeControlRow("range min", minInput));
    block.appendChild(makeControlRow("range max", maxInput));

    // reset + y-grid on one line
    const gridInput = document.createElement("input");
    gridInput.type = "checkbox";
    gridInput.checked = !!axis.showgrid;
    gridInput.addEventListener("change", () => {{
      setAxisGrid(axis.axis_id, gridInput.checked);
    }});

    const inlineRow = document.createElement("div");
    inlineRow.className = "inline-row";
    inlineRow.appendChild(resetBtn);
    const gridLabel = document.createElement("label");
    gridLabel.textContent = "grid";
    inlineRow.appendChild(gridInput);
    inlineRow.appendChild(gridLabel);
    block.appendChild(inlineRow);

    div.appendChild(block);
  }});
}}

function buildTraceControls() {{
  if (!ADD_TRACE_CONTROLS) return;

  const container = document.getElementById("trace-controls");
  if (!container) return;

  TRACE.forEach(t => {{
    const block = document.createElement("div");
    block.className = "trace-block";

    // Title row: checkbox + name
    const titleRow = document.createElement("div");
    titleRow.className = "trace-title";

    const visibleInput = document.createElement("input");
    visibleInput.type = "checkbox";
    visibleInput.checked = !!t.visible;
    visibleInput.addEventListener("change", () => {{
      restyleTrace(t.index, "visible", visibleInput.checked);
    }});
    titleRow.appendChild(visibleInput);

    const nameSpan = document.createElement("span");
    nameSpan.textContent = t.name;
    titleRow.appendChild(nameSpan);

    block.appendChild(titleRow);

    // legend name
    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.value = t.name;
    nameInput.addEventListener("input", () => {{
      t.name = nameInput.value;
      nameSpan.textContent = nameInput.value;
      restyleTrace(t.index, "name", nameInput.value);
    }});
    block.appendChild(makeControlRow("name", nameInput));

    // Compact row: color + width + dash
    const compactRow = document.createElement("div");
    compactRow.className = "compact-row";

    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = normalizeColor(t.color);
    colorInput.addEventListener("input", () => {{
      restyleTrace(t.index, "line.color", colorInput.value);
    }});
    compactRow.appendChild(colorInput);

    const widthInput = document.createElement("input");
    widthInput.type = "number";
    widthInput.min = "0.5";
    widthInput.max = "10";
    widthInput.step = "0.5";
    widthInput.value = t.width;
    widthInput.addEventListener("input", () => {{
      const v = parseFloat(widthInput.value);
      if (!Number.isNaN(v)) {{
        restyleTrace(t.index, "line.width", v);
      }}
    }});
    compactRow.appendChild(widthInput);

    const dashInput = document.createElement("select");
    ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"].forEach(v => {{
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      if ((t.dash || "solid") === v) opt.selected = true;
      dashInput.appendChild(opt);
    }});
    dashInput.addEventListener("change", () => {{
      restyleTrace(t.index, "line.dash", dashInput.value);
    }});
    compactRow.appendChild(dashInput);

    block.appendChild(compactRow);

    container.appendChild(block);
  }});
}}

// ---- Shade Controls ----

// Count non-shade shapes so we can preserve them during recomputation.
// Shade shapes are the last N shapes in the initial layout.
let NON_SHADE_SHAPE_COUNT = 0;
(function countNonShadeShapes() {{
  const plot = document.getElementById(PLOT_ID);
  if (!plot || !plot.layout || !plot.layout.shapes) return;
  // All shapes were created by the Python code;
  // we need to know how many are shade vs non-shade.
  // Since the Python code only generates shade shapes, NON_SHADE = 0.
  // But if the user added extra shapes, we track the total.
  const totalShapes = plot.layout.shapes.length;
  // shade shapes = shapes generated from SHADE_BANDS; count them
  let shadeShapeCount = 0;
  SHADE_BANDS.forEach(band => {{
    const col = SHADE_COLUMNS[band.col];
    if (!col) return;
    // count intervals
    let inInterval = false;
    let intervals = 0;
    for (let i = 0; i < col.length; i++) {{
      const v = col[i];
      const inRange = (v !== null && v >= band.low && v <= band.high);
      if (inRange && !inInterval) {{
        inInterval = true;
        intervals++;
      }} else if (!inRange) {{
        inInterval = false;
      }}
    }}
    shadeShapeCount += intervals;  // one rect per interval
    // hatch shapes: not counted individually here; we regenerate everything
  }});
  // Approximate: non-shade = total - everything from shades
  // Since hatch shapes are hard to count exactly, just mark all as shade
  NON_SHADE_SHAPE_COUNT = 0;
}})();

function computeShadeShapes() {{
  const shapes = [];
  SHADE_BANDS.forEach(band => {{
    if (band._disabled) return;
    const col = SHADE_COLUMNS[band.col];
    if (!col || col.length === 0) return;

    // Find contiguous intervals where value is in [low, high]
    const intervals = [];
    let start = -1;
    for (let i = 0; i < col.length; i++) {{
      const v = col[i];
      const inRange = (v !== null && v >= band.low && v <= band.high);
      if (inRange && start < 0) {{
        start = i;
      }} else if (!inRange && start >= 0) {{
        intervals.push([start, i - 1]);
        start = -1;
      }}
    }}
    if (start >= 0) intervals.push([start, col.length - 1]);

    intervals.forEach(([s, e]) => {{
      if (s >= SHADE_TIME_NS.length || e >= SHADE_TIME_NS.length) return;
      const x0ms = SHADE_TIME_NS[s] / 1e6;
      const x1ms = SHADE_TIME_NS[e] / 1e6;
      function msToISO(ms) {{ return new Date(ms).toISOString(); }}
      const x0 = msToISO(x0ms);
      const x1 = msToISO(x1ms);

      const lineDict = {{width: band.line_width || 0}};
      if (band.line_color) lineDict.color = band.line_color;

      const rect = {{
        type: "rect",
        xref: band.xref || "x",
        yref: band.yref || "paper",
        x0: x0,
        x1: x1,
        y0: band.y0 !== undefined ? band.y0 : 0,
        y1: band.y1 !== undefined ? band.y1 : 1,
        fillcolor: band.fillcolor || "rgba(128,128,128,0.18)",
        line: lineDict,
        layer: band.layer || "below",
      }};
      if (band.opacity !== null && band.opacity !== undefined) {{
        rect.opacity = band.opacity;
      }}
      shapes.push(rect);

      // Generate hatch lines if specified
      const hatchPattern = (band.hatch || "").trim().toLowerCase();
      if (hatchPattern && hatchPattern !== "none" && hatchPattern !== "off") {{
        const step = Math.min(Math.max(band.hatch_step || 0.08, 0.01), 0.5);
        const hColor = band.hatch_color || band.line_color || "rgba(80,80,80,0.45)";
        const hWidth = Math.max(band.hatch_width || 1, 0.1);
        const hOpacity = band.hatch_opacity;
        const y0h = band.y0 !== undefined ? band.y0 : 0;
        const y1h = band.y1 !== undefined ? band.y1 : 1;

        const BACKSLASH = String.fromCharCode(92);
        const addVert = ["|" ,"v", "vertical", "+", "grid"].includes(hatchPattern);
        const addHorz = ["-", "h", "horizontal", "+", "grid"].includes(hatchPattern);
        const addDiagUp = ["/", "diag", "diag_up", "x"].includes(hatchPattern);
        const addDiagDown = [BACKSLASH, "diag_down", "x"].includes(hatchPattern);


        const fracs = [];
        for (let f = 0; f <= 1.0 + step * 0.5; f += step) fracs.push(f);

        function xAt(frac) {{
          return msToISO(x0ms + frac * (x1ms - x0ms));
        }}
        function mkLine(xa, ya, xb, yb) {{
          const ln = {{
            type: "line", xref: band.xref || "x", yref: band.yref || "paper",
            x0: xa, y0: ya, x1: xb, y1: yb,
            line: {{color: hColor, width: hWidth}},
            layer: band.layer || "below",
          }};
          if (hOpacity !== null && hOpacity !== undefined) ln.opacity = hOpacity;
          return ln;
        }}

        if (addVert) fracs.forEach(f => shapes.push(mkLine(xAt(f), y0h, xAt(f), y1h)));
        if (addHorz) fracs.forEach(f => {{
          const y = y0h + f * (y1h - y0h);
          shapes.push(mkLine(x0, y, x1, y));
        }});

        if (addDiagUp || addDiagDown) {{
          const dOffsets = [];
          for (let d = -1.0; d <= 1.0 + step * 0.5; d += step) dOffsets.push(d);
          if (addDiagUp) dOffsets.forEach(d => {{
            const f0 = Math.max(0, d), f1 = Math.min(1, 1 + d);
            shapes.push(mkLine(xAt(f0), y0h, xAt(f1), y1h));
          }});
          if (addDiagDown) dOffsets.forEach(d => {{
            const f0 = Math.max(0, d), f1 = Math.min(1, 1 + d);
            shapes.push(mkLine(xAt(f0), y1h, xAt(f1), y0h));
          }});
        }}
      }}
    }});
  }});
  return shapes;
}}

function refreshShades() {{
  const shapes = computeShadeShapes();
  relayout({{shapes: shapes}});
}}

function buildShadeControls() {{
  if (!ADD_SHADE_CONTROLS) return;
  const container = document.getElementById("shade-controls");
  if (!container) return;
  if (SHADE_BANDS.length === 0) {{
    container.innerHTML = "<div class='small'>No shade bands configured.</div>";
    return;
  }}

  SHADE_BANDS.forEach((band, idx) => {{
    band._disabled = !!band.initially_hidden;
    const block = document.createElement("div");
    block.className = "trace-block";

    const title = document.createElement("div");
    title.className = "trace-title";
    title.textContent = (band.key || "shade") + " / " + band.col;
    block.appendChild(title);

    // Enabled checkbox
    const enabledInput = document.createElement("input");
    enabledInput.type = "checkbox";
    enabledInput.checked = !band._disabled;
    enabledInput.addEventListener("change", () => {{
      band._disabled = !enabledInput.checked;
      refreshShades();
    }});
    block.appendChild(makeControlRow("enabled", enabledInput));

    // Column dropdown
    const colSelect = document.createElement("select");
    SHADE_COL_NAMES.forEach(cn => {{
      const opt = document.createElement("option");
      opt.value = cn;
      opt.textContent = cn;
      if (cn === band.col) opt.selected = true;
      colSelect.appendChild(opt);
    }});
    colSelect.addEventListener("change", () => {{
      band.col = colSelect.value;
      title.textContent = (band.key || "shade") + " / " + band.col;
      refreshShades();
    }});
    block.appendChild(makeControlRow("column", colSelect));

    // Range min
    const rMinInput = document.createElement("input");
    rMinInput.type = "number";
    rMinInput.step = "any";
    rMinInput.value = band.low;
    rMinInput.addEventListener("change", () => {{
      band.low = parseFloat(rMinInput.value);
      refreshShades();
    }});
    block.appendChild(makeControlRow("range min", rMinInput));

    // Range max
    const rMaxInput = document.createElement("input");
    rMaxInput.type = "number";
    rMaxInput.step = "any";
    rMaxInput.value = band.high;
    rMaxInput.addEventListener("change", () => {{
      band.high = parseFloat(rMaxInput.value);
      refreshShades();
    }});
    block.appendChild(makeControlRow("range max", rMaxInput));

    // Fill color
    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = normalizeColor(band.fillcolor || "#808080");
    colorInput.addEventListener("input", () => {{
      band.fillcolor = colorInput.value;
      refreshShades();
    }});
    block.appendChild(makeControlRow("fill color", colorInput));

    // Opacity
    const opacityInput = document.createElement("input");
    opacityInput.type = "number";
    opacityInput.min = "0";
    opacityInput.max = "1";
    opacityInput.step = "0.05";
    opacityInput.value = (band.opacity !== null && band.opacity !== undefined) ? band.opacity : 0.18;
    opacityInput.addEventListener("input", () => {{
      band.opacity = parseFloat(opacityInput.value);
      refreshShades();
    }});
    block.appendChild(makeControlRow("opacity", opacityInput));

    // Hatch pattern
    const hatchSelect = document.createElement("select");
    const BS = String.fromCharCode(92);
    ["none", "/", BS, "|", "-", "+", "x"].forEach(v => {{
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v === "none" ? "none" : v;
      if ((band.hatch || "none") === v) opt.selected = true;
      hatchSelect.appendChild(opt);
    }});
    hatchSelect.addEventListener("change", () => {{
      band.hatch = hatchSelect.value === "none" ? null : hatchSelect.value;
      refreshShades();
    }});
    block.appendChild(makeControlRow("hatch", hatchSelect));

    container.appendChild(block);
  }});
}}

buildLayoutControls();
buildAxisControls();
buildTraceControls();
buildShadeControls();
if (SHADE_BANDS.some(b => b.initially_hidden)) refreshShades();

// Sync sidebar range inputs on any relayout (zoom, double-click reset, etc.)
const _plotEl = document.getElementById(PLOT_ID);
if (_plotEl) {{
  _plotEl.on("plotly_relayout", function(ed) {{
    AXES.forEach(axis => {{
      if (!axis._minInput || !axis._maxInput) return;
      const rKey = axis.layout_key + ".range";
      const arKey = axis.layout_key + ".autorange";
      if (ed[rKey]) {{
        axis._minInput.value = ed[rKey][0];
        axis._maxInput.value = ed[rKey][1];
      }} else if (ed[arKey] || ed["autosize"]) {{
        // autorange or reset: read back after Plotly finishes
        setTimeout(() => {{
          const la = _plotEl.layout[axis.layout_key];
          if (la && la.range) {{
            axis._minInput.value = la.range[0];
            axis._maxInput.value = la.range[1];
          }}
        }}, 50);
      }}
    }});
  }});
}}

// ---- Drag-to-resize ----
(function initResize() {{
  const handle = document.getElementById("resize-handle");
  if (!handle) return;
  const plot = document.getElementById(PLOT_ID);
  if (!plot) return;
  let dragging = false;
  let startX, startY, startW, startH;

  handle.addEventListener("mousedown", (e) => {{
    e.preventDefault();
    dragging = true;
    startX = e.clientX;
    startY = e.clientY;
    startW = plot.offsetWidth;
    startH = plot.offsetHeight;
  }});

  document.addEventListener("mousemove", (e) => {{
    if (!dragging) return;
    const newW = Math.max(400, startW + (e.clientX - startX));
    const newH = Math.max(300, startH + (e.clientY - startY));
    Plotly.relayout(plot, {{width: newW, height: newH}});
  }});

  document.addEventListener("mouseup", () => {{
    dragging = false;
  }});
}})();

// ---- Sidebar splitter ----
(function initSplitter() {{
  const splitter = document.getElementById("sidebar-splitter");
  const sidebar = document.querySelector(".controls");
  if (!splitter || !sidebar) return;
  let dragging = false;
  let startX, startW;

  splitter.addEventListener("mousedown", (e) => {{
    e.preventDefault();
    dragging = true;
    startX = e.clientX;
    startW = sidebar.offsetWidth;
    splitter.classList.add("active");
    document.body.style.userSelect = "none";
  }});

  document.addEventListener("mousemove", (e) => {{
    if (!dragging) return;
    const newW = Math.min(600, Math.max(200, startW + (e.clientX - startX)));
    sidebar.style.width = newW + "px";
    sidebar.style.minWidth = newW + "px";
  }});

  document.addEventListener("mouseup", () => {{
    if (!dragging) return;
    dragging = false;
    splitter.classList.remove("active");
    document.body.style.userSelect = "";
  }});
}})();
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path
