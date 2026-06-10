"""Two squares on a triangle and the midpoint of the third side."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt

ComplexArray = npt.NDArray[np.complexfloating]


def _as_triangle_vertices(vertices: Sequence[complex] | npt.ArrayLike) -> ComplexArray:
    """Return a three-point complex array with the first point fixed at zero."""
    points = np.asarray(vertices, dtype=complex)
    if points.shape != (3,):
        raise ValueError("vertices must contain exactly three complex points")

    points = points.copy()
    points[0] = 0 + 0j
    return points


def default_triangle_vertices() -> ComplexArray:
    """Return a sample triangle with one vertex at the origin."""
    return np.asarray([0 + 0j, 0.18 + 1.468j, 2.41 + 1.746j], dtype=complex)


def triangle_half_sides(vertices: Sequence[complex] | npt.ArrayLike) -> tuple[complex, complex]:
    """Return ``a`` and ``b`` when the first two triangle sides are ``2a`` and ``2b``."""
    points = _as_triangle_vertices(vertices)
    return (points[1] - points[0]) / 2, (points[2] - points[1]) / 2


def triangle_square_vertices(vertices: Sequence[complex] | npt.ArrayLike) -> ComplexArray:
    """Return closed paths for the two left-facing squares on sides ``2a`` and ``2b``."""
    points = _as_triangle_vertices(vertices)
    squares = []
    for tail, head in [(points[0], points[1]), (points[1], points[2])]:
        side = head - tail
        offset = 1j * side
        squares.append([tail, head, head + offset, tail + offset, tail])

    return np.asarray(squares, dtype=complex)


def triangle_square_centers(vertices: Sequence[complex] | npt.ArrayLike) -> tuple[complex, complex]:
    """Return the centers ``p`` and ``s`` of the two side squares."""
    points = _as_triangle_vertices(vertices)
    a, b = triangle_half_sides(points)
    return a + 1j * a, 2 * a + b + 1j * b


def triangle_third_side_midpoint(vertices: Sequence[complex] | npt.ArrayLike) -> complex:
    """Return the midpoint ``m`` of the third side from ``0`` to ``2a + 2b``."""
    points = _as_triangle_vertices(vertices)
    return (points[0] + points[2]) / 2


def make_triangle_midpoint_bokeh(
    vertices: Sequence[complex] | npt.ArrayLike | None = None,
):
    """Build a draggable Bokeh visualization for the triangle midpoint picture."""
    try:
        from bokeh.layouts import column, row
        from bokeh.models import ColumnDataSource, CustomJS, Div, LabelSet, PointDrawTool, Range1d
        from bokeh.plotting import figure
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Bokeh is required for this interactive visualization. "
            'Install it with `pip install -e ".[dev]"`.'
        ) from exc

    points = _as_triangle_vertices(vertices if vertices is not None else default_triangle_vertices())
    squares = triangle_square_vertices(points)
    p_center, s_center = triangle_square_centers(points)
    midpoint = triangle_third_side_midpoint(points)

    vertex_source = ColumnDataSource(
        data=dict(
            x=[points[1].real, points[2].real],
            y=[points[1].imag, points[2].imag],
            label=["2a", "2a+2b"],
        )
    )
    origin_source = ColumnDataSource(data=dict(x=[0.0], y=[0.0], label=["0"]))
    triangle_source = ColumnDataSource(
        data=dict(
            x=[point.real for point in [points[0], points[1], points[2], points[0]]],
            y=[point.imag for point in [points[0], points[1], points[2], points[0]]],
        )
    )
    square_source = ColumnDataSource(
        data=dict(
            xs=[[point.real for point in path] for path in squares],
            ys=[[point.imag for point in path] for path in squares],
        )
    )
    center_source = ColumnDataSource(
        data=dict(
            x=[p_center.real, s_center.real, midpoint.real],
            y=[p_center.imag, s_center.imag, midpoint.imag],
            label=["p", "s", "m"],
            color=["#7c3aed", "#7c3aed", "#111827"],
            size=[9, 9, 8],
        )
    )
    segment_source = ColumnDataSource(
        data=dict(
            xs=[[p_center.real, midpoint.real], [s_center.real, midpoint.real]],
            ys=[[p_center.imag, midpoint.imag], [s_center.imag, midpoint.imag]],
        )
    )
    side_label_source = ColumnDataSource(
        data=dict(
            x=[(points[0].real + points[1].real) / 2, (points[1].real + points[2].real) / 2],
            y=[(points[0].imag + points[1].imag) / 2, (points[1].imag + points[2].imag) / 2],
            label=["2a", "2b"],
        )
    )
    summary = Div(
        text=_format_summary(points, p_center, s_center, midpoint),
        width=310,
        styles={
            "font-family": "JetBrains Mono, Menlo, Consolas, monospace",
            "font-size": "13px",
            "line-height": "1.55",
            "border": "1px solid #d8dee9",
            "border-radius": "6px",
            "padding": "12px",
            "background": "#fbfbfc",
        },
    )

    plot = figure(
        width=720,
        height=720,
        x_range=Range1d(-2.5, 5.5, bounds=(-8.0, 8.0)),
        y_range=Range1d(-1.5, 6.5, bounds=(-8.0, 8.0)),
        x_axis_label="Re",
        y_axis_label="Im",
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
        toolbar_location="above",
        title="Two side squares and the midpoint of the third side",
    )
    plot.grid.grid_line_alpha = 0.35
    plot.xaxis.axis_label_text_font_style = "normal"
    plot.yaxis.axis_label_text_font_style = "normal"

    plot.patches(
        xs="xs",
        ys="ys",
        source=square_source,
        line_color="#4c78a8",
        line_width=2,
        fill_color="#dbeafe",
        fill_alpha=0.35,
    )
    plot.line(x="x", y="y", source=triangle_source, line_color="#111827", line_width=3)
    plot.multi_line(
        xs="xs",
        ys="ys",
        source=segment_source,
        line_color="#d97706",
        line_width=3,
    )
    plot.scatter(
        x="x",
        y="y",
        source=center_source,
        size="size",
        color="color",
        alpha=0.95,
    )
    plot.scatter(x="x", y="y", source=origin_source, size=10, color="#111827")
    plot.add_layout(
        LabelSet(
            x="x",
            y="y",
            text="label",
            source=center_source,
            x_offset=7,
            y_offset=7,
            text_font_size="14px",
            text_color="#4c1d95",
        )
    )
    plot.add_layout(
        LabelSet(
            x="x",
            y="y",
            text="label",
            source=side_label_source,
            x_offset=6,
            y_offset=6,
            text_font_size="13px",
            text_color="#374151",
        )
    )
    plot.add_layout(
        LabelSet(
            x="x",
            y="y",
            text="label",
            source=origin_source,
            x_offset=-16,
            y_offset=-18,
            text_font_size="13px",
            text_color="#111827",
        )
    )
    plot.scatter(
        x="x",
        y="y",
        source=vertex_source,
        size=30,
        color="#ef4444",
        alpha=0.82,
        line_color="#991b1b",
        line_width=2,
    )
    vertex_handle_renderer = plot.scatter(
        x="x",
        y="y",
        source=vertex_source,
        marker="square",
        size=42,
        fill_color="#ef4444",
        fill_alpha=0.12,
        line_alpha=0.0,
    )

    draw_tool = PointDrawTool(renderers=[vertex_handle_renderer], add=False)
    plot.add_tools(draw_tool)
    plot.toolbar.active_tap = draw_tool

    callback = CustomJS(
        args=dict(
            vertex_source=vertex_source,
            triangle_source=triangle_source,
            square_source=square_source,
            center_source=center_source,
            segment_source=segment_source,
            side_label_source=side_label_source,
            summary=summary,
        ),
        code=_BOKEH_UPDATE_JS,
    )
    vertex_source.js_on_change("data", callback)

    return row(plot, column(summary, width=330), sizing_mode="stretch_width")


def export_triangle_midpoint_html(
    path: str | Path = "docs/assets/plots/triangle_midpoint.html",
    vertices: Sequence[complex] | npt.ArrayLike | None = None,
) -> Path:
    """Write a standalone Bokeh HTML file for the triangle midpoint example."""
    try:
        from bokeh.io import save
        from bokeh.resources import INLINE
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Bokeh is required to export this interactive visualization. "
            'Install it with `pip install -e ".[dev]"`.'
        ) from exc

    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    save(
        make_triangle_midpoint_bokeh(vertices),
        filename=str(out),
        resources=INLINE,
        title="Triangle side squares and midpoint",
    )
    return out


def _format_complex(z: complex) -> str:
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.3f} {sign} {abs(z.imag):.3f}i"


def _format_summary(points: ComplexArray, p_center: complex, s_center: complex, midpoint: complex) -> str:
    a, b = triangle_half_sides(points)
    pm = midpoint - p_center
    sm = midpoint - s_center
    midpoint_identity = pm + 1j * (s_center - midpoint)
    return f"""
<b>Complex summary</b><br>
a = {_format_complex(a)}<br>
b = {_format_complex(b)}<br>
p = a + ia = {_format_complex(p_center)}<br>
s = 2a + b + ib = {_format_complex(s_center)}<br>
m = a + b = {_format_complex(midpoint)}<br>
m - p = {_format_complex(pm)}<br>
m - s = {_format_complex(sm)}<br>
(m - p) + i(s - m) = {_format_complex(midpoint_identity)}
"""


_BOKEH_UPDATE_JS = r"""
const z = [
  {re: 0, im: 0},
  {re: vertex_source.data.x[0], im: vertex_source.data.y[0]},
  {re: vertex_source.data.x[1], im: vertex_source.data.y[1]},
];

function c(re, im) {
  return {re: re, im: im};
}
function add(u, v) {
  return c(u.re + v.re, u.im + v.im);
}
function sub(u, v) {
  return c(u.re - v.re, u.im - v.im);
}
function mulReal(u, r) {
  return c(u.re * r, u.im * r);
}
function mulI(u) {
  return c(-u.im, u.re);
}
function fmt(u) {
  const sign = u.im >= 0 ? " + " : " - ";
  return `${u.re.toFixed(3)}${sign}${Math.abs(u.im).toFixed(3)}i`;
}

const a = mulReal(sub(z[1], z[0]), 0.5);
const b = mulReal(sub(z[2], z[1]), 0.5);
const p = add(a, mulI(a));
const s = add(add(mulReal(a, 2), b), mulI(b));
const m = mulReal(add(z[0], z[2]), 0.5);

triangle_source.data = {
  x: [z[0].re, z[1].re, z[2].re, z[0].re],
  y: [z[0].im, z[1].im, z[2].im, z[0].im],
};

const squareXs = [];
const squareYs = [];
for (const [tail, head] of [[z[0], z[1]], [z[1], z[2]]]) {
  const side = sub(head, tail);
  const offset = mulI(side);
  const p0 = tail;
  const p1 = head;
  const p2 = add(head, offset);
  const p3 = add(tail, offset);
  squareXs.push([p0.re, p1.re, p2.re, p3.re, p0.re]);
  squareYs.push([p0.im, p1.im, p2.im, p3.im, p0.im]);
}
square_source.data = {xs: squareXs, ys: squareYs};

center_source.data = {
  x: [p.re, s.re, m.re],
  y: [p.im, s.im, m.im],
  label: ["p", "s", "m"],
  color: ["#7c3aed", "#7c3aed", "#111827"],
  size: [9, 9, 8],
};
segment_source.data = {
  xs: [[p.re, m.re], [s.re, m.re]],
  ys: [[p.im, m.im], [s.im, m.im]],
};
side_label_source.data = {
  x: [(z[0].re + z[1].re) / 2, (z[1].re + z[2].re) / 2],
  y: [(z[0].im + z[1].im) / 2, (z[1].im + z[2].im) / 2],
  label: ["2a", "2b"],
};

summary.text = `
<b>Complex summary</b><br>
a = ${fmt(a)}<br>
b = ${fmt(b)}<br>
p = a + ia = ${fmt(p)}<br>
s = 2a + b + ib = ${fmt(s)}<br>
m = a + b = ${fmt(m)}<br>
m - p = ${fmt(sub(m, p))}<br>
m - s = ${fmt(sub(m, s))}<br>
(m - p) + i(s - m) = ${fmt(add(sub(m, p), mulI(sub(s, m))))}
`;
"""
