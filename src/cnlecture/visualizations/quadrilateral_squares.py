"""Squares on the sides of a quadrilateral.

The quadrilateral is written in Needham's convenient half-side notation:

    z0 = 0
    z1 = 2a
    z2 = 2a + 2b
    z3 = 2a + 2b + 2c
    z0 = z3 + 2d

so that ``a + b + c + d = 0``.  If squares are built to the left of the
oriented sides, their centers are

    a + ia,  2a + b + ib,  2a + 2b + c + ic,  -d + id.

The two segments connecting opposite square centers satisfy ``A + iB = 0``
with the orientation convention used below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt

ComplexArray = npt.NDArray[np.complexfloating]


def _as_vertices(vertices: Sequence[complex] | npt.ArrayLike) -> ComplexArray:
    """Return a four-point complex array with the first point fixed at zero."""
    points = np.asarray(vertices, dtype=complex)
    if points.shape != (4,):
        raise ValueError("vertices must contain exactly four complex points")

    points = points.copy()
    points[0] = 0 + 0j
    return points


def quadrilateral_half_sides(vertices: Sequence[complex] | npt.ArrayLike) -> ComplexArray:
    """Return the half-side complex numbers ``a, b, c, d``.

    The full oriented sides of the quadrilateral are ``2a, 2b, 2c, 2d``.
    The returned values always satisfy ``a + b + c + d == 0`` up to floating
    point roundoff.
    """
    points = _as_vertices(vertices)
    closed = np.append(points, points[0])
    return np.diff(closed) / 2


def square_vertices(vertices: Sequence[complex] | npt.ArrayLike) -> ComplexArray:
    """Return closed vertex paths for the four left-facing side squares.

    The result has shape ``(4, 5)``.  Each row contains the closed path of one
    square, starting with the corresponding quadrilateral side.
    """
    points = _as_vertices(vertices)
    halves = quadrilateral_half_sides(points)

    squares = []
    for tail, half_side in zip(points, halves):
        full_side = 2 * half_side
        offset = 1j * full_side
        squares.append(
            [
                tail,
                tail + full_side,
                tail + full_side + offset,
                tail + offset,
                tail,
            ]
        )

    return np.asarray(squares, dtype=complex)


def square_centers(vertices: Sequence[complex] | npt.ArrayLike) -> ComplexArray:
    """Return the centers of the four left-facing side squares."""
    points = _as_vertices(vertices)
    halves = quadrilateral_half_sides(points)
    return points + halves + 1j * halves


def opposite_center_segments(vertices: Sequence[complex] | npt.ArrayLike) -> tuple[complex, complex]:
    """Return the oriented opposite-center segments ``A`` and ``B``.

    ``A`` runs from the center of the square on side ``2a`` to the center of
    the square on side ``2c``.  ``B`` runs from the center of the square on
    side ``2d`` to the center of the square on side ``2b``.  With these
    orientations, ``A + iB = 0``.
    """
    centers = square_centers(vertices)
    a_segment = centers[2] - centers[0]
    b_segment = centers[1] - centers[3]
    return a_segment, b_segment


def opposite_center_identity(vertices: Sequence[complex] | npt.ArrayLike) -> complex:
    """Return ``A + iB`` for the opposite square-center segments."""
    a_segment, b_segment = opposite_center_segments(vertices)
    return a_segment + 1j * b_segment


def default_quadrilateral_vertices() -> ComplexArray:
    """Return a clockwise sample quadrilateral with one vertex at the origin."""
    return np.asarray(
        [
            0 + 0j,
            -0.7 + 1.3j,
            0.75 + 2.78j,
            1.594 + 1.538j,
        ],
        dtype=complex,
    )


def make_quadrilateral_squares_bokeh(
    vertices: Sequence[complex] | npt.ArrayLike | None = None,
    show_connectors: bool = True,
    show_diagonal: bool = False,
):
    """Build a draggable Bokeh visualization of squares on a quadrilateral.

    The first quadrilateral vertex is fixed at the origin.  The other three
    vertices can be dragged in the browser; the squares, centers, center
    connectors, and complex-number summary update immediately.

    Parameters
    ----------
    vertices:
        Four complex vertices.  The first is forced to ``0``.
    show_connectors:
        Whether to show the two segments connecting opposite square centers.
    show_diagonal:
        Whether to show the quadrilateral diagonal from ``0`` to ``2a + 2b``.

    Returns
    -------
    Bokeh layout object
        A standalone-saveable Bokeh layout.
    """
    try:
        from bokeh.layouts import column, row
        from bokeh.models import (
            CheckboxGroup,
            ColumnDataSource,
            CustomJS,
            Div,
            LabelSet,
            PointDrawTool,
            Range1d,
        )
        from bokeh.plotting import figure
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Bokeh is required for this interactive visualization. "
            'Install it with `pip install -e ".[dev]"`.'
        ) from exc

    points = _as_vertices(vertices if vertices is not None else default_quadrilateral_vertices())
    halves = quadrilateral_half_sides(points)
    squares = square_vertices(points)
    centers = square_centers(points)
    a_segment, b_segment = opposite_center_segments(points)

    vertex_source = ColumnDataSource(
        data=dict(
            x=[point.real for point in points[1:]],
            y=[point.imag for point in points[1:]],
            label=["z1", "z2", "z3"],
        )
    )
    origin_source = ColumnDataSource(data=dict(x=[0.0], y=[0.0], label=["0"]))
    quad_source = ColumnDataSource(
        data=dict(
            x=[point.real for point in np.append(points, points[0])],
            y=[point.imag for point in np.append(points, points[0])],
        )
    )
    square_source = ColumnDataSource(
        data=dict(
            xs=[[point.real for point in path] for path in squares],
            ys=[[point.imag for point in path] for path in squares],
            name=["square on 2a", "square on 2b", "square on 2c", "square on 2d"],
        )
    )
    center_source = ColumnDataSource(
        data=dict(
            x=[center.real for center in centers],
            y=[center.imag for center in centers],
            label=["a+ia", "2a+b+ib", "2a+2b+c+ic", "-d+id"],
        )
    )
    connector_source = ColumnDataSource(
        data=dict(
            xs=[[centers[0].real, centers[2].real], [centers[3].real, centers[1].real]],
            ys=[[centers[0].imag, centers[2].imag], [centers[3].imag, centers[1].imag]],
            name=["A", "B"],
            color=["#d97706", "#d97706"],
            dash=["solid", "solid"],
        )
    )
    diagonal_source = ColumnDataSource(
        data=dict(
            x=[points[0].real, points[2].real],
            y=[points[0].imag, points[2].imag],
        )
    )
    side_label_source = ColumnDataSource(
        data=dict(
            x=[(points[k].real + points[(k + 1) % 4].real) / 2 for k in range(4)],
            y=[(points[k].imag + points[(k + 1) % 4].imag) / 2 for k in range(4)],
            label=["2a", "2b", "2c", "2d"],
        )
    )
    summary = Div(
        text=_format_summary(halves, a_segment, b_segment),
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
        x_range=Range1d(-3.5, 4.5, bounds=(-8.0, 8.0)),
        y_range=Range1d(-2.0, 6.0, bounds=(-8.0, 8.0)),
        x_axis_label="Re",
        y_axis_label="Im",
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
        toolbar_location="above",
        title="Squares on the sides of a quadrilateral",
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
    plot.line(
        x="x",
        y="y",
        source=quad_source,
        line_color="#111827",
        line_width=3,
    )
    connector_renderer = plot.multi_line(
        xs="xs",
        ys="ys",
        source=connector_source,
        line_color="color",
        line_width=3,
        line_dash="dash",
        visible=show_connectors,
    )
    diagonal_renderer = plot.line(
        x="x",
        y="y",
        source=diagonal_source,
        line_color="#6b7280",
        line_width=2,
        line_dash="dashed",
        visible=show_diagonal,
    )
    plot.scatter(
        x="x",
        y="y",
        source=center_source,
        size=8,
        color="#7c3aed",
        alpha=0.9,
    )
    plot.scatter(
        x="x",
        y="y",
        source=origin_source,
        size=10,
        color="#111827",
    )
    plot.scatter(
        x="x",
        y="y",
        source=vertex_source,
        size=22,
        color="#ef4444",
        alpha=0.82,
        line_color="#991b1b",
        line_width=1.5,
    )
    vertex_handle_renderer = plot.scatter(
        x="x",
        y="y",
        source=vertex_source,
        marker="square",
        size=34,
        fill_color="#ef4444",
        fill_alpha=0.12,
        line_alpha=0.0,
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
            source=center_source,
            x_offset=7,
            y_offset=-7,
            text_font_size="12px",
            text_color="#5b21b6",
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

    draw_tool = PointDrawTool(renderers=[vertex_handle_renderer], add=False)
    plot.add_tools(draw_tool)
    plot.toolbar.active_tap = draw_tool

    checkbox = CheckboxGroup(
        labels=[
            "Show opposite-center segments A and B",
            "Show diagonal 0 to 2a+2b",
        ],
        active=([0] if show_connectors else []) + ([1] if show_diagonal else []),
        width=310,
    )

    callback = CustomJS(
        args=dict(
            vertex_source=vertex_source,
            quad_source=quad_source,
            square_source=square_source,
            center_source=center_source,
            connector_source=connector_source,
            diagonal_source=diagonal_source,
            side_label_source=side_label_source,
            summary=summary,
            checkbox=checkbox,
            connector_renderer=connector_renderer,
            diagonal_renderer=diagonal_renderer,
        ),
        code=_BOKEH_UPDATE_JS,
    )
    vertex_source.js_on_change("data", callback)
    checkbox.js_on_change("active", callback)

    controls = column(checkbox, summary, width=330)
    return row(plot, controls, sizing_mode="stretch_width")


def export_quadrilateral_squares_html(
    path: str | Path = "docs/assets/plots/quadrilateral_squares.html",
    vertices: Sequence[complex] | npt.ArrayLike | None = None,
    show_connectors: bool = True,
    show_diagonal: bool = False,
) -> Path:
    """Write a standalone Bokeh HTML file for the quadrilateral example."""
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
    layout = make_quadrilateral_squares_bokeh(
        vertices,
        show_connectors=show_connectors,
        show_diagonal=show_diagonal,
    )
    save(
        layout,
        filename=str(out),
        resources=INLINE,
        title="Squares on the sides of a quadrilateral",
    )
    return out


def _format_complex(z: complex) -> str:
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.3f} {sign} {abs(z.imag):.3f}i"


def _format_summary(halves: ComplexArray, a_segment: complex, b_segment: complex) -> str:
    side_sum = np.sum(halves)
    identity = a_segment + 1j * b_segment
    angle = np.degrees(np.angle(a_segment / b_segment)) if abs(b_segment) > 0 else np.nan
    return f"""
<b>Complex summary</b><br>
a + b + c + d = {_format_complex(side_sum)}<br>
A = {_format_complex(a_segment)}<br>
B = {_format_complex(b_segment)}<br>
A + iB = {_format_complex(identity)}<br>
|A| = {abs(a_segment):.3f}, |B| = {abs(b_segment):.3f}<br>
arg(A/B) = {angle:.1f}&deg;
"""


_BOKEH_UPDATE_JS = r"""
const xs = [0].concat(vertex_source.data.x);
const ys = [0].concat(vertex_source.data.y);

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
function absC(u) {
  return Math.hypot(u.re, u.im);
}
function divC(u, v) {
  const den = v.re * v.re + v.im * v.im;
  return c((u.re * v.re + u.im * v.im) / den, (u.im * v.re - u.re * v.im) / den);
}
function fmt(u) {
  const sign = u.im >= 0 ? " + " : " - ";
  return `${u.re.toFixed(3)}${sign}${Math.abs(u.im).toFixed(3)}i`;
}

const z = xs.map((x, i) => c(x, ys[i]));
const halves = [];
for (let k = 0; k < 4; k += 1) {
  halves.push(mulReal(sub(z[(k + 1) % 4], z[k]), 0.5));
}

const qx = xs.concat([0]);
const qy = ys.concat([0]);
quad_source.data = {x: qx, y: qy};

const squareXs = [];
const squareYs = [];
const centers = [];
for (let k = 0; k < 4; k += 1) {
  const tail = z[k];
  const full = mulReal(halves[k], 2);
  const offset = mulI(full);
  const p0 = tail;
  const p1 = add(tail, full);
  const p2 = add(p1, offset);
  const p3 = add(tail, offset);
  squareXs.push([p0.re, p1.re, p2.re, p3.re, p0.re]);
  squareYs.push([p0.im, p1.im, p2.im, p3.im, p0.im]);
  centers.push(add(tail, add(halves[k], mulI(halves[k]))));
}
square_source.data = {
  xs: squareXs,
  ys: squareYs,
  name: ["square on 2a", "square on 2b", "square on 2c", "square on 2d"],
};

center_source.data = {
  x: centers.map((p) => p.re),
  y: centers.map((p) => p.im),
  label: ["a+ia", "2a+b+ib", "2a+2b+c+ic", "-d+id"],
};
connector_source.data = {
  xs: [[centers[0].re, centers[2].re], [centers[3].re, centers[1].re]],
  ys: [[centers[0].im, centers[2].im], [centers[3].im, centers[1].im]],
  name: ["A", "B"],
  color: ["#d97706", "#d97706"],
  dash: ["solid", "solid"],
};
diagonal_source.data = {
  x: [z[0].re, z[2].re],
  y: [z[0].im, z[2].im],
};
side_label_source.data = {
  x: [0, 1, 2, 3].map((k) => (z[k].re + z[(k + 1) % 4].re) / 2),
  y: [0, 1, 2, 3].map((k) => (z[k].im + z[(k + 1) % 4].im) / 2),
  label: ["2a", "2b", "2c", "2d"],
};

const sideSum = halves.reduce((acc, value) => add(acc, value), c(0, 0));
const A = sub(centers[2], centers[0]);
const B = sub(centers[1], centers[3]);
const identity = add(A, mulI(B));
let angle = NaN;
if (absC(B) > 0) {
  const ratio = divC(A, B);
  angle = Math.atan2(ratio.im, ratio.re) * 180 / Math.PI;
}
summary.text = `
<b>Complex summary</b><br>
a + b + c + d = ${fmt(sideSum)}<br>
A = ${fmt(A)}<br>
B = ${fmt(B)}<br>
A + iB = ${fmt(identity)}<br>
|A| = ${absC(A).toFixed(3)}, |B| = ${absC(B).toFixed(3)}<br>
arg(A/B) = ${angle.toFixed(1)}&deg;
`;
connector_renderer.visible = checkbox.active.includes(0);
diagonal_renderer.visible = checkbox.active.includes(1);
"""
