"""Locate a moved point from its distances to a moved triangle."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt

ComplexArray = npt.NDArray[np.complexfloating]

SOURCE_COLOR = "#111827"
TARGET_COLOR = "#0072b2"
CIRCLE_COLORS = ["#d55e00", "#009e73", "#cc79a7"]
CANDIDATE_COLOR = "#e69f00"
POINT_COLOR = "#7c3aed"


@dataclass(frozen=True)
class LocatorGeometry:
    """Geometry for the triangle distance locator construction."""

    source_triangle: ComplexArray
    source_point: complex
    target_triangle: ComplexArray
    target_point: complex
    radii: npt.NDArray[np.floating]
    candidates: ComplexArray
    candidate_residuals: npt.NDArray[np.floating]


def _as_triangle(vertices: Sequence[complex] | npt.ArrayLike) -> ComplexArray:
    points = np.asarray(vertices, dtype=complex)
    if points.shape != (3,):
        raise ValueError("vertices must contain exactly three complex points")

    if abs(oriented_triangle_area(points)) < 1e-10:
        raise ValueError("triangle vertices must be non-collinear")

    return points.copy()


def default_locator_triangle() -> ComplexArray:
    """Return a sample non-collinear reference triangle."""
    return np.asarray([-0.75 - 0.45j, 1.05 - 0.25j, -0.05 + 1.15j], dtype=complex)


def default_locator_point() -> complex:
    """Return a sample point located by distances to the reference triangle."""
    return 0.35 + 0.32j


def oriented_triangle_area(vertices: Sequence[complex] | npt.ArrayLike) -> float:
    """Return the signed area of a complex triangle."""
    points = np.asarray(vertices, dtype=complex)
    if points.shape != (3,):
        raise ValueError("vertices must contain exactly three complex points")

    return 0.5 * float(np.imag(np.conj(points[1] - points[0]) * (points[2] - points[0])))


def apply_direct_motion(
    points: Sequence[complex] | npt.ArrayLike,
    angle_degrees: float,
    translation: complex,
) -> ComplexArray:
    """Apply the direct motion ``z -> exp(i theta) z + translation``."""
    zs = np.asarray(points, dtype=complex)
    angle = math.radians(float(angle_degrees))
    rotation = complex(math.cos(angle), math.sin(angle))
    return rotation * zs + complex(translation)


def locator_radii(
    vertices: Sequence[complex] | npt.ArrayLike,
    point: complex,
) -> npt.NDArray[np.floating]:
    """Return distances from ``point`` to the three triangle vertices."""
    triangle = _as_triangle(vertices)
    return np.abs(complex(point) - triangle)


def circle_intersections(
    center_a: complex,
    radius_a: float,
    center_b: complex,
    radius_b: float,
    *,
    tol: float = 1e-10,
) -> ComplexArray:
    """Return the intersection points of two circles."""
    c0 = complex(center_a)
    c1 = complex(center_b)
    r0 = float(radius_a)
    r1 = float(radius_b)
    delta = c1 - c0
    distance = abs(delta)

    if distance < tol:
        return np.asarray([], dtype=complex)
    if distance > r0 + r1 + tol:
        return np.asarray([], dtype=complex)
    if distance < abs(r0 - r1) - tol:
        return np.asarray([], dtype=complex)

    along = (r0**2 - r1**2 + distance**2) / (2 * distance)
    height_squared = max(0.0, r0**2 - along**2)
    height = math.sqrt(height_squared)
    unit = delta / distance
    foot = c0 + along * unit
    perpendicular = 1j * unit

    if height <= tol:
        return np.asarray([foot], dtype=complex)
    return np.asarray([foot + height * perpendicular, foot - height * perpendicular], dtype=complex)


def locator_geometry(
    vertices: Sequence[complex] | npt.ArrayLike | None = None,
    point: complex | None = None,
    angle_degrees: float = 34.0,
    translation: complex = 2.1 + 0.45j,
) -> LocatorGeometry:
    """Return the target triangle, target point, and circle-intersection candidates."""
    source_triangle = _as_triangle(vertices if vertices is not None else default_locator_triangle())
    source_point = complex(default_locator_point() if point is None else point)
    target_triangle = apply_direct_motion(source_triangle, angle_degrees, translation)
    target_point = apply_direct_motion([source_point], angle_degrees, translation)[0]
    radii = locator_radii(source_triangle, source_point)
    candidates = circle_intersections(target_triangle[0], radii[0], target_triangle[1], radii[1])
    residuals = np.asarray([abs(abs(candidate - target_triangle[2]) - radii[2]) for candidate in candidates])
    order = np.argsort(residuals)

    return LocatorGeometry(
        source_triangle=source_triangle,
        source_point=source_point,
        target_triangle=target_triangle,
        target_point=target_point,
        radii=radii,
        candidates=candidates[order],
        candidate_residuals=residuals[order],
    )


def make_triangle_distance_locator_bokeh(
    vertices: Sequence[complex] | npt.ArrayLike | None = None,
    point: complex | None = None,
    angle_degrees: float = 34.0,
    translation: complex = 2.1 + 0.45j,
):
    """Build a Bokeh visualization for locating a moved point from three distances."""
    try:
        from bokeh.layouts import column, row
        from bokeh.models import (
            ColumnDataSource,
            CustomJS,
            Div,
            LabelSet,
            PointDrawTool,
            Range1d,
            Slider,
            Toggle,
        )
        from bokeh.plotting import figure
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Bokeh is required for this interactive visualization. "
            'Install it with `pip install -e ".[dev]"`.'
        ) from exc

    geometry = locator_geometry(vertices, point, angle_degrees, translation)
    sources = _bokeh_sources(geometry)

    source_triangle = ColumnDataSource(data=sources["source_triangle"])
    target_triangle = ColumnDataSource(data=sources["target_triangle"])
    source_point = ColumnDataSource(data=sources["source_point"])
    target_point = ColumnDataSource(data=sources["target_point"])
    source_labels = ColumnDataSource(data=sources["source_labels"])
    target_labels = ColumnDataSource(data=sources["target_labels"])
    circle_source = ColumnDataSource(data=sources["circles"])
    candidate_source = ColumnDataSource(data=sources["candidates"])
    source_distance_source = ColumnDataSource(data=sources["source_distances"])
    target_distance_source = ColumnDataSource(data=sources["target_distances"])
    motion_source = ColumnDataSource(data=sources["motion_lines"])
    zero_state = ColumnDataSource(
        data=dict(
            theta=[angle_degrees],
            tx=[translation.real],
            ty=[translation.imag],
        )
    )

    summary = Div(text=_format_summary(geometry), width=370, styles=_panel_styles())

    plot = figure(
        width=760,
        height=620,
        x_range=Range1d(-2.0, 4.4, bounds=(-5.5, 6.5)),
        y_range=Range1d(-2.1, 3.3, bounds=(-5.5, 5.5)),
        x_axis_label="Re",
        y_axis_label="Im",
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
        toolbar_location="above",
        title="Three distances determine the image point",
    )
    plot.grid.grid_line_alpha = 0.28
    plot.xaxis.axis_label_text_font_style = "normal"
    plot.yaxis.axis_label_text_font_style = "normal"

    circle_renderer = plot.patches(
        xs="xs",
        ys="ys",
        source=circle_source,
        fill_color="color",
        fill_alpha=0.10,
        line_alpha=0.0,
    )
    circle_renderer.level = "underlay"
    plot.multi_line(xs="xs", ys="ys", source=motion_source, line_color="#9ca3af", line_dash="dashed", line_width=1.4)
    plot.line(x="x", y="y", source=source_triangle, line_color=SOURCE_COLOR, line_width=3)
    plot.line(x="x", y="y", source=target_triangle, line_color=TARGET_COLOR, line_width=3)
    plot.multi_line(
        xs="xs",
        ys="ys",
        source=source_distance_source,
        line_color="color",
        line_width=3,
        line_alpha=0.9,
    )
    plot.multi_line(
        xs="xs",
        ys="ys",
        source=target_distance_source,
        line_color="color",
        line_width=3,
        line_alpha=0.9,
    )
    plot.scatter(x="x", y="y", source=candidate_source, size="size", fill_color="#ffffff", line_color="color", line_width=2)
    plot.scatter(x="x", y="y", source=target_point, size=12, fill_color="#ffffff", line_color=TARGET_COLOR, line_width=2.4)
    plot.scatter(x="x", y="y", source=source_point, size=15, fill_color=POINT_COLOR, line_color="#4c1d95", line_width=2)
    point_handle = plot.scatter(
        x="x",
        y="y",
        source=source_point,
        marker="square",
        size=34,
        fill_color=POINT_COLOR,
        fill_alpha=0.12,
        line_alpha=0.0,
    )
    plot.scatter(x="x", y="y", source=source_labels, size=9, color=SOURCE_COLOR)
    plot.scatter(x="x", y="y", source=target_labels, size=9, color=TARGET_COLOR)
    plot.add_layout(_labels(LabelSet, source_labels, "#111827"))
    plot.add_layout(_labels(LabelSet, target_labels, TARGET_COLOR))
    plot.add_layout(_labels(LabelSet, source_point, POINT_COLOR))
    plot.add_layout(_labels(LabelSet, target_point, TARGET_COLOR))

    draw_tool = PointDrawTool(renderers=[point_handle], add=False)
    plot.add_tools(draw_tool)
    plot.toolbar.active_tap = draw_tool

    angle_slider = Slider(title="theta (degrees)", start=-150, end=150, step=1, value=angle_degrees, width=330)
    tx_slider = Slider(title="translation x", start=-0.5, end=3.2, step=0.02, value=translation.real, width=330)
    ty_slider = Slider(title="translation y", start=-1.2, end=1.8, step=0.02, value=translation.imag, width=330)
    zero_theta_toggle = Toggle(label="theta = 0", active=False, width=108)
    zero_translation_toggle = Toggle(label="translation = 0", active=False, width=138)
    overlap_toggle = Toggle(label="overlap triangles", active=False, width=152)

    callback = CustomJS(
        args=dict(
            angle_slider=angle_slider,
            tx_slider=tx_slider,
            ty_slider=ty_slider,
            source_triangle=source_triangle,
            target_triangle=target_triangle,
            source_point=source_point,
            target_point=target_point,
            target_labels=target_labels,
            circle_source=circle_source,
            candidate_source=candidate_source,
            source_distance_source=source_distance_source,
            target_distance_source=target_distance_source,
            motion_source=motion_source,
            summary=summary,
        ),
        code=_BOKEH_UPDATE_JS,
    )
    for control in (angle_slider, tx_slider, ty_slider):
        control.js_on_change("value", callback)
    source_point.js_on_change("data", callback)

    zero_callback = CustomJS(
        args=dict(
            angle_slider=angle_slider,
            tx_slider=tx_slider,
            ty_slider=ty_slider,
            zero_theta_toggle=zero_theta_toggle,
            zero_translation_toggle=zero_translation_toggle,
            overlap_toggle=overlap_toggle,
            zero_state=zero_state,
        ),
        code=_ZERO_TOGGLE_JS,
    )
    for control in (zero_theta_toggle, zero_translation_toggle, overlap_toggle):
        control.js_on_change("active", zero_callback)

    zero_controls = row(zero_theta_toggle, zero_translation_toggle, overlap_toggle, width=390)
    controls = column(angle_slider, tx_slider, ty_slider, zero_controls, summary, width=390)
    return row(plot, controls, sizing_mode="stretch_width")


def export_triangle_distance_locator_html(
    path: str | Path = "docs/assets/plots/triangle_distance_locator.html",
    vertices: Sequence[complex] | npt.ArrayLike | None = None,
    point: complex | None = None,
    angle_degrees: float = 34.0,
    translation: complex = 2.1 + 0.45j,
) -> Path:
    """Write a standalone Bokeh HTML file for the triangle distance locator."""
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
        make_triangle_distance_locator_bokeh(vertices, point, angle_degrees, translation),
        filename=str(out),
        resources=INLINE,
        title="Triangle distance locator",
    )
    return out


def _circle_path(center: complex, radius: float, n_points: int = 180) -> ComplexArray:
    angles = np.linspace(0, 2 * math.pi, n_points)
    return center + radius * np.exp(1j * angles)


def _closed_triangle(points: ComplexArray) -> ComplexArray:
    return np.append(points, points[0])


def _bokeh_sources(geometry: LocatorGeometry) -> dict[str, dict[str, list]]:
    source_closed = _closed_triangle(geometry.source_triangle)
    target_closed = _closed_triangle(geometry.target_triangle)
    circle_paths = [_circle_path(center, radius) for center, radius in zip(geometry.target_triangle, geometry.radii)]
    labels = ["A", "B", "C"]
    target_labels = ["A'", "B'", "C'"]
    candidate_colors = [TARGET_COLOR if residual < 1e-8 else CANDIDATE_COLOR for residual in geometry.candidate_residuals]
    candidate_sizes = [13 if residual < 1e-8 else 10 for residual in geometry.candidate_residuals]

    return {
        "source_triangle": {"x": source_closed.real.tolist(), "y": source_closed.imag.tolist()},
        "target_triangle": {"x": target_closed.real.tolist(), "y": target_closed.imag.tolist()},
        "source_point": {"x": [geometry.source_point.real], "y": [geometry.source_point.imag], "label": ["P"]},
        "target_point": {"x": [geometry.target_point.real], "y": [geometry.target_point.imag], "label": ["P'"]},
        "source_labels": {
            "x": geometry.source_triangle.real.tolist(),
            "y": geometry.source_triangle.imag.tolist(),
            "label": labels,
        },
        "target_labels": {
            "x": geometry.target_triangle.real.tolist(),
            "y": geometry.target_triangle.imag.tolist(),
            "label": target_labels,
        },
        "circles": {
            "xs": [path.real.tolist() for path in circle_paths],
            "ys": [path.imag.tolist() for path in circle_paths],
            "color": CIRCLE_COLORS,
        },
        "candidates": {
            "x": geometry.candidates.real.tolist(),
            "y": geometry.candidates.imag.tolist(),
            "color": candidate_colors,
            "size": candidate_sizes,
        },
        "source_distances": {
            "xs": [[center.real, geometry.source_point.real] for center in geometry.source_triangle],
            "ys": [[center.imag, geometry.source_point.imag] for center in geometry.source_triangle],
            "color": CIRCLE_COLORS,
        },
        "target_distances": {
            "xs": [[center.real, geometry.target_point.real] for center in geometry.target_triangle],
            "ys": [[center.imag, geometry.target_point.imag] for center in geometry.target_triangle],
            "color": CIRCLE_COLORS,
        },
        "motion_lines": {
            "xs": [[a.real, b.real] for a, b in zip(geometry.source_triangle, geometry.target_triangle)],
            "ys": [[a.imag, b.imag] for a, b in zip(geometry.source_triangle, geometry.target_triangle)],
        },
    }


def _labels(label_set_class, source, color: str, size: str = "13px"):
    return label_set_class(
        x="x",
        y="y",
        text="label",
        source=source,
        x_offset=8,
        y_offset=7,
        text_font_size=size,
        text_color=color,
        background_fill_color="#ffffff",
        background_fill_alpha=0.72,
    )


def _format_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _format_complex(z: complex) -> str:
    sign = "+" if z.imag >= 0 else "-"
    return f"{_format_number(z.real)} {sign} {_format_number(abs(z.imag))}i"


def _distance_table_rows(geometry: LocatorGeometry) -> str:
    labels = ["AP", "BP", "CP"]
    target_labels = ["A'P'", "B'P'", "C'P'"]
    target_distances = np.abs(geometry.target_point - geometry.target_triangle)
    rows = []
    for label, target_label, source_distance, target_distance, color in zip(
        labels,
        target_labels,
        geometry.radii,
        target_distances,
        CIRCLE_COLORS,
    ):
        rows.append(
            "<tr>"
            f'<td><span style="color:{color}; font-weight:700;">{label}</span></td>'
            f"<td>{_format_number(float(source_distance))}</td>"
            f'<td><span style="color:{color}; font-weight:700;">{target_label}</span></td>'
            f"<td>{_format_number(float(target_distance))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _format_summary(geometry: LocatorGeometry) -> str:
    residual = float(np.min(geometry.candidate_residuals)) if len(geometry.candidate_residuals) else math.nan
    return f"""
<b>Distance preservation</b><br>
The image point P' must keep the same distances to A', B', C'.<br>
<table style="border-collapse:collapse; margin-top:10px; width:100%;">
  <thead>
    <tr>
      <th style="text-align:left; border-bottom:1px solid #d8dee9;">source</th>
      <th style="text-align:right; border-bottom:1px solid #d8dee9;">length</th>
      <th style="text-align:left; border-bottom:1px solid #d8dee9;">target</th>
      <th style="text-align:right; border-bottom:1px solid #d8dee9;">length</th>
    </tr>
  </thead>
  <tbody>
    {_distance_table_rows(geometry)}
  </tbody>
</table>
<br>
The first two target circles usually give two candidates; the third selects P'.<br>
best circle residual = {_format_number(residual)}
"""


def _panel_styles() -> dict[str, str]:
    return {
        "font-family": "JetBrains Mono, Menlo, Consolas, monospace",
        "font-size": "13px",
        "line-height": "1.55",
        "border": "1px solid #d8dee9",
        "border-radius": "6px",
        "padding": "12px",
        "background": "#fbfbfc",
    }


_BOKEH_UPDATE_JS = r"""
const theta = angle_slider.value * Math.PI / 180;
const tx = tx_slider.value;
const ty = ty_slider.value;
const sourceTri = source_triangle.data.x.slice(0, 3).map((x, i) => ({re: x, im: source_triangle.data.y[i]}));
const p = {re: source_point.data.x[0], im: source_point.data.y[0]};
const CIRCLE_COLORS = ["#d55e00", "#009e73", "#cc79a7"];
const TARGET_COLOR = "#0072b2";
const CANDIDATE_COLOR = "#e69f00";

function c(re, im) {
  return {re: re, im: im};
}
function sub(a, b) {
  return c(a.re - b.re, a.im - b.im);
}
function abs(z) {
  return Math.hypot(z.re, z.im);
}
function mul(a, b) {
  return c(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
function add(a, b) {
  return c(a.re + b.re, a.im + b.im);
}
function transform(z) {
  const rotation = c(Math.cos(theta), Math.sin(theta));
  return add(mul(rotation, z), c(tx, ty));
}
function circleIntersections(c0, r0, c1, r1) {
  const dx = c1.re - c0.re;
  const dy = c1.im - c0.im;
  const d = Math.hypot(dx, dy);
  if (d < 1e-10 || d > r0 + r1 + 1e-10 || d < Math.abs(r0 - r1) - 1e-10) {
    return [];
  }
  const along = (r0 * r0 - r1 * r1 + d * d) / (2 * d);
  const h2 = Math.max(0, r0 * r0 - along * along);
  const h = Math.sqrt(h2);
  const ux = dx / d;
  const uy = dy / d;
  const fx = c0.re + along * ux;
  const fy = c0.im + along * uy;
  if (h <= 1e-10) {
    return [c(fx, fy)];
  }
  return [c(fx - h * uy, fy + h * ux), c(fx + h * uy, fy - h * ux)];
}
function circlePath(center, radius) {
  const xs = [];
  const ys = [];
  for (let k = 0; k < 180; k += 1) {
    const a = 2 * Math.PI * k / 179;
    xs.push(center.re + radius * Math.cos(a));
    ys.push(center.im + radius * Math.sin(a));
  }
  return {xs: xs, ys: ys};
}
function fmt(value) {
  return value.toFixed(4).replace(/\.?0+$/, "");
}
function fmtComplex(z) {
  const sign = z.im >= 0 ? " + " : " - ";
  return `${fmt(z.re)}${sign}${fmt(Math.abs(z.im))}i`;
}
function tableRow(sourceLabel, targetLabel, sourceDistance, targetDistance, color) {
  return `<tr>
    <td><span style="color:${color}; font-weight:700;">${sourceLabel}</span></td>
    <td style="text-align:right;">${fmt(sourceDistance)}</td>
    <td><span style="color:${color}; font-weight:700;">${targetLabel}</span></td>
    <td style="text-align:right;">${fmt(targetDistance)}</td>
  </tr>`;
}

const targetTri = sourceTri.map(transform);
const targetP = transform(p);
const radii = sourceTri.map((vertex) => abs(sub(p, vertex)));
const targetDistances = targetTri.map((vertex) => abs(sub(targetP, vertex)));
const candidates = circleIntersections(targetTri[0], radii[0], targetTri[1], radii[1]);
const candidateData = candidates.map((candidate) => ({
  point: candidate,
  residual: Math.abs(abs(sub(candidate, targetTri[2])) - radii[2]),
})).sort((a, b) => a.residual - b.residual);

target_triangle.data = {
  x: [targetTri[0].re, targetTri[1].re, targetTri[2].re, targetTri[0].re],
  y: [targetTri[0].im, targetTri[1].im, targetTri[2].im, targetTri[0].im],
};
target_point.data = {x: [targetP.re], y: [targetP.im], label: ["P'"]};
target_labels.data = {
  x: targetTri.map((z) => z.re),
  y: targetTri.map((z) => z.im),
  label: ["A'", "B'", "C'"],
};
const circleXs = [];
const circleYs = [];
for (let i = 0; i < 3; i += 1) {
  const path = circlePath(targetTri[i], radii[i]);
  circleXs.push(path.xs);
  circleYs.push(path.ys);
}
circle_source.data = {xs: circleXs, ys: circleYs, color: CIRCLE_COLORS};
candidate_source.data = {
  x: candidateData.map((item) => item.point.re),
  y: candidateData.map((item) => item.point.im),
  color: candidateData.map((item) => item.residual < 1e-8 ? TARGET_COLOR : CANDIDATE_COLOR),
  size: candidateData.map((item) => item.residual < 1e-8 ? 13 : 10),
};
source_distance_source.data = {
  xs: sourceTri.map((center) => [center.re, p.re]),
  ys: sourceTri.map((center) => [center.im, p.im]),
  color: CIRCLE_COLORS,
};
target_distance_source.data = {
  xs: targetTri.map((center) => [center.re, targetP.re]),
  ys: targetTri.map((center) => [center.im, targetP.im]),
  color: CIRCLE_COLORS,
};
motion_source.data = {
  xs: sourceTri.map((vertex, i) => [vertex.re, targetTri[i].re]),
  ys: sourceTri.map((vertex, i) => [vertex.im, targetTri[i].im]),
};

const residual = candidateData.length > 0 ? candidateData[0].residual : NaN;
summary.text = `
<b>Distance preservation</b><br>
The image point P' must keep the same distances to A', B', C'.<br>
<table style="border-collapse:collapse; margin-top:10px; width:100%;">
  <thead>
    <tr>
      <th style="text-align:left; border-bottom:1px solid #d8dee9;">source</th>
      <th style="text-align:right; border-bottom:1px solid #d8dee9;">length</th>
      <th style="text-align:left; border-bottom:1px solid #d8dee9;">target</th>
      <th style="text-align:right; border-bottom:1px solid #d8dee9;">length</th>
    </tr>
  </thead>
  <tbody>
    ${tableRow("AP", "A'P'", radii[0], targetDistances[0], CIRCLE_COLORS[0])}
    ${tableRow("BP", "B'P'", radii[1], targetDistances[1], CIRCLE_COLORS[1])}
    ${tableRow("CP", "C'P'", radii[2], targetDistances[2], CIRCLE_COLORS[2])}
  </tbody>
</table>
<br>
The first two target circles usually give two candidates; the third selects P'.<br>
best circle residual = ${fmt(residual)}
`;
"""


_ZERO_TOGGLE_JS = r"""
const state = zero_state.data;
const source = cb_obj;

function rememberTheta() {
  if (!angle_slider.disabled) {
    state.theta = [angle_slider.value];
  }
}
function rememberTranslation() {
  if (!tx_slider.disabled && !ty_slider.disabled) {
    state.tx = [tx_slider.value];
    state.ty = [ty_slider.value];
  }
}
function setThetaZero() {
  angle_slider.value = 0;
  angle_slider.disabled = true;
}
function restoreTheta() {
  angle_slider.disabled = false;
  angle_slider.value = state.theta[0];
}
function setTranslationZero() {
  tx_slider.value = 0;
  ty_slider.value = 0;
  tx_slider.disabled = true;
  ty_slider.disabled = true;
}
function restoreTranslation() {
  tx_slider.disabled = false;
  ty_slider.disabled = false;
  tx_slider.value = state.tx[0];
  ty_slider.value = state.ty[0];
}

if (source === overlap_toggle) {
  if (overlap_toggle.active) {
    rememberTheta();
    rememberTranslation();
    setThetaZero();
    setTranslationZero();
    zero_theta_toggle.disabled = true;
    zero_translation_toggle.disabled = true;
  } else {
    zero_theta_toggle.disabled = false;
    zero_translation_toggle.disabled = false;
    if (zero_theta_toggle.active) {
      setThetaZero();
    } else {
      restoreTheta();
    }
    if (zero_translation_toggle.active) {
      setTranslationZero();
    } else {
      restoreTranslation();
    }
  }
} else if (source === zero_theta_toggle) {
  if (zero_theta_toggle.active) {
    rememberTheta();
    setThetaZero();
  } else if (!overlap_toggle.active) {
    restoreTheta();
  }
} else if (source === zero_translation_toggle) {
  if (zero_translation_toggle.active) {
    rememberTranslation();
    setTranslationZero();
  } else if (!overlap_toggle.active) {
    restoreTranslation();
  }
}
zero_state.change.emit();
"""
