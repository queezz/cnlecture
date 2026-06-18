"""Classify the direct motion carrying one segment to another."""

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
CONSTRUCTION_COLOR = "#d55e00"
CENTER_COLOR = "#009e73"
POINT_COLOR = "#7c3aed"


@dataclass(frozen=True)
class SegmentMotionGeometry:
    """Geometry for a direct motion carrying ``AB`` to ``A'B'``."""

    source_segment: ComplexArray
    target_segment: ComplexArray
    angle_degrees: float
    translation: complex
    motion_kind: str
    rotation_center: complex | None
    segment_length: float


def default_segment() -> ComplexArray:
    """Return a sample source segment."""
    return np.asarray([-0.85 - 0.35j, 0.9 + 0.25j], dtype=complex)


def _as_segment(points: Sequence[complex] | npt.ArrayLike) -> ComplexArray:
    segment = np.asarray(points, dtype=complex)
    if segment.shape != (2,):
        raise ValueError("points must contain exactly two complex points")
    if abs(segment[1] - segment[0]) < 1e-10:
        raise ValueError("segment endpoints must be distinct")
    return segment.copy()


def apply_segment_motion(
    points: Sequence[complex] | npt.ArrayLike,
    angle_degrees: float,
    translation: complex,
) -> ComplexArray:
    """Apply the direct motion ``z -> exp(i theta) z + translation``."""
    zs = np.asarray(points, dtype=complex)
    angle = math.radians(float(angle_degrees))
    rotation = complex(math.cos(angle), math.sin(angle))
    return rotation * zs + complex(translation)


def direct_motion_rotation_center(angle_degrees: float, translation: complex, *, tol: float = 1e-10) -> complex | None:
    """Return the fixed point of ``z -> exp(i theta) z + translation`` when it is a rotation."""
    angle = math.radians(float(angle_degrees))
    rotation = complex(math.cos(angle), math.sin(angle))
    if abs(rotation - 1) < tol:
        return None
    return complex(translation) / (1 - rotation)


def rotate_image_back(
    points: Sequence[complex] | npt.ArrayLike,
    center: complex,
    angle_degrees: float,
    fraction: float,
) -> ComplexArray:
    """Rotate image points backward by ``fraction * theta`` around ``center``."""
    zs = np.asarray(points, dtype=complex)
    amount = -math.radians(float(angle_degrees)) * float(fraction)
    rotation = complex(math.cos(amount), math.sin(amount))
    fixed = complex(center)
    return fixed + rotation * (zs - fixed)


def classify_direct_motion(angle_degrees: float, *, tol: float = 1e-10) -> str:
    """Classify ``z -> exp(i theta) z + v`` as a translation or rotation."""
    angle = math.radians(float(angle_degrees))
    return "translation" if abs(complex(math.cos(angle), math.sin(angle)) - 1) < tol else "rotation"


def segment_motion_geometry(
    points: Sequence[complex] | npt.ArrayLike | None = None,
    angle_degrees: float = 42.0,
    translation: complex = 1.85 + 0.55j,
) -> SegmentMotionGeometry:
    """Return geometry for the direct motion carrying ``AB`` to ``A'B'``."""
    source_segment = _as_segment(points if points is not None else default_segment())
    target_segment = apply_segment_motion(source_segment, angle_degrees, translation)
    motion_kind = classify_direct_motion(angle_degrees)
    center = direct_motion_rotation_center(angle_degrees, translation)

    return SegmentMotionGeometry(
        source_segment=source_segment,
        target_segment=target_segment,
        angle_degrees=float(angle_degrees),
        translation=complex(translation),
        motion_kind=motion_kind,
        rotation_center=center,
        segment_length=float(abs(source_segment[1] - source_segment[0])),
    )


def make_direct_segment_motion_bokeh(
    points: Sequence[complex] | npt.ArrayLike | None = None,
    angle_degrees: float = 42.0,
    translation: complex = 1.85 + 0.55j,
):
    """Build a Bokeh visualization for the direct motion determined by a segment."""
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
        )
        from bokeh.plotting import figure
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Bokeh is required for this interactive visualization. "
            'Install it with `pip install -e ".[dev]"`.'
        ) from exc

    geometry = segment_motion_geometry(points, angle_degrees, translation)
    sources = _bokeh_sources(geometry)

    source_segment = ColumnDataSource(data=sources["source_segment"])
    target_anchor = ColumnDataSource(data=sources["target_anchor"])
    target_segment = ColumnDataSource(data=sources["target_segment"])
    target_labels = ColumnDataSource(data=sources["target_labels"])
    connector_source = ColumnDataSource(data=sources["connectors"])
    vector_source = ColumnDataSource(data=sources["vector"])
    center_source = ColumnDataSource(data=sources["center"])
    radius_source = ColumnDataSource(data=sources["radii"])
    arc_source = ColumnDataSource(data=sources["arc"])
    back_segment_source = ColumnDataSource(data=sources["back_segment"])
    back_radius_source = ColumnDataSource(data=sources["back_radii"])

    summary = Div(text=_format_summary(geometry), width=370, styles=_panel_styles())

    plot = figure(
        width=760,
        height=560,
        x_range=Range1d(-2.0, 4.4, bounds=(-5.5, 6.5)),
        y_range=Range1d(-2.0, 3.2, bounds=(-5.5, 5.5)),
        x_axis_label="Re",
        y_axis_label="Im",
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
        toolbar_location="above",
        title="Direct motion carrying AB to A'B'",
    )
    plot.grid.grid_line_alpha = 0.28
    plot.xaxis.axis_label_text_font_style = "normal"
    plot.yaxis.axis_label_text_font_style = "normal"

    plot.multi_line(xs="xs", ys="ys", source=connector_source, line_color="#9ca3af", line_dash="dashed", line_width=1.4)
    plot.multi_line(xs="xs", ys="ys", source=radius_source, line_color=CENTER_COLOR, line_dash="dotted", line_width=1.5)
    plot.multi_line(xs="xs", ys="ys", source=back_radius_source, line_color=CONSTRUCTION_COLOR, line_dash="dotted", line_width=1.5)
    plot.line(x="x", y="y", source=arc_source, line_color=CONSTRUCTION_COLOR, line_width=3)
    plot.line(x="x", y="y", source=source_segment, line_color=SOURCE_COLOR, line_width=4)
    plot.line(x="x", y="y", source=target_segment, line_color=TARGET_COLOR, line_width=4)
    plot.line(
        x="x",
        y="y",
        source=back_segment_source,
        line_color=CONSTRUCTION_COLOR,
        line_dash="dashed",
        line_width=3,
    )
    plot.segment(
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        source=vector_source,
        line_color=CONSTRUCTION_COLOR,
        line_width=3,
    )
    plot.scatter(x="x", y="y", source=center_source, size="size", fill_color="#ffffff", line_color=CENTER_COLOR, line_width=2.4)
    plot.scatter(x="x", y="y", source=target_labels, size=11, fill_color="#ffffff", line_color=TARGET_COLOR, line_width=2)
    target_anchor_handle = plot.scatter(
        x="x",
        y="y",
        source=target_anchor,
        marker="square",
        size=34,
        fill_color=TARGET_COLOR,
        fill_alpha=0.12,
        line_alpha=0.0,
    )
    plot.scatter(x="x", y="y", source=back_segment_source, size=10, fill_color="#ffffff", line_color=CONSTRUCTION_COLOR, line_width=2)
    source_points = plot.scatter(x="x", y="y", source=source_segment, size=13, fill_color=POINT_COLOR, line_color="#4c1d95", line_width=2)
    drag_handles = plot.scatter(
        x="x",
        y="y",
        source=source_segment,
        marker="square",
        size=34,
        fill_color=POINT_COLOR,
        fill_alpha=0.12,
        line_alpha=0.0,
    )
    plot.add_layout(_labels(LabelSet, source_segment, SOURCE_COLOR))
    plot.add_layout(_labels(LabelSet, target_labels, TARGET_COLOR))
    plot.add_layout(_labels(LabelSet, back_segment_source, CONSTRUCTION_COLOR, size="12px"))
    plot.add_layout(_labels(LabelSet, center_source, CENTER_COLOR, size="12px"))

    draw_tool = PointDrawTool(renderers=[drag_handles, source_points, target_anchor_handle], add=False)
    plot.add_tools(draw_tool)
    plot.toolbar.active_tap = draw_tool

    angle_slider = Slider(title="theta (degrees)", start=-180, end=180, step=1, value=angle_degrees, width=330)
    back_slider = Slider(title="rotate image back (%)", start=0, end=100, step=1, value=0, width=330)

    callback = CustomJS(
        args=dict(
            angle_slider=angle_slider,
            source_segment=source_segment,
            target_anchor=target_anchor,
            target_segment=target_segment,
            target_labels=target_labels,
            connector_source=connector_source,
            vector_source=vector_source,
            center_source=center_source,
            radius_source=radius_source,
            arc_source=arc_source,
            back_segment_source=back_segment_source,
            back_radius_source=back_radius_source,
            back_slider=back_slider,
            summary=summary,
        ),
        code=_BOKEH_UPDATE_JS,
    )
    for control in (angle_slider, back_slider):
        control.js_on_change("value", callback)
    source_segment.js_on_change("data", callback)
    target_anchor.js_on_change("data", callback)

    controls = column(angle_slider, back_slider, summary, width=390)
    return row(plot, controls, sizing_mode="stretch_width")


def export_direct_segment_motion_html(
    path: str | Path = "docs/assets/plots/direct_segment_motion.html",
    points: Sequence[complex] | npt.ArrayLike | None = None,
    angle_degrees: float = 42.0,
    translation: complex = 1.85 + 0.55j,
) -> Path:
    """Write a standalone Bokeh HTML file for the direct segment motion figure."""
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
        make_direct_segment_motion_bokeh(points, angle_degrees, translation),
        filename=str(out),
        resources=INLINE,
        title="Direct segment motion",
    )
    return out


def _arc_path(center: complex, start: complex, angle_degrees: float, n_points: int = 80) -> ComplexArray:
    if center is None:
        return np.asarray([], dtype=complex)
    radius = abs(start - center) * 0.28
    if radius < 1e-8:
        return np.asarray([], dtype=complex)
    start_angle = math.atan2((start - center).imag, (start - center).real)
    stop_angle = start_angle + math.radians(float(angle_degrees))
    angles = np.linspace(start_angle, stop_angle, n_points)
    return center + radius * np.exp(1j * angles)


def _bokeh_sources(geometry: SegmentMotionGeometry) -> dict[str, dict[str, list]]:
    source = geometry.source_segment
    target = geometry.target_segment
    center = geometry.rotation_center
    arc = _arc_path(center, source[0], geometry.angle_degrees) if center is not None else np.asarray([], dtype=complex)

    center_data = {"x": [], "y": [], "label": [], "size": []}
    radius_data = {"xs": [], "ys": []}
    vector_data = {"x0": [], "y0": [], "x1": [], "y1": []}
    if center is not None:
        center_data = {"x": [center.real], "y": [center.imag], "label": ["O"], "size": [13]}
        radius_data = {
            "xs": [[center.real, source[0].real], [center.real, target[0].real]],
            "ys": [[center.imag, source[0].imag], [center.imag, target[0].imag]],
        }
    else:
        vector_data = {
            "x0": [source[0].real],
            "y0": [source[0].imag],
            "x1": [target[0].real],
            "y1": [target[0].imag],
        }

    return {
        "source_segment": {
            "x": source.real.tolist(),
            "y": source.imag.tolist(),
            "label": ["A", "B"],
        },
        "target_anchor": {
            "x": [target[0].real],
            "y": [target[0].imag],
            "label": ["A'"],
        },
        "target_segment": {
            "x": target.real.tolist(),
            "y": target.imag.tolist(),
        },
        "target_labels": {
            "x": target.real.tolist(),
            "y": target.imag.tolist(),
            "label": ["A'", "B'"],
        },
        "connectors": {
            "xs": [[source[0].real, target[0].real], [source[1].real, target[1].real]],
            "ys": [[source[0].imag, target[0].imag], [source[1].imag, target[1].imag]],
        },
        "vector": vector_data,
        "center": center_data,
        "radii": radius_data,
        "arc": {"x": arc.real.tolist(), "y": arc.imag.tolist()},
        "back_segment": {"x": [], "y": [], "label": []},
        "back_radii": {"xs": [], "ys": []},
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


def _format_summary(geometry: SegmentMotionGeometry) -> str:
    target_length = float(abs(geometry.target_segment[1] - geometry.target_segment[0]))
    center_line = (
        "fixed center O = none"
        if geometry.rotation_center is None
        else f"fixed center O = {_format_complex(geometry.rotation_center)}"
    )
    return f"""
<b>Direct motion</b><br>
M(z) = e<sup>i theta</sup>z + v<br>
theta = {_format_number(geometry.angle_degrees)} degrees<br>
v = {_format_complex(geometry.translation)}<br>
classification = <b>{geometry.motion_kind}</b><br>
{center_line}<br>
image back-rotation = 0%<br>
<br>
<table style="border-collapse:collapse; width:100%;">
  <tbody>
    <tr>
      <td style="border-bottom:1px solid #d8dee9;">|AB|</td>
      <td style="text-align:right; border-bottom:1px solid #d8dee9;">{_format_number(geometry.segment_length)}</td>
    </tr>
    <tr>
      <td>|A'B'|</td>
      <td style="text-align:right;">{_format_number(target_length)}</td>
    </tr>
  </tbody>
</table>
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
const thetaDegrees = angle_slider.value;
const theta = thetaDegrees * Math.PI / 180;
const source = source_segment.data.x.map((x, i) => ({re: x, im: source_segment.data.y[i]}));
const anchor = {re: target_anchor.data.x[0], im: target_anchor.data.y[0]};

function c(re, im) {
  return {re: re, im: im};
}
function add(a, b) {
  return c(a.re + b.re, a.im + b.im);
}
function sub(a, b) {
  return c(a.re - b.re, a.im - b.im);
}
function mul(a, b) {
  return c(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
function div(a, b) {
  const den = b.re * b.re + b.im * b.im;
  return c((a.re * b.re + a.im * b.im) / den, (a.im * b.re - a.re * b.im) / den);
}
function abs(z) {
  return Math.hypot(z.re, z.im);
}
function rotateAround(z, center, degrees) {
  const amount = degrees * Math.PI / 180;
  const turn = c(Math.cos(amount), Math.sin(amount));
  return add(center, mul(turn, sub(z, center)));
}
function fmt(value) {
  return value.toFixed(4).replace(/\.?0+$/, "");
}
function fmtComplex(z) {
  const sign = z.im >= 0 ? " + " : " - ";
  return `${fmt(z.re)}${sign}${fmt(Math.abs(z.im))}i`;
}
function arcPath(center, start, degrees) {
  const radius = abs(sub(start, center)) * 0.28;
  if (radius < 1e-8) {
    return {x: [], y: []};
  }
  const base = Math.atan2(start.im - center.im, start.re - center.re);
  const stop = base + degrees * Math.PI / 180;
  const xs = [];
  const ys = [];
  for (let i = 0; i < 80; i += 1) {
    const a = base + (stop - base) * i / 79;
    xs.push(center.re + radius * Math.cos(a));
    ys.push(center.im + radius * Math.sin(a));
  }
  return {x: xs, y: ys};
}

const rotation = c(Math.cos(theta), Math.sin(theta));
const target = [anchor, add(anchor, mul(rotation, sub(source[1], source[0])))];
const translation = sub(anchor, mul(rotation, source[0]));
const isTranslation = abs(sub(rotation, c(1, 0))) < 1e-10;
let center = null;
if (!isTranslation) {
  center = div(translation, sub(c(1, 0), rotation));
}
const backFraction = back_slider.value / 100;

target_segment.data = {
  x: target.map((z) => z.re),
  y: target.map((z) => z.im),
};
target_labels.data = {
  x: target.map((z) => z.re),
  y: target.map((z) => z.im),
  label: ["A'", "B'"],
};
connector_source.data = {
  xs: [[source[0].re, target[0].re], [source[1].re, target[1].re]],
  ys: [[source[0].im, target[0].im], [source[1].im, target[1].im]],
};

if (center === null) {
  vector_source.data = {
    x0: [source[0].re],
    y0: [source[0].im],
    x1: [target[0].re],
    y1: [target[0].im],
  };
  center_source.data = {x: [], y: [], label: [], size: []};
  radius_source.data = {xs: [], ys: []};
  arc_source.data = {x: [], y: []};
  back_segment_source.data = {x: [], y: [], label: []};
  back_radius_source.data = {xs: [], ys: []};
  back_slider.disabled = true;
} else {
  const arc = arcPath(center, source[0], thetaDegrees);
  const backSegment = backFraction <= 0
    ? []
    : target.map((z) => rotateAround(z, center, -thetaDegrees * backFraction));
  vector_source.data = {x0: [], y0: [], x1: [], y1: []};
  center_source.data = {x: [center.re], y: [center.im], label: ["O"], size: [13]};
  radius_source.data = {
    xs: [[center.re, source[0].re], [center.re, target[0].re]],
    ys: [[center.im, source[0].im], [center.im, target[0].im]],
  };
  arc_source.data = arc;
  back_segment_source.data = {
    x: backSegment.map((z) => z.re),
    y: backSegment.map((z) => z.im),
    label: backSegment.length > 0 ? ["A_t", "B_t"] : [],
  };
  back_radius_source.data = {
    xs: backSegment.map((z) => [center.re, z.re]),
    ys: backSegment.map((z) => [center.im, z.im]),
  };
  back_slider.disabled = false;
}

const sourceLength = abs(sub(source[1], source[0]));
const targetLength = abs(sub(target[1], target[0]));
const centerLine = center === null ? "fixed center O = none" : `fixed center O = ${fmtComplex(center)}`;
summary.text = `
<b>Direct motion</b><br>
M(z) = e<sup>i theta</sup>z + v<br>
theta = ${fmt(thetaDegrees)} degrees<br>
v = ${fmtComplex(translation)}<br>
classification = <b>${isTranslation ? "translation" : "rotation"}</b><br>
${centerLine}<br>
image back-rotation = ${isTranslation ? "n/a" : `${fmt(back_slider.value)}%`}<br>
<br>
<table style="border-collapse:collapse; width:100%;">
  <tbody>
    <tr>
      <td style="border-bottom:1px solid #d8dee9;">|AB|</td>
      <td style="text-align:right; border-bottom:1px solid #d8dee9;">${fmt(sourceLength)}</td>
    </tr>
    <tr>
      <td>|A'B'|</td>
      <td style="text-align:right;">${fmt(targetLength)}</td>
    </tr>
  </tbody>
</table>
`;
"""
