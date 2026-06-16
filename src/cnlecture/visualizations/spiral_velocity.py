"""Velocity geometry for the logarithmic spiral ``Z(t)=e^{at}e^{ibt}``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

ComplexArray = npt.NDArray[np.complexfloating]

SPIRAL_COLOR = "#111827"
UNIT_COLOR = "#94a3b8"
GUIDE_COLOR = "#64748b"
POINT_COLOR = "#111827"
A_COLOR = "#e69f00"
B_COLOR = "#009e73"
M_COLOR = "#cc79a7"

COMPONENT_COLORS = [A_COLOR, B_COLOR, M_COLOR]


@dataclass(frozen=True)
class SpiralStepGeometry:
    """Finite-step and infinitesimal geometry around one point of the spiral."""

    z: complex
    z_next: complex
    radial_point: complex
    m: complex
    a_exact: complex
    b_exact: complex
    a_linear: complex
    b_linear: complex
    m_linear: complex
    angle_degrees: float
    perpendicular_error_degrees: float


def spiral_point(a: float, b: float, t: float) -> complex:
    """Return ``Z(t)=e^(at)e^(ibt)``."""
    radius = math.exp(a * t)
    angle = b * t
    return radius * complex(math.cos(angle), math.sin(angle))


def spiral_curve(
    a: float = 0.18,
    b: float = 1.0,
    t_min: float = -8.0,
    t_max: float = 5.5,
    n_points: int = 700,
) -> ComplexArray:
    """Return sampled points on the spiral."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    ts = np.linspace(t_min, t_max, n_points)
    return np.exp(a * ts) * np.exp(1j * b * ts)


def angle_between(u: complex, v: complex) -> float:
    """Return the smaller angle between two non-zero complex vectors in degrees."""
    if abs(u) == 0 or abs(v) == 0:
        return math.nan

    cosine = (u.real * v.real + u.imag * v.imag) / (abs(u) * abs(v))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def spiral_step_geometry(
    a: float = 0.18,
    b: float = 1.0,
    t: float = 1.25,
    delta: float = 0.22,
) -> SpiralStepGeometry:
    """Return the exact finite-step picture and its linear right-triangle model.

    The exact finite step decomposes

    ``M = Z(t + delta) - Z(t)``

    as a radial change ``A`` plus a circular chord ``B``.  For small ``delta``,
    these approach ``a Z delta`` and ``i b Z delta``.
    """
    z = spiral_point(a, b, t)
    z_next = spiral_point(a, b, t + delta)
    old_ray = complex(math.cos(b * t), math.sin(b * t))
    radial_point = abs(z_next) * old_ray

    a_exact = radial_point - z
    b_exact = z_next - radial_point
    m = z_next - z
    a_linear = a * z * delta
    b_linear = 1j * b * z * delta
    m_linear = a_linear + b_linear
    angle = angle_between(a_exact, b_exact)

    return SpiralStepGeometry(
        z=z,
        z_next=z_next,
        radial_point=radial_point,
        m=m,
        a_exact=a_exact,
        b_exact=b_exact,
        a_linear=a_linear,
        b_linear=b_linear,
        m_linear=m_linear,
        angle_degrees=angle,
        perpendicular_error_degrees=abs(angle - 90.0),
    )


def make_spiral_velocity_bokeh(
    a: float = 0.18,
    b: float = 1.0,
    t: float = 1.25,
    delta: float = 0.22,
):
    """Build a Bokeh figure for the spiral velocity right-triangle argument."""
    try:
        from bokeh.layouts import column, row
        from bokeh.models import (
            ColumnDataSource,
            CustomJS,
            Div,
            Label,
            LabelSet,
            Range1d,
            Slider,
            TeX,
        )
        from bokeh.plotting import figure
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Bokeh is required for this interactive visualization. "
            'Install it with `pip install -e ".[dev]"`.'
        ) from exc

    sources = _bokeh_sources(a, b, t, delta)
    curve = spiral_curve(a, b)
    unit = np.exp(1j * np.linspace(0, 2 * math.pi, 240))

    curve_source = ColumnDataSource(data=dict(x=curve.real.tolist(), y=curve.imag.tolist()))
    unit_source = ColumnDataSource(data=dict(x=unit.real.tolist(), y=unit.imag.tolist()))
    overview_guides = ColumnDataSource(data=sources["overview_guides"])
    overview_local_shade = ColumnDataSource(data=sources["overview_local_shade"])
    overview_origin_shade = ColumnDataSource(data=sources["overview_origin_shade"])
    overview_rotated_shade = ColumnDataSource(data=sources["overview_rotated_shade"])
    overview_origin_model = ColumnDataSource(data=sources["overview_origin_model"])
    overview_rotated_model = ColumnDataSource(data=sources["overview_rotated_model"])
    overview_b_chord = ColumnDataSource(data=sources["overview_b_chord"])
    overview_step = ColumnDataSource(data=sources["overview_step"])
    overview_points = ColumnDataSource(data=sources["overview_points"])
    overview_labels = ColumnDataSource(data=sources["overview_labels"])
    finite_shade = ColumnDataSource(data=sources["finite_shade"])
    finite_exact = ColumnDataSource(data=sources["finite_exact"])
    finite_ideal = ColumnDataSource(data=sources["finite_ideal"])
    finite_points = ColumnDataSource(data=sources["finite_points"])
    finite_labels = ColumnDataSource(data=sources["finite_labels"])
    summary = Div(text=_format_summary(spiral_step_geometry(a, b, t, delta)), width=360, styles=_panel_styles())
    legend = Div(
        text=f"""
<b>Color key</b><br>
<span style="color:{SPIRAL_COLOR}">spiral</span>: path of Z(t)<br>
<span style="color:{A_COLOR}">A</span>: radial change<br>
<span style="color:{B_COLOR}">B</span>: turning chord<br>
<span style="color:{M_COLOR}">M</span>: finite movement<br>
dashed lines/circles: rotation guides and infinitesimal model
""",
        width=360,
        styles={
            "font-family": "Source Sans 3, system-ui, sans-serif",
            "font-size": "13px",
            "line-height": "1.55",
            "padding": "8px 0",
        },
    )

    overview = figure(
        width=520,
        height=520,
        x_range=Range1d(-2.2, 2.2, bounds=(-5.0, 5.0)),
        y_range=Range1d(-2.2, 2.2, bounds=(-5.0, 5.0)),
        x_axis_label="Re",
        y_axis_label="Im",
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
        toolbar_location="above",
        title="1. Actual finite step on the spiral",
    )
    _quiet_axes(overview)
    overview.line(
        x="x",
        y="y",
        source=unit_source,
        line_color=UNIT_COLOR,
        line_dash="dashed",
        line_width=1.5,
        alpha=0.75,
    )
    overview.line(x="x", y="y", source=curve_source, line_color=SPIRAL_COLOR, line_width=1.6, alpha=0.72)
    overview.patch(
        x="x",
        y="y",
        source=overview_origin_shade,
        fill_color=M_COLOR,
        fill_alpha=0.10,
        line_alpha=0,
    )
    overview.patch(
        x="x",
        y="y",
        source=overview_rotated_shade,
        fill_color=M_COLOR,
        fill_alpha=0.10,
        line_alpha=0,
    )
    overview.patch(
        x="x",
        y="y",
        source=overview_local_shade,
        fill_color=M_COLOR,
        fill_alpha=0.18,
        line_alpha=0,
    )
    overview.multi_line(
        xs="xs",
        ys="ys",
        source=overview_guides,
        line_color="color",
        line_dash="dashed",
        line_width=1.4,
        alpha=0.58,
    )
    overview.multi_line(
        xs="xs",
        ys="ys",
        source=overview_origin_model,
        line_color="color",
        line_width=3,
        alpha=0.74,
    )
    overview.multi_line(
        xs="xs",
        ys="ys",
        source=overview_rotated_model,
        line_color="color",
        line_dash="dashed",
        line_width=3,
        alpha=0.72,
    )
    overview.multi_line(
        xs="xs",
        ys="ys",
        source=overview_b_chord,
        line_color="color",
        line_dash="dashed",
        line_width=3,
        alpha=0.78,
    )
    overview.multi_line(
        xs="xs",
        ys="ys",
        source=overview_step,
        line_color="color",
        line_width=4,
        alpha=0.92,
    )
    overview.scatter(x="x", y="y", source=overview_points, size="size", color="color", alpha=0.96)
    overview.add_layout(_labels(LabelSet, overview_labels, size="12px"))
    overview.add_layout(
        Label(
            x=sources["overview_math_label"]["x"],
            y=sources["overview_math_label"]["y"],
            text=TeX(text=r"e^{ibt}(a+ib)", inline=True),
            x_offset=8,
            y_offset=8,
            text_font_size="12px",
            text_color="#111827",
            background_fill_color="#ffffff",
            background_fill_alpha=0.72,
        )
    )

    finite = figure(
        width=520,
        height=520,
        x_range=Range1d(-0.35, 1.35, bounds=(-2.0, 2.0)),
        y_range=Range1d(-0.35, 1.35, bounds=(-2.0, 2.0)),
        x_axis_label="radial component divided by |Z|δ",
        y_axis_label="turning component divided by |Z|δ",
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
        toolbar_location="above",
        title="2. Zoom: finite A,B approach a right triangle",
    )
    _quiet_axes(finite)
    finite.patch(
        x="x",
        y="y",
        source=finite_shade,
        fill_color=M_COLOR,
        fill_alpha=0.18,
        line_alpha=0,
    )
    finite.multi_line(
        xs="xs",
        ys="ys",
        source=finite_exact,
        line_color="color",
        line_width=5,
        alpha=0.9,
    )
    finite.multi_line(
        xs="xs",
        ys="ys",
        source=finite_ideal,
        line_color="color",
        line_dash="dashed",
        line_width=4,
        alpha=0.82,
    )
    finite.scatter(x="x", y="y", source=finite_points, size="size", color="color", alpha=0.96)
    finite.add_layout(_labels(LabelSet, finite_labels, size="12px"))

    slider = Slider(title="δ", start=0.02, end=1.2, step=0.02, value=delta, width=360)
    callback = CustomJS(
        args=dict(
            slider=slider,
            overview_guides=overview_guides,
            overview_local_shade=overview_local_shade,
            overview_origin_shade=overview_origin_shade,
            overview_rotated_shade=overview_rotated_shade,
            overview_origin_model=overview_origin_model,
            overview_rotated_model=overview_rotated_model,
            overview_b_chord=overview_b_chord,
            overview_step=overview_step,
            overview_points=overview_points,
            overview_labels=overview_labels,
            finite_shade=finite_shade,
            finite_exact=finite_exact,
            finite_ideal=finite_ideal,
            finite_points=finite_points,
            finite_labels=finite_labels,
            summary=summary,
        ),
        code=(
            _BOKEH_UPDATE_JS.replace("__A__", repr(a))
            .replace("__B__", repr(b))
            .replace("__T__", repr(t))
            .replace("__GUIDE_COLOR__", GUIDE_COLOR)
            .replace("__UNIT_COLOR__", UNIT_COLOR)
            .replace("__POINT_COLOR__", POINT_COLOR)
            .replace("__A_COLOR__", A_COLOR)
            .replace("__B_COLOR__", B_COLOR)
            .replace("__M_COLOR__", M_COLOR)
        ),
    )
    slider.js_on_change("value", callback)

    return column(
        row(overview, finite),
        row(column(slider, summary, legend, width=380)),
        sizing_mode="stretch_width",
    )


def export_spiral_velocity_html(
    path: str | Path = "docs/assets/plots/spiral_velocity.html",
    a: float = 0.18,
    b: float = 1.0,
    t: float = 1.25,
    delta: float = 0.22,
) -> Path:
    """Write a standalone Bokeh HTML file for the spiral velocity example."""
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
        make_spiral_velocity_bokeh(a=a, b=b, t=t, delta=delta),
        filename=str(out),
        resources=INLINE,
        title="Spiral velocity geometry",
    )
    return out


def _panel_styles() -> dict[str, str]:
    return {
        "font-family": "JetBrains Mono, Menlo, Consolas, monospace",
        "font-size": "12px",
        "line-height": "1.55",
        "border": "1px solid #d8dee9",
        "border-radius": "6px",
        "padding": "12px",
        "background": "#fbfbfc",
    }


def _quiet_axes(plot) -> None:
    plot.grid.grid_line_alpha = 0.25
    plot.xaxis.axis_label_text_font_style = "normal"
    plot.yaxis.axis_label_text_font_style = "normal"


def _labels(label_set_class, source, size: str):
    return label_set_class(
        x="x",
        y="y",
        text="label",
        source=source,
        x_offset="x_offset",
        y_offset="y_offset",
        text_font_size=size,
        text_color="#111827",
        background_fill_color="#ffffff",
        background_fill_alpha=0.72,
    )


def _format_complex(z: complex) -> str:
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.3f} {sign} {abs(z.imag):.3f}i"


def _format_summary(geometry: SpiralStepGeometry) -> str:
    linear_error = abs(geometry.m - geometry.m_linear)
    return f"""
<b>Finite δ</b><br>
M = {_format_complex(geometry.m)}<br>
A + B = {_format_complex(geometry.a_exact + geometry.b_exact)}<br>
angle(A,B) = {geometry.angle_degrees:.2f} deg<br>
error from 90 deg = {geometry.perpendicular_error_degrees:.2f} deg<br>
<br>
<b>As δ tends to 0</b><br>
A/(|Z|δ) -> a = 0.180<br>
B/(|Z|δ) -> ib = 1.000i<br>
M/(|Z|δ) -> a+ib<br>
|M - model| = {linear_error:.4f}
"""


def _local(z: complex, p: complex, theta: float, scale: float = 1.0) -> complex:
    """Return point ``p`` in the local frame based at ``z`` and angle ``theta``."""
    return (p - z) * complex(math.cos(-theta), math.sin(-theta)) / scale


def _bokeh_sources(a: float, b: float, t: float, delta: float) -> dict[str, dict[str, list]]:
    geom = spiral_step_geometry(a, b, t, delta)
    theta = b * t
    next_theta = b * (t + delta)
    max_radius = max(abs(geom.z), abs(geom.z_next)) * 1.18
    old_ray = max_radius * complex(math.cos(theta), math.sin(theta))
    new_ray = max_radius * complex(math.cos(next_theta), math.sin(next_theta))
    arc_old = abs(geom.z) * np.exp(1j * np.linspace(theta, next_theta, 80))
    arc_next = abs(geom.z_next) * np.exp(1j * np.linspace(theta, next_theta, 80))
    scale = abs(geom.z) * delta

    z = geom.z
    z_next = geom.z_next
    radial_point = geom.radial_point
    turn_point = abs(z) * complex(math.cos(next_theta), math.sin(next_theta))
    model_a = complex(a, 0.0)
    model_m = complex(a, b)
    rotation = complex(math.cos(theta), math.sin(theta))
    rotated_a = model_a * rotation
    rotated_m = model_m * rotation
    linear_end = z + geom.m_linear
    local_origin = 0 + 0j
    local_radial = _local(z, radial_point, theta, scale)
    local_next = _local(z, z_next, theta, scale)
    local_ideal = _local(z, linear_end, theta, scale)

    return {
        "overview_guides": {
            "xs": [
                [0.0, old_ray.real],
                [0.0, new_ray.real],
                arc_old.real.tolist(),
                arc_next.real.tolist(),
            ],
            "ys": [
                [0.0, old_ray.imag],
                [0.0, new_ray.imag],
                arc_old.imag.tolist(),
                arc_next.imag.tolist(),
            ],
            "color": [GUIDE_COLOR, GUIDE_COLOR, UNIT_COLOR, UNIT_COLOR],
        },
        "overview_local_shade": {
            "x": [z.real, radial_point.real, z_next.real],
            "y": [z.imag, radial_point.imag, z_next.imag],
        },
        "overview_origin_shade": {
            "x": [0.0, model_a.real, model_m.real],
            "y": [0.0, model_a.imag, model_m.imag],
        },
        "overview_rotated_shade": {
            "x": [0.0, rotated_a.real, rotated_m.real],
            "y": [0.0, rotated_a.imag, rotated_m.imag],
        },
        "overview_origin_model": {
            "xs": [[0.0, model_a.real], [model_a.real, model_m.real], [0.0, model_m.real]],
            "ys": [[0.0, model_a.imag], [model_a.imag, model_m.imag], [0.0, model_m.imag]],
            "color": COMPONENT_COLORS,
        },
        "overview_rotated_model": {
            "xs": [
                [0.0, rotated_a.real],
                [rotated_a.real, rotated_m.real],
                [0.0, rotated_m.real],
            ],
            "ys": [
                [0.0, rotated_a.imag],
                [rotated_a.imag, rotated_m.imag],
                [0.0, rotated_m.imag],
            ],
            "color": COMPONENT_COLORS,
        },
        "overview_b_chord": {
            "xs": [[z.real, turn_point.real]],
            "ys": [[z.imag, turn_point.imag]],
            "color": [B_COLOR],
        },
        "overview_step": {
            "xs": [
                [z.real, radial_point.real],
                [radial_point.real, z_next.real],
                [z.real, z_next.real],
            ],
            "ys": [
                [z.imag, radial_point.imag],
                [radial_point.imag, z_next.imag],
                [z.imag, z_next.imag],
            ],
            "color": COMPONENT_COLORS,
        },
        "overview_points": {
            "x": [0.0, z.real, radial_point.real, z_next.real, model_m.real, rotated_m.real],
            "y": [0.0, z.imag, radial_point.imag, z_next.imag, model_m.imag, rotated_m.imag],
            "color": [POINT_COLOR, POINT_COLOR, A_COLOR, M_COLOR, M_COLOR, M_COLOR],
            "size": [7, 9, 8, 9, 8, 8],
        },
        "overview_labels": {
            "x": [
                0.0,
                z.real,
                z_next.real,
                model_m.real,
            ],
            "y": [
                0.0,
                z.imag,
                z_next.imag,
                model_m.imag,
            ],
            "label": ["0", "Z(t)", "Z(t+δ)", "a+ib"],
            "x_offset": [8, 8, 8, 8],
            "y_offset": [8, 8, 8, 8],
        },
        "overview_math_label": {
            "x": rotated_m.real,
            "y": rotated_m.imag,
        },
        "finite_shade": {
            "x": [local_origin.real, local_radial.real, local_next.real],
            "y": [local_origin.imag, local_radial.imag, local_next.imag],
        },
        "finite_exact": {
            "xs": [
                [local_origin.real, local_radial.real],
                [local_radial.real, local_next.real],
                [local_origin.real, local_next.real],
            ],
            "ys": [
                [local_origin.imag, local_radial.imag],
                [local_radial.imag, local_next.imag],
                [local_origin.imag, local_next.imag],
            ],
            "color": COMPONENT_COLORS,
        },
        "finite_ideal": {
            "xs": [[0.0, a], [a, a], [0.0, a]],
            "ys": [[0.0, 0.0], [0.0, b], [0.0, b]],
            "color": COMPONENT_COLORS,
        },
        "finite_points": {
            "x": [local_origin.real, local_radial.real, local_next.real, local_ideal.real],
            "y": [local_origin.imag, local_radial.imag, local_next.imag, local_ideal.imag],
            "color": [POINT_COLOR, A_COLOR, M_COLOR, M_COLOR],
            "size": [9, 9, 9, 8],
        },
        "finite_labels": {
            "x": [
                local_origin.real,
                local_radial.real / 2,
                (local_radial.real + local_next.real) / 2,
                local_next.real / 2,
                local_ideal.real,
            ],
            "y": [
                local_origin.imag,
                local_radial.imag / 2,
                (local_radial.imag + local_next.imag) / 2,
                local_next.imag / 2,
                local_ideal.imag,
            ],
            "label": ["Z(t)", "A", "B", "M", "a+ib limit"],
            "x_offset": [7, 8, 8, -34, 8],
            "y_offset": [-17, -18, 8, 6, 8],
        },
    }


_BOKEH_UPDATE_JS = r"""
const a = __A__;
const b = __B__;
const t = __T__;
const delta = slider.value;
const GUIDE_COLOR = "__GUIDE_COLOR__";
const UNIT_COLOR = "__UNIT_COLOR__";
const POINT_COLOR = "__POINT_COLOR__";
const A_COLOR = "__A_COLOR__";
const B_COLOR = "__B_COLOR__";
const M_COLOR = "__M_COLOR__";
const COMPONENT_COLORS = [A_COLOR, B_COLOR, M_COLOR];

function c(re, im) {
  return {re: re, im: im};
}
function add(u, v) {
  return c(u.re + v.re, u.im + v.im);
}
function sub(u, v) {
  return c(u.re - v.re, u.im - v.im);
}
function mul(u, r) {
  return c(u.re * r, u.im * r);
}
function muli(u) {
  return c(-u.im, u.re);
}
function abs(u) {
  return Math.hypot(u.re, u.im);
}
function dot(u, v) {
  return u.re * v.re + u.im * v.im;
}
function expPoint(radius, angle) {
  return c(radius * Math.cos(angle), radius * Math.sin(angle));
}
function rotate(u, angle) {
  const ca = Math.cos(angle);
  const sa = Math.sin(angle);
  return c(u.re * ca - u.im * sa, u.re * sa + u.im * ca);
}
function localPoint(base, point, angle, scale) {
  return mul(rotate(sub(point, base), -angle), 1 / scale);
}
function fmt(u) {
  const sign = u.im >= 0 ? " + " : " - ";
  return `${u.re.toFixed(3)}${sign}${Math.abs(u.im).toFixed(3)}i`;
}

const theta = b * t;
const thetaNext = b * (t + delta);
const z = expPoint(Math.exp(a * t), theta);
const zNext = expPoint(Math.exp(a * (t + delta)), thetaNext);
const radialPoint = expPoint(abs(zNext), theta);
const turnPoint = expPoint(abs(z), thetaNext);
const m = sub(zNext, z);
const aExact = sub(radialPoint, z);
const bExact = sub(zNext, radialPoint);
const aLinear = mul(z, a * delta);
const bLinear = mul(muli(z), b * delta);
const mLinear = add(aLinear, bLinear);
const linearEnd = add(z, mLinear);
const modelA = c(a, 0);
const modelM = c(a, b);
const rotatedA = rotate(modelA, theta);
const rotatedM = rotate(modelM, theta);

const angleCos = dot(aExact, bExact) / (abs(aExact) * abs(bExact));
const angle = Math.acos(Math.max(-1, Math.min(1, angleCos))) * 180 / Math.PI;
const perpendicularError = Math.abs(angle - 90);
const linearError = abs(sub(m, mLinear));
const scale = abs(z) * delta;

const maxRadius = Math.max(abs(z), abs(zNext)) * 1.18;
const oldRay = expPoint(maxRadius, theta);
const newRay = expPoint(maxRadius, thetaNext);
const oldArcX = [];
const oldArcY = [];
const nextArcX = [];
const nextArcY = [];
for (let k = 0; k < 80; k += 1) {
  const u = theta + (thetaNext - theta) * k / 79;
  const oldPoint = expPoint(abs(z), u);
  const nextPoint = expPoint(abs(zNext), u);
  oldArcX.push(oldPoint.re);
  oldArcY.push(oldPoint.im);
  nextArcX.push(nextPoint.re);
  nextArcY.push(nextPoint.im);
}

overview_guides.data = {
  xs: [[0, oldRay.re], [0, newRay.re], oldArcX, nextArcX],
  ys: [[0, oldRay.im], [0, newRay.im], oldArcY, nextArcY],
  color: [GUIDE_COLOR, GUIDE_COLOR, UNIT_COLOR, UNIT_COLOR],
};
overview_local_shade.data = {
  x: [z.re, radialPoint.re, zNext.re],
  y: [z.im, radialPoint.im, zNext.im],
};
overview_origin_shade.data = {
  x: [0, modelA.re, modelM.re],
  y: [0, modelA.im, modelM.im],
};
overview_rotated_shade.data = {
  x: [0, rotatedA.re, rotatedM.re],
  y: [0, rotatedA.im, rotatedM.im],
};
overview_origin_model.data = {
  xs: [[0, modelA.re], [modelA.re, modelM.re], [0, modelM.re]],
  ys: [[0, modelA.im], [modelA.im, modelM.im], [0, modelM.im]],
  color: COMPONENT_COLORS,
};
overview_rotated_model.data = {
  xs: [
    [0, rotatedA.re],
    [rotatedA.re, rotatedM.re],
    [0, rotatedM.re],
  ],
  ys: [
    [0, rotatedA.im],
    [rotatedA.im, rotatedM.im],
    [0, rotatedM.im],
  ],
  color: COMPONENT_COLORS,
};
overview_b_chord.data = {
  xs: [[z.re, turnPoint.re]],
  ys: [[z.im, turnPoint.im]],
  color: [B_COLOR],
};
overview_step.data = {
  xs: [[z.re, radialPoint.re], [radialPoint.re, zNext.re], [z.re, zNext.re]],
  ys: [[z.im, radialPoint.im], [radialPoint.im, zNext.im], [z.im, zNext.im]],
  color: COMPONENT_COLORS,
};
overview_points.data = {
  x: [0, z.re, radialPoint.re, zNext.re, modelM.re, rotatedM.re],
  y: [0, z.im, radialPoint.im, zNext.im, modelM.im, rotatedM.im],
  color: [POINT_COLOR, POINT_COLOR, A_COLOR, M_COLOR, M_COLOR, M_COLOR],
  size: [7, 9, 8, 9, 8, 8],
};
overview_labels.data = {
  x: [
    0,
    z.re,
    zNext.re,
    modelM.re,
  ],
  y: [
    0,
    z.im,
    zNext.im,
    modelM.im,
  ],
  label: ["0", "Z(t)", "Z(t+δ)", "a+ib"],
  x_offset: [8, 8, 8, 8],
  y_offset: [8, 8, 8, 8],
};

const localOrigin = c(0, 0);
const localRadial = localPoint(z, radialPoint, theta, scale);
const localNext = localPoint(z, zNext, theta, scale);
const localIdeal = localPoint(z, linearEnd, theta, scale);

finite_shade.data = {
  x: [localOrigin.re, localRadial.re, localNext.re],
  y: [localOrigin.im, localRadial.im, localNext.im],
};
finite_exact.data = {
  xs: [
    [localOrigin.re, localRadial.re],
    [localRadial.re, localNext.re],
    [localOrigin.re, localNext.re],
  ],
  ys: [
    [localOrigin.im, localRadial.im],
    [localRadial.im, localNext.im],
    [localOrigin.im, localNext.im],
  ],
  color: COMPONENT_COLORS,
};
finite_ideal.data = {
  xs: [[0, a], [a, a], [0, a]],
  ys: [[0, 0], [0, b], [0, b]],
  color: COMPONENT_COLORS,
};
finite_points.data = {
  x: [localOrigin.re, localRadial.re, localNext.re, localIdeal.re],
  y: [localOrigin.im, localRadial.im, localNext.im, localIdeal.im],
  color: [POINT_COLOR, A_COLOR, M_COLOR, M_COLOR],
  size: [9, 9, 9, 8],
};
finite_labels.data = {
  x: [
    localOrigin.re,
    localRadial.re / 2,
    (localRadial.re + localNext.re) / 2,
    localNext.re / 2,
    localIdeal.re,
  ],
  y: [
    localOrigin.im,
    localRadial.im / 2,
    (localRadial.im + localNext.im) / 2,
    localNext.im / 2,
    localIdeal.im,
  ],
  label: ["Z(t)", "A", "B", "M", "a+ib limit"],
  x_offset: [7, 8, 8, -34, 8],
  y_offset: [-17, -18, 8, 6, 8],
};

summary.text = `
<b>Finite δ</b><br>
M = ${fmt(m)}<br>
A + B = ${fmt(add(aExact, bExact))}<br>
angle(A,B) = ${angle.toFixed(2)} deg<br>
error from 90 deg = ${perpendicularError.toFixed(2)} deg<br>
<br>
<b>As δ tends to 0</b><br>
A/(|Z|δ) -> a = ${a.toFixed(3)}<br>
B/(|Z|δ) -> ib = ${b.toFixed(3)}i<br>
M/(|Z|δ) -> a+ib<br>
|M - model| = ${linearError.toFixed(4)}
`;
"""
