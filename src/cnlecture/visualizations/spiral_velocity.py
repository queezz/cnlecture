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
V_COLOR = "#0072b2"

COMPONENT_COLORS = [A_COLOR, B_COLOR, M_COLOR]

# Radius of the small ``bδ`` angle arc drawn at the origin inset.
BDELTA_ARC_RADIUS = 0.34

# Continue the plotted spiral a little past ``Z(t+δ)``.
SPIRAL_EXTRA_ANGLE = math.pi / 12

# Padding around the overview geometry used for panel 1 auto-fit.
OVERVIEW_FIT_PADDING = 1.0

# Positive values move the panel-1 view center upward, placing the origin lower.
OVERVIEW_VERTICAL_BIAS = 0.18


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
    a: float = 0.4,
    b: float = 0.95,
    t: float = 1.85,
    delta: float = 0.2,
):
    """Build a Bokeh figure for the spiral velocity right-triangle argument."""
    try:
        from bokeh.layouts import column, row
        from bokeh.models import (
            Arrow,
            Checkbox,
            ColumnDataSource,
            CustomJS,
            Div,
            Label,
            LabelSet,
            Range1d,
            Slider,
            TeX,
            VeeHead,
        )
        from bokeh.plotting import figure
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Bokeh is required for this interactive visualization. "
            'Install it with `pip install -e ".[dev]"`.'
        ) from exc

    sources = _bokeh_sources(a, b, t, delta)
    t_start, t_end = _spiral_plot_window(b, t, delta)
    curve = spiral_curve(a, b, t_min=t_start, t_max=t_end)
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
    overview_bdelta_arc = ColumnDataSource(data=sources["overview_bdelta_arc"])
    finite_shade = ColumnDataSource(data=sources["finite_shade"])
    finite_exact = ColumnDataSource(data=sources["finite_exact"])
    finite_ideal = ColumnDataSource(data=sources["finite_ideal"])
    finite_points = ColumnDataSource(data=sources["finite_points"])
    finite_labels = ColumnDataSource(data=sources["finite_labels"])
    summary = Div(
        text=_format_summary(spiral_step_geometry(a, b, t, delta), a, b),
        width=360,
        styles=_panel_styles(),
    )
    legend = Div(
        text=f"""
<b>Color key</b><br>
<span style="color:{SPIRAL_COLOR}">spiral</span>: path of Z(t)<br>
<span style="color:{A_COLOR}">A</span>: radial change<br>
<span style="color:{B_COLOR}">B</span>: turning chord<br>
<span style="color:{M_COLOR}">M</span>: finite movement<br>
<span style="color:{V_COLOR}">V</span>: velocity (a+ib)Z<br>
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

    overview_x0, overview_x1, overview_y0, overview_y1 = _overview_view(sources)
    overview = figure(
        width=560,
        height=560,
        x_range=Range1d(overview_x0, overview_x1, bounds=(-16.0, 16.0)),
        y_range=Range1d(overview_y0, overview_y1, bounds=(-16.0, 16.0)),
        x_axis_label="Re",
        y_axis_label="Im",
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
        toolbar_location="above",
        title="1. Velocity on the spiral:  V = (a+ib) Z",
    )
    _quiet_axes(overview)
    overview.grid.grid_line_alpha = 0.12
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
    overview.line(
        x="x",
        y="y",
        source=overview_bdelta_arc,
        line_color=GUIDE_COLOR,
        line_width=1.8,
        alpha=0.9,
    )
    overview.multi_line(
        xs="xs",
        ys="ys",
        source=overview_origin_model,
        line_color="color",
        line_width=2,
        alpha=0.5,
    )
    overview.multi_line(
        xs="xs",
        ys="ys",
        source=overview_rotated_model,
        line_color="color",
        line_width=2.5,
        alpha=0.7,
    )
    overview.multi_line(
        xs="xs",
        ys="ys",
        source=overview_b_chord,
        line_color="color",
        line_width=2.5,
        alpha=0.7,
    )
    overview.multi_line(
        xs="xs",
        ys="ys",
        source=overview_step,
        line_color="color",
        line_width=2.5,
        alpha=0.95,
    )
    overview.scatter(x="x", y="y", source=overview_points, size="size", color="color", alpha=0.96)
    overview.add_layout(_labels(LabelSet, overview_labels, size="12px"))
    rotated_math_label = Label(
        x=sources["overview_math_label"]["x"],
        y=sources["overview_math_label"]["y"],
        text=TeX(text=r"e^{ibt}(a+ib)", inline=True),
        x_offset=4,
        y_offset=4,
        text_font_size="12px",
        text_color="#111827",
        background_fill_color="#ffffff",
        background_fill_alpha=0.72,
    )
    overview.add_layout(rotated_math_label)

    velocity = sources["overview_velocity"]
    velocity_arrow = Arrow(
        end=VeeHead(size=16, fill_color=V_COLOR, line_color=V_COLOR),
        x_start=velocity["x_start"][0],
        y_start=velocity["y_start"][0],
        x_end=velocity["x_end"][0],
        y_end=velocity["y_end"][0],
        line_color=V_COLOR,
        line_width=4,
        line_alpha=0.95,
        level="underlay",
    )
    overview.add_layout(velocity_arrow)
    velocity_label = Label(
        x=velocity["x_start"][0] + 0.58 * (velocity["x_end"][0] - velocity["x_start"][0]),
        y=velocity["y_start"][0] + 0.58 * (velocity["y_end"][0] - velocity["y_start"][0]),
        text=TeX(text="V", inline=True),
        x_offset=7,
        y_offset=5,
        text_font_size="16px",
        text_color=V_COLOR,
        background_fill_color="#ffffff",
        background_fill_alpha=0.72,
    )
    overview.add_layout(velocity_label)
    formula_label = Label(
        x=1.03,
        y=0.24,
        text=TeX(text=r"Z(t)=e^{at}e^{ibt}", inline=True),
        x_offset=8,
        y_offset=0,
        text_font_size="12px",
        text_color="#111827",
        background_fill_color="#ffffff",
        background_fill_alpha=0.72,
    )
    overview.add_layout(formula_label)

    finite_x0, finite_x1, finite_y0, finite_y1 = _finite_view(sources, a, b)
    finite = figure(
        width=560,
        height=560,
        x_range=Range1d(finite_x0, finite_x1, bounds=(-8.0, 8.0)),
        y_range=Range1d(finite_y0, finite_y1, bounds=(-8.0, 8.0)),
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

    a_slider = Slider(title="a  (growth rate)", start=0.0, end=0.45, step=0.01, value=a, width=360)
    b_slider = Slider(title="b  (angular speed)", start=0.3, end=1.8, step=0.05, value=b, width=360)
    t_slider = Slider(title="t  (point on spiral)", start=0.2, end=2.6, step=0.05, value=t, width=360)
    delta_slider = Slider(title="δ  (finite step)", start=0.02, end=1.2, step=0.02, value=delta, width=360)
    autofit_checkbox = Checkbox(label="Auto-fit panel 1", active=True, width=360)
    callback = CustomJS(
        args=dict(
            a_slider=a_slider,
            b_slider=b_slider,
            t_slider=t_slider,
            delta_slider=delta_slider,
            autofit_checkbox=autofit_checkbox,
            curve_source=curve_source,
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
            overview_bdelta_arc=overview_bdelta_arc,
            velocity_arrow=velocity_arrow,
            velocity_label=velocity_label,
            rotated_math_label=rotated_math_label,
            x_range=overview.x_range,
            y_range=overview.y_range,
            finite_x_range=finite.x_range,
            finite_y_range=finite.y_range,
            finite_shade=finite_shade,
            finite_exact=finite_exact,
            finite_ideal=finite_ideal,
            finite_points=finite_points,
            finite_labels=finite_labels,
            summary=summary,
        ),
        code=(
            _BOKEH_UPDATE_JS.replace("__BDELTA_ARC_RADIUS__", repr(BDELTA_ARC_RADIUS))
            .replace("__SPIRAL_EXTRA_ANGLE__", repr(SPIRAL_EXTRA_ANGLE))
            .replace("__OVERVIEW_FIT_PADDING__", repr(OVERVIEW_FIT_PADDING))
            .replace("__OVERVIEW_VERTICAL_BIAS__", repr(OVERVIEW_VERTICAL_BIAS))
            .replace("__GUIDE_COLOR__", GUIDE_COLOR)
            .replace("__UNIT_COLOR__", UNIT_COLOR)
            .replace("__POINT_COLOR__", POINT_COLOR)
            .replace("__A_COLOR__", A_COLOR)
            .replace("__B_COLOR__", B_COLOR)
            .replace("__M_COLOR__", M_COLOR)
        ),
    )
    for control in (a_slider, b_slider, t_slider, delta_slider):
        control.js_on_change("value", callback)
    autofit_checkbox.js_on_change("active", callback)

    controls = column(a_slider, b_slider, t_slider, delta_slider, autofit_checkbox, width=380)
    return column(
        row(overview, finite),
        row(controls, row(summary, legend)),
        sizing_mode="stretch_width",
    )


def export_spiral_velocity_html(
    path: str | Path = "docs/assets/plots/spiral_velocity.html",
    a: float = 0.4,
    b: float = 0.95,
    t: float = 1.85,
    delta: float = 0.2,
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


def _format_summary(geometry: SpiralStepGeometry, a: float, b: float) -> str:
    linear_error = abs(geometry.m - geometry.m_linear)
    return f"""
<b>Finite δ</b><br>
M = {_format_complex(geometry.m)}<br>
A + B = {_format_complex(geometry.a_exact + geometry.b_exact)}<br>
angle(A,B) = {geometry.angle_degrees:.2f} deg<br>
error from 90 deg = {geometry.perpendicular_error_degrees:.2f} deg<br>
<br>
<b>As δ tends to 0</b><br>
A/(|Z|δ) -> a = {a:.3f}<br>
B/(|Z|δ) -> ib = {b:.3f}i<br>
M/(|Z|δ) -> a+ib<br>
|M - model| = {linear_error:.4f}
"""


def _outside_circle_label_point(point: complex, offset: float = 0.14) -> complex:
    """Return a label anchor just beyond ``point`` on its origin-centered circle."""
    radius = abs(point)
    if radius == 0:
        return point + offset
    return point * ((radius + offset) / radius)


def _local(z: complex, p: complex, theta: float, scale: float = 1.0) -> complex:
    """Return point ``p`` in the local frame based at ``z`` and angle ``theta``."""
    return (p - z) * complex(math.cos(-theta), math.sin(-theta)) / scale


def _overview_radius(sources: dict) -> float:
    """Half-size of the origin-centred square view that frames the overview."""
    xs = sources["overview_fit"]["x"]
    ys = sources["overview_fit"]["y"]
    max_r = max([1.05, *(math.hypot(x, y) for x, y in zip(xs, ys))])
    return max(1.3, OVERVIEW_FIT_PADDING * max_r)


def _overview_view(sources: dict) -> tuple[float, float, float, float]:
    """Square panel-1 view, shifted upward when the fitted geometry allows it."""
    ys = sources["overview_fit"]["y"]
    radius = _overview_radius(sources)
    desired_center_y = OVERVIEW_VERTICAL_BIAS * radius
    min_center_y = max(ys) - radius
    max_center_y = min(ys) + radius
    center_y = min(max(desired_center_y, min_center_y), max_center_y)
    return -radius, radius, center_y - radius, center_y + radius


def _spiral_plot_window(b: float, t: float, delta: float) -> tuple[float, float]:
    """Return the ``t`` interval for the overview spiral segment."""
    angular_speed = max(abs(b), 1e-6)
    period = 2 * math.pi / angular_speed
    return t - 2.5 * period, t + delta + SPIRAL_EXTRA_ANGLE / angular_speed


def _finite_view(sources: dict, a: float, b: float) -> tuple[float, float, float, float]:
    """Right-biased square view ``(x0, x1, y0, y1)`` framing the zoom triangle."""
    xs = [*sources["finite_points"]["x"], 0.0, a, a]
    ys = [*sources["finite_points"]["y"], 0.0, 0.0, b]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cx, cy = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)
    half = max(0.4, 0.5 * max(max_x - min_x, max_y - min_y) * 1.22)
    return (cx - 1.5 * half, cx + 0.5 * half, cy - half, cy + half)


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
    model_m_label = _outside_circle_label_point(model_m)
    rotated_m_label = _outside_circle_label_point(rotated_m)
    linear_end = z + geom.m_linear
    local_origin = 0 + 0j
    local_radial = _local(z, radial_point, theta, scale)
    local_next = _local(z, z_next, theta, scale)
    local_ideal = _local(z, linear_end, theta, scale)

    v_vec = complex(a, b) * z
    v_end = z + v_vec
    _, spiral_t_end = _spiral_plot_window(b, t, delta)
    spiral_end = spiral_point(a, b, spiral_t_end)
    bdelta_arc = BDELTA_ARC_RADIUS * np.exp(1j * np.linspace(theta, next_theta, 40))
    bdelta_mid = 0.5 * (theta + next_theta)
    bdelta_label = (BDELTA_ARC_RADIUS + 0.13) * complex(
        math.cos(bdelta_mid), math.sin(bdelta_mid)
    )
    unit_point = complex(1.0, 0.0)

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
            "x": [
                0.0,
                z.real,
                radial_point.real,
                z_next.real,
                model_m.real,
                rotated_m.real,
                unit_point.real,
            ],
            "y": [
                0.0,
                z.imag,
                radial_point.imag,
                z_next.imag,
                model_m.imag,
                rotated_m.imag,
                unit_point.imag,
            ],
            "color": [POINT_COLOR, POINT_COLOR, A_COLOR, M_COLOR, M_COLOR, M_COLOR, POINT_COLOR],
            "size": [5, 5, 5, 5, 5, 5, 5],
        },
        "overview_labels": {
            "x": [
                0.0,
                z.real,
                z_next.real,
                model_m_label.real,
                unit_point.real,
                bdelta_label.real,
            ],
            "y": [
                0.0,
                z.imag,
                z_next.imag,
                model_m_label.imag,
                unit_point.imag,
                bdelta_label.imag,
            ],
            "label": ["0", "Z(t)", "Z(t+δ)", "a+ib", "1", "bδ"],
            "x_offset": [-12, 11, -12, 4, 7, 2],
            "y_offset": [-6, 2, 12, 4, -10, 4],
        },
        "overview_math_label": {
            "x": rotated_m_label.real,
            "y": rotated_m_label.imag,
        },
        "overview_velocity": {
            "x_start": [z.real],
            "y_start": [z.imag],
            "x_end": [v_end.real],
            "y_end": [v_end.imag],
        },
        "overview_fit": {
            "x": [
                0.0,
                z.real,
                radial_point.real,
                z_next.real,
                model_m.real,
                rotated_m.real,
                unit_point.real,
                spiral_end.real,
            ],
            "y": [
                0.0,
                z.imag,
                radial_point.imag,
                z_next.imag,
                model_m.imag,
                rotated_m.imag,
                unit_point.imag,
                spiral_end.imag,
            ],
        },
        "overview_bdelta_arc": {
            "x": bdelta_arc.real.tolist(),
            "y": bdelta_arc.imag.tolist(),
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
            "size": [6, 6, 6, 5],
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
const a = a_slider.value;
const b = b_slider.value;
const t = t_slider.value;
const delta = delta_slider.value;
const autofitPanel1 = autofit_checkbox.active;
const ARC_RADIUS = __BDELTA_ARC_RADIUS__;
const SPIRAL_EXTRA_ANGLE = __SPIRAL_EXTRA_ANGLE__;
const OVERVIEW_FIT_PADDING = __OVERVIEW_FIT_PADDING__;
const OVERVIEW_VERTICAL_BIAS = __OVERVIEW_VERTICAL_BIAS__;
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
function outsideCircleLabelPoint(point, offset = 0.14) {
  const radius = abs(point);
  if (radius === 0) {
    return c(offset, 0);
  }
  return mul(point, (radius + offset) / radius);
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
const modelMLabel = outsideCircleLabelPoint(modelM);
const rotatedMLabel = outsideCircleLabelPoint(rotatedM);

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
const bdArcX = [];
const bdArcY = [];
for (let k = 0; k < 40; k += 1) {
  const u = theta + (thetaNext - theta) * k / 39;
  const p = expPoint(ARC_RADIUS, u);
  bdArcX.push(p.re);
  bdArcY.push(p.im);
}
overview_bdelta_arc.data = { x: bdArcX, y: bdArcY };
const bdMid = 0.5 * (theta + thetaNext);
const bdLabel = expPoint(ARC_RADIUS + 0.13, bdMid);

overview_points.data = {
  x: [0, z.re, radialPoint.re, zNext.re, modelM.re, rotatedM.re, 1],
  y: [0, z.im, radialPoint.im, zNext.im, modelM.im, rotatedM.im, 0],
  color: [POINT_COLOR, POINT_COLOR, A_COLOR, M_COLOR, M_COLOR, M_COLOR, POINT_COLOR],
  size: [5, 5, 5, 5, 5, 5, 5],
};
overview_labels.data = {
  x: [
    0,
    z.re,
    zNext.re,
    modelMLabel.re,
    1,
    bdLabel.re,
  ],
  y: [
    0,
    z.im,
    zNext.im,
    modelMLabel.im,
    0,
    bdLabel.im,
  ],
  label: ["0", "Z(t)", "Z(t+δ)", "a+ib", "1", "bδ"],
  x_offset: [-12, 11, -12, 4, 7, 2],
  y_offset: [-6, 2, 12, 4, -10, 4],
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
  size: [6, 6, 6, 5],
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

// velocity vector, moving labels, recomputed spiral, and auto-fit view
const vVec = add(mul(z, a), mul(muli(z), b));
const vEnd = add(z, vVec);
velocity_arrow.x_start = z.re;
velocity_arrow.y_start = z.im;
velocity_arrow.x_end = vEnd.re;
velocity_arrow.y_end = vEnd.im;
velocity_arrow.change.emit();
velocity_label.x = z.re + 0.58 * (vEnd.re - z.re);
velocity_label.y = z.im + 0.58 * (vEnd.im - z.im);
rotated_math_label.x = rotatedMLabel.re;
rotated_math_label.y = rotatedMLabel.im;

const period = 2 * Math.PI / Math.max(Math.abs(b), 1e-6);
const tStart = t - 2.5 * period;
const tEnd = t + delta + SPIRAL_EXTRA_ANGLE / Math.max(Math.abs(b), 1e-6);
const NS = 600;
const curveX = [];
const curveY = [];
for (let k = 0; k < NS; k += 1) {
  const tt = tStart + (tEnd - tStart) * k / (NS - 1);
  const p = expPoint(Math.exp(a * tt), b * tt);
  curveX.push(p.re);
  curveY.push(p.im);
}
curve_source.data = { x: curveX, y: curveY };
const spiralEnd = expPoint(Math.exp(a * tEnd), b * tEnd);

const fitX = [0, z.re, zNext.re, radialPoint.re, rotatedM.re, modelM.re, 1, spiralEnd.re];
const fitY = [0, z.im, zNext.im, radialPoint.im, rotatedM.im, modelM.im, 0, spiralEnd.im];
let maxr = 1.05;
for (let k = 0; k < fitX.length; k += 1) {
  maxr = Math.max(maxr, Math.hypot(fitX[k], fitY[k]));
}
const R = Math.max(1.3, OVERVIEW_FIT_PADDING * maxr);
const desiredCenterY = OVERVIEW_VERTICAL_BIAS * R;
const minCenterY = Math.max(...fitY) - R;
const maxCenterY = Math.min(...fitY) + R;
const centerY = Math.min(Math.max(desiredCenterY, minCenterY), maxCenterY);
if (autofitPanel1) {
  x_range.setv({ start: -R, end: R });
  y_range.setv({ start: centerY - R, end: centerY + R });
}

// auto-fit the zoom panel, biased so the triangle sits right of centre
const f2x = [localOrigin.re, localRadial.re, localNext.re, localIdeal.re, 0, a, a];
const f2y = [localOrigin.im, localRadial.im, localNext.im, localIdeal.im, 0, 0, b];
const f2minx = Math.min(...f2x);
const f2maxx = Math.max(...f2x);
const f2miny = Math.min(...f2y);
const f2maxy = Math.max(...f2y);
const f2cx = 0.5 * (f2minx + f2maxx);
const f2cy = 0.5 * (f2miny + f2maxy);
const f2half = Math.max(0.4, 0.5 * Math.max(f2maxx - f2minx, f2maxy - f2miny) * 1.22);
finite_x_range.setv({ start: f2cx - 1.5 * f2half, end: f2cx + 0.5 * f2half });
finite_y_range.setv({ start: f2cy - f2half, end: f2cy + f2half });
"""
