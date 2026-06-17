"""Cotes' theorem for roots of unity and distance products.

For the regular ``n``-gon with vertices

    C_k = exp(2 pi i (k - 1) / n),

and a point ``P = x`` on the ray through ``C_1 = 1``, Cotes' theorem reads

    PC_1 PC_2 ... PC_n = x^n - 1

when ``x > 1``.  Equivalently, the product of the complex factors
``(x - C_k)`` is ``x^n - 1``; conjugate vertices combine into real quadratic
factors.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt

ComplexArray = npt.NDArray[np.complexfloating]


def _validate_n(n: int) -> int:
    n_int = int(n)
    if n_int < 2:
        raise ValueError("n must be at least 2")
    return n_int


def cotes_vertices(n: int) -> ComplexArray:
    """Return the vertices ``C_1, ..., C_n`` of the regular unit ``n``-gon."""
    n = _validate_n(n)
    angles = 2 * math.pi * np.arange(n) / n
    return np.exp(1j * angles)


def cotes_distances(n: int, x: float) -> npt.NDArray[np.floating]:
    """Return the distances from ``P=x`` to the Cotes ``n``-gon vertices."""
    vertices = cotes_vertices(n)
    return np.abs(float(x) - vertices)


def cotes_distance_product(n: int, x: float) -> float:
    """Return ``PC_1 PC_2 ... PC_n`` for ``P=x``."""
    return float(np.prod(cotes_distances(n, x)))


def cotes_quadratic_coefficients(n: int) -> list[float]:
    """Return the real-pair coefficients in ``x^2 + c x + 1``.

    The factor from the conjugate pair with angles ``+- 2 pi k/n`` is

        x^2 - 2 cos(2 pi k/n) x + 1.

    This function returns the coefficient ``c = -2 cos(2 pi k/n)`` for the
    conjugate pairs.  The separate linear factors ``x - 1`` and, for even
    ``n``, ``x + 1`` are not included.
    """
    n = _validate_n(n)
    return [-2 * math.cos(2 * math.pi * k / n) for k in range(1, (n + 1) // 2)]


PRODUCT_VECTOR_MAX = 1.75


def make_cotes_theorem_bokeh(n: int = 5, x: float = 1.65, y: float = 0.45):
    """Build a Bokeh visualization of Cotes' distance-product theorem."""
    try:
        from bokeh.layouts import column, row
        from bokeh.models import (
            Checkbox,
            ColumnDataSource,
            CustomJS,
            Div,
            LabelSet,
            Range1d,
            Slider,
        )
        from bokeh.plotting import figure
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Bokeh is required for this interactive visualization. "
            'Install it with `pip install -e ".[dev]"`.'
        ) from exc

    n = _validate_n(n)
    x = float(x)
    sources = _bokeh_sources(n, x)
    circle = np.exp(1j * np.linspace(0, 2 * math.pi, 240))

    circle_source = ColumnDataSource(
        data=dict(x=[z.real for z in circle], y=[z.imag for z in circle])
    )
    axis_source = ColumnDataSource(data=sources["axis"])
    polygon_source = ColumnDataSource(data=sources["polygon"])
    ray_source = ColumnDataSource(data=sources["rays"])
    vertex_source = ColumnDataSource(data=sources["vertices"])
    vertex_label_source = ColumnDataSource(data=sources["vertex_labels"])
    point_source = ColumnDataSource(data=sources["point"])
    center_source = ColumnDataSource(data=dict(x=[0.0], y=[0.0], label=["O"]))
    product_sources = _product_bokeh_sources(n, x, y)
    product_factor_source = ColumnDataSource(data=product_sources["factors"])
    product_root_source = ColumnDataSource(data=product_sources["roots"])
    product_root_label_source = ColumnDataSource(data=product_sources["root_labels"])
    product_z_source = ColumnDataSource(data=product_sources["z"])
    product_arrow_source = ColumnDataSource(data=product_sources["product_arrow"])
    product_point_source = ColumnDataSource(data=product_sources["product_point"])

    summary = Div(
        text=_format_summary(n, x),
        width=390,
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
    factorization = Div(
        text=_format_factorization(n),
        width=390,
        styles={
            "font-family": "JetBrains Mono, Menlo, Consolas, monospace",
            "font-size": "13px",
            "line-height": "1.55",
            "border": "1px solid #d8dee9",
            "border-radius": "6px",
            "padding": "12px",
            "background": "#fffdf7",
        },
    )
    product_summary = Div(
        text=_format_product_summary(n, x, y),
        width=390,
        styles={
            "font-family": "JetBrains Mono, Menlo, Consolas, monospace",
            "font-size": "13px",
            "line-height": "1.55",
            "border": "1px solid #d8dee9",
            "border-radius": "6px",
            "padding": "12px",
            "background": "#f8fbff",
        },
    )

    plot = figure(
        width=560,
        height=560,
        x_range=Range1d(-1.45, 2.65, bounds=(-2.0, 3.2)),
        y_range=Range1d(-2.05, 2.05, bounds=(-2.4, 2.4)),
        x_axis_label="Re",
        y_axis_label="Im",
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
        toolbar_location="above",
        title="Cotes' theorem: roots of unity as a distance product",
    )
    plot.grid.grid_line_alpha = 0.28
    plot.xaxis.axis_label_text_font_style = "normal"
    plot.yaxis.axis_label_text_font_style = "normal"

    plot.line(
        x="x",
        y="y",
        source=circle_source,
        line_color="#6b7280",
        line_width=1.5,
        line_dash="dashed",
        alpha=0.9,
    )
    plot.line(
        x="x",
        y="y",
        source=axis_source,
        line_color="#374151",
        line_width=1.5,
    )
    plot.multi_line(
        xs="xs",
        ys="ys",
        source=ray_source,
        line_color="color",
        line_width="width",
        line_alpha="alpha",
    )
    plot.line(x="x", y="y", source=polygon_source, line_color="#111827", line_width=3)
    plot.scatter(
        x="x",
        y="y",
        source=vertex_source,
        size=12,
        fill_color="#ffffff",
        line_color="#111827",
        line_width=1.8,
        alpha=1.0,
    )
    plot.scatter(x="x", y="y", source=center_source, size=9, color="#111827")
    plot.scatter(
        x="x",
        y="y",
        source=point_source,
        size=12,
        fill_color="#ffffff",
        line_color="#b45309",
        line_width=2,
    )
    plot.add_layout(
        LabelSet(
            x="x",
            y="y",
            text="label",
            source=vertex_label_source,
            x_offset=3,
            y_offset=3,
            text_font_size="13px",
            text_color="#111827",
        )
    )
    plot.add_layout(
        LabelSet(
            x="x",
            y="y",
            text="label",
            source=point_source,
            x_offset=8,
            y_offset=-3,
            text_font_size="16px",
            text_color="#92400e",
        )
    )
    plot.add_layout(
        LabelSet(
            x="x",
            y="y",
            text="label",
            source=center_source,
            x_offset=-18,
            y_offset=-18,
            text_font_size="13px",
            text_color="#111827",
        )
    )

    product_plot = figure(
        width=560,
        height=560,
        x_range=Range1d(-2.4, 2.8, bounds=(-4.0, 4.0)),
        y_range=Range1d(-2.6, 2.6, bounds=(-4.0, 4.0)),
        x_axis_label="Re",
        y_axis_label="Im",
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
        toolbar_location="above",
        title="Complex product:  U_n(z) = (z-C_1)...(z-C_n)",
    )
    product_plot.grid.grid_line_alpha = 0.28
    product_plot.xaxis.axis_label_text_font_style = "normal"
    product_plot.yaxis.axis_label_text_font_style = "normal"
    product_plot.line(
        x="x",
        y="y",
        source=circle_source,
        line_color="#6b7280",
        line_width=1.5,
        line_dash="dashed",
        alpha=0.8,
    )
    product_plot.line(
        x="x",
        y="y",
        source=axis_source,
        line_color="#374151",
        line_width=1.3,
    )
    product_plot.multi_line(
        xs="xs",
        ys="ys",
        source=product_factor_source,
        line_color="color",
        line_width="width",
        line_alpha="alpha",
    )
    product_plot.line(
        x="x",
        y="y",
        source=product_arrow_source,
        line_color="#cc79a7",
        line_width=3,
    )
    product_plot.scatter(
        x="x",
        y="y",
        source=product_root_source,
        size=11,
        fill_color="#ffffff",
        line_color="#111827",
        line_width=1.6,
    )
    product_plot.scatter(
        x="x",
        y="y",
        source=center_source,
        size=9,
        color="#111827",
    )
    product_plot.scatter(
        x="x",
        y="y",
        source=product_z_source,
        size=13,
        fill_color="#ffffff",
        line_color="#b45309",
        line_width=2,
    )
    product_plot.scatter(
        x="x",
        y="y",
        source=product_point_source,
        size=11,
        fill_color="#ffffff",
        line_color="#cc79a7",
        line_width=2,
    )
    product_plot.add_layout(
        LabelSet(
            x="x",
            y="y",
            text="label",
            source=product_root_label_source,
            x_offset=3,
            y_offset=3,
            text_font_size="12px",
            text_color="#111827",
        )
    )
    product_plot.add_layout(
        LabelSet(
            x="x",
            y="y",
            text="label",
            source=product_z_source,
            x_offset=8,
            y_offset=6,
            text_font_size="15px",
            text_color="#92400e",
        )
    )
    product_plot.add_layout(
        LabelSet(
            x="x",
            y="y",
            text="label",
            source=product_point_source,
            x_offset=8,
            y_offset=6,
            text_font_size="12px",
            text_color="#9f2f75",
        )
    )

    n_slider = Slider(title="n  (number of vertices)", start=2, end=12, step=1, value=n, width=360)
    x_slider = Slider(title="x  (distance OP)", start=1.05, end=2.4, step=0.01, value=x, width=360)
    y_slider = Slider(title="Im(z)  (product panel)", start=-1.2, end=1.2, step=0.02, value=y, width=360)
    show_product_checkbox = Checkbox(label="Show complex product panel", active=True, width=360)
    callback = CustomJS(
        args=dict(
            n_slider=n_slider,
            x_slider=x_slider,
            y_slider=y_slider,
            axis_source=axis_source,
            polygon_source=polygon_source,
            ray_source=ray_source,
            vertex_source=vertex_source,
            vertex_label_source=vertex_label_source,
            point_source=point_source,
            product_factor_source=product_factor_source,
            product_root_source=product_root_source,
            product_root_label_source=product_root_label_source,
            product_z_source=product_z_source,
            product_arrow_source=product_arrow_source,
            product_point_source=product_point_source,
            product_plot=product_plot,
            summary=summary,
            factorization=factorization,
            product_summary=product_summary,
            x_range=plot.x_range,
            show_product_checkbox=show_product_checkbox,
        ),
        code=_BOKEH_UPDATE_JS,
    )
    n_slider.js_on_change("value", callback)
    x_slider.js_on_change("value", callback)
    y_slider.js_on_change("value", callback)
    show_product_checkbox.js_on_change("active", callback)

    controls = column(n_slider, x_slider, y_slider, show_product_checkbox, width=380)
    return column(
        row(plot, product_plot),
        row(controls, summary, product_summary, factorization),
        sizing_mode="stretch_width",
    )


def export_cotes_theorem_html(
    path: str | Path = "docs/assets/plots/cotes_theorem.html",
    n: int = 5,
    x: float = 1.65,
    y: float = 0.45,
) -> Path:
    """Write a standalone Bokeh HTML file for the Cotes theorem example."""
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
        make_cotes_theorem_bokeh(n=n, x=x, y=y),
        filename=str(out),
        resources=INLINE,
        title="Cotes' theorem distance product",
    )
    return out


def _bokeh_sources(n: int, x: float) -> dict[str, dict[str, list]]:
    n = _validate_n(n)
    x = float(x)
    vertices = cotes_vertices(n)
    closed = np.append(vertices, vertices[0])
    p = complex(x, 0.0)

    return {
        "axis": {
            "x": [-1.2, max(2.55, x + 0.18)],
            "y": [0.0, 0.0],
        },
        "polygon": {
            "x": [z.real for z in closed],
            "y": [z.imag for z in closed],
        },
        "rays": {
            "xs": [[p.real, z.real] for z in vertices],
            "ys": [[p.imag, z.imag] for z in vertices],
            "color": ["#0072b2"] + ["#9ca3af"] * (n - 1),
            "width": [3.0] + [1.4] * (n - 1),
            "alpha": [0.95] + [0.68] * (n - 1),
        },
        "vertices": {
            "x": [z.real for z in vertices],
            "y": [z.imag for z in vertices],
        },
        "vertex_labels": {
            "x": [1.1 * z.real for z in vertices],
            "y": [1.1 * z.imag for z in vertices],
            "label": [f"C{i}" for i in range(1, n + 1)],
        },
        "point": {
            "x": [x],
            "y": [0.0],
            "label": ["P"],
        },
    }


def _product_bokeh_sources(n: int, x: float, y: float) -> dict[str, dict[str, list]]:
    n = _validate_n(n)
    x = float(x)
    y = float(y)
    vertices = cotes_vertices(n)
    z = complex(x, y)
    product = z**n - 1
    scale = _product_display_scale(product)
    shown_product = scale * product
    return {
        "factors": {
            "xs": [[root.real, z.real] for root in vertices],
            "ys": [[root.imag, z.imag] for root in vertices],
            "color": ["#0072b2"] + ["#9ca3af"] * (n - 1),
            "width": [3.0] + [1.4] * (n - 1),
            "alpha": [0.95] + [0.68] * (n - 1),
        },
        "roots": {
            "x": [root.real for root in vertices],
            "y": [root.imag for root in vertices],
        },
        "root_labels": {
            "x": [1.1 * root.real for root in vertices],
            "y": [1.1 * root.imag for root in vertices],
            "label": [f"C{i}" for i in range(1, n + 1)],
        },
        "z": {
            "x": [z.real],
            "y": [z.imag],
            "label": ["z"],
        },
        "product_arrow": {
            "x": [0.0, shown_product.real],
            "y": [0.0, shown_product.imag],
        },
        "product_point": {
            "x": [shown_product.real],
            "y": [shown_product.imag],
            "label": [_product_point_label(scale)],
        },
    }


def _product_display_scale(product: complex) -> float:
    magnitude = abs(product)
    if magnitude <= PRODUCT_VECTOR_MAX or magnitude == 0:
        return 1.0
    return PRODUCT_VECTOR_MAX / magnitude


def _product_point_label(scale: float) -> str:
    if math.isclose(scale, 1.0):
        return "U_n(z)"
    return f"{scale:.2g} U_n(z)"


def _format_number(value: float) -> str:
    if abs(value) >= 10000:
        return f"{value:.6e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_signed(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign} {_format_number(abs(value))}"


def _format_quadratic_factor(coefficient: float) -> str:
    return f"(x^2 {_format_signed(coefficient)} x + 1)"


def _factor_terms(n: int) -> list[str]:
    n = _validate_n(n)
    terms = ["(x - 1)"]
    if n % 2 == 0:
        terms.append("(x + 1)")
    terms.extend(_format_quadratic_factor(c) for c in cotes_quadratic_coefficients(n))
    return terms


def _format_summary(n: int, x: float) -> str:
    product = cotes_distance_product(n, x)
    polynomial = x**n - 1
    difference = product - polynomial
    return f"""
<b>Distance product</b><br>
n = {n}, x = {_format_number(x)}<br>
PC<sub>1</sub> PC<sub>2</sub> ... PC<sub>{n}</sub> = {_format_number(product)}<br>
x<sup>{n}</sup> - 1 = {_format_number(polynomial)}<br>
difference = {_format_number(difference)}
"""


def _format_factorization(n: int) -> str:
    terms = " ".join(_factor_terms(n))
    return f"""
<b>Real factor grouping</b><br>
<code>x^{n} - 1 = {terms}</code><br>
Each conjugate pair gives<br>
<code>(x - C)(x - C*) = x^2 - 2 cos(theta) x + 1</code>.
"""


def _format_complex(z: complex) -> str:
    sign = "+" if z.imag >= 0 else "-"
    return f"{_format_number(z.real)} {sign} {_format_number(abs(z.imag))}i"


def _format_angle_radians(angle: float) -> str:
    return f"{angle:.3f}"


def _format_product_summary(n: int, x: float, y: float) -> str:
    roots = cotes_vertices(n)
    z = complex(x, y)
    factors = z - roots
    product = z**n - 1
    radius_product = float(np.prod(np.abs(factors)))
    angle_sum = float(np.sum(np.angle(factors)))
    scale = _product_display_scale(product)
    scale_note = "unscaled" if math.isclose(scale, 1.0) else f"shown at {scale:.3g}x"
    return f"""
<b>Complex product</b><br>
z = {_format_complex(z)}<br>
&prod;(z - C<sub>k</sub>) = {_format_complex(product)}<br>
z<sup>{n}</sup> - 1 = {_format_complex(product)}<br>
R<sub>1</sub>...R<sub>{n}</sub> = {_format_number(radius_product)}<br>
arg sum = {_format_angle_radians(angle_sum)} rad<br>
product arrow: {scale_note}
"""


_BOKEH_UPDATE_JS = r"""
const n = Math.round(n_slider.value);
const x = x_slider.value;
const y = y_slider.value;
const twoPi = 2 * Math.PI;
const PRODUCT_VECTOR_MAX = 1.75;

function fmt(value) {
  if (Math.abs(value) >= 10000) {
    return value.toExponential(6);
  }
  return value.toFixed(6).replace(/\.?0+$/, "");
}

function fmtSigned(value) {
  const sign = value >= 0 ? "+" : "-";
  return `${sign} ${fmt(Math.abs(value))}`;
}
function c(re, im) {
  return {re: re, im: im};
}
function sub(u, v) {
  return c(u.re - v.re, u.im - v.im);
}
function mul(u, v) {
  return c(u.re * v.re - u.im * v.im, u.re * v.im + u.im * v.re);
}
function powInt(u, power) {
  let result = c(1, 0);
  for (let i = 0; i < power; i += 1) {
    result = mul(result, u);
  }
  return result;
}
function absC(u) {
  return Math.hypot(u.re, u.im);
}
function argC(u) {
  return Math.atan2(u.im, u.re);
}
function fmtComplex(u) {
  const sign = u.im >= 0 ? " + " : " - ";
  return `${fmt(u.re)}${sign}${fmt(Math.abs(u.im))}i`;
}
function productScale(u) {
  const magnitude = absC(u);
  if (magnitude <= PRODUCT_VECTOR_MAX || magnitude === 0) {
    return 1;
  }
  return PRODUCT_VECTOR_MAX / magnitude;
}
function productLabel(scale) {
  if (Math.abs(scale - 1) < 1e-12) {
    return "U_n(z)";
  }
  return `${scale.toPrecision(2)} U_n(z)`;
}

const vertices = [];
for (let k = 0; k < n; k += 1) {
  const theta = twoPi * k / n;
  vertices.push({re: Math.cos(theta), im: Math.sin(theta)});
}

const closed = vertices.concat([vertices[0]]);
axis_source.data = {
  x: [-1.2, Math.max(2.55, x + 0.18)],
  y: [0, 0],
};
polygon_source.data = {
  x: closed.map((z) => z.re),
  y: closed.map((z) => z.im),
};
ray_source.data = {
  xs: vertices.map((z) => [x, z.re]),
  ys: vertices.map((z) => [0, z.im]),
  color: ["#0072b2"].concat(Array(n - 1).fill("#9ca3af")),
  width: [3.0].concat(Array(n - 1).fill(1.4)),
  alpha: [0.95].concat(Array(n - 1).fill(0.68)),
};
vertex_source.data = {
  x: vertices.map((z) => z.re),
  y: vertices.map((z) => z.im),
};
vertex_label_source.data = {
  x: vertices.map((z) => 1.1 * z.re),
  y: vertices.map((z) => 1.1 * z.im),
  label: vertices.map((_, i) => `C${i + 1}`),
};
point_source.data = {
  x: [x],
  y: [0],
  label: ["P"],
};

let product = 1;
for (const z of vertices) {
  product *= Math.hypot(x - z.re, z.im);
}
const polynomial = Math.pow(x, n) - 1;
summary.text = `
<b>Distance product</b><br>
n = ${n}, x = ${fmt(x)}<br>
PC<sub>1</sub> PC<sub>2</sub> ... PC<sub>${n}</sub> = ${fmt(product)}<br>
x<sup>${n}</sup> - 1 = ${fmt(polynomial)}<br>
difference = ${fmt(product - polynomial)}
`;

const terms = ["(x - 1)"];
if (n % 2 === 0) {
  terms.push("(x + 1)");
}
for (let k = 1; k < (n + 1) / 2; k += 1) {
  const coefficient = -2 * Math.cos(twoPi * k / n);
  terms.push(`(x^2 ${fmtSigned(coefficient)} x + 1)`);
}
factorization.text = `
<b>Real factor grouping</b><br>
<code>x^${n} - 1 = ${terms.join(" ")}</code><br>
Each conjugate pair gives<br>
<code>(x - C)(x - C*) = x^2 - 2 cos(theta) x + 1</code>.
`;

x_range.end = Math.max(2.65, x + 0.25);

const z = c(x, y);
const factorXs = [];
const factorYs = [];
const factorAngles = [];
let radiusProduct = 1;
let complexProduct = c(1, 0);
for (const root of vertices) {
  const factor = sub(z, root);
  factorXs.push([root.re, z.re]);
  factorYs.push([root.im, z.im]);
  factorAngles.push(argC(factor));
  radiusProduct *= absC(factor);
  complexProduct = mul(complexProduct, factor);
}
const znMinusOne = sub(powInt(z, n), c(1, 0));
const productVectorScale = productScale(znMinusOne);
const shownProduct = c(productVectorScale * znMinusOne.re, productVectorScale * znMinusOne.im);
const angleSum = factorAngles.reduce((acc, value) => acc + value, 0);
const scaleNote = Math.abs(productVectorScale - 1) < 1e-12
  ? "unscaled"
  : `shown at ${productVectorScale.toPrecision(3)}x`;

product_factor_source.data = {
  xs: factorXs,
  ys: factorYs,
  color: ["#0072b2"].concat(Array(n - 1).fill("#9ca3af")),
  width: [3.0].concat(Array(n - 1).fill(1.4)),
  alpha: [0.95].concat(Array(n - 1).fill(0.68)),
};
product_root_source.data = {
  x: vertices.map((root) => root.re),
  y: vertices.map((root) => root.im),
};
product_root_label_source.data = {
  x: vertices.map((root) => 1.1 * root.re),
  y: vertices.map((root) => 1.1 * root.im),
  label: vertices.map((_, i) => `C${i + 1}`),
};
product_z_source.data = {
  x: [z.re],
  y: [z.im],
  label: ["z"],
};
product_arrow_source.data = {
  x: [0, shownProduct.re],
  y: [0, shownProduct.im],
};
product_point_source.data = {
  x: [shownProduct.re],
  y: [shownProduct.im],
  label: [productLabel(productVectorScale)],
};
product_summary.text = `
<b>Complex product</b><br>
z = ${fmtComplex(z)}<br>
&prod;(z - C<sub>k</sub>) = ${fmtComplex(complexProduct)}<br>
z<sup>${n}</sup> - 1 = ${fmtComplex(znMinusOne)}<br>
R<sub>1</sub>...R<sub>${n}</sub> = ${fmt(radiusProduct)}<br>
arg sum = ${angleSum.toFixed(3)} rad<br>
product arrow: ${scaleNote}
`;
product_plot.visible = show_product_checkbox.active;
"""
