"""
Geometric power-series construction of Euler's formula.

The Taylor expansion of exp(i·φ) is:

    exp(i·φ) = 1 + i·φ + (i·φ)²/2! + (i·φ)³/3! + …

Each term is a complex number.  When laid head-to-tail in the complex plane
(a *vector chain*), the partial sums spiral inward and converge to the
exact point exp(i·φ) on the unit circle.

This module provides:
- Pure math helpers (no plotting dependency)
- A Plotly figure factory
- An HTML export helper for static hosting / MkDocs
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Pure math — no plotting imports
# ---------------------------------------------------------------------------


def euler_series_terms(phi: float, n_terms: int = 30) -> npt.NDArray[np.complexfloating]:
    """Return the individual terms of the Taylor series of exp(i·φ).

    The k-th term is ``(i·φ)^k / k!`` for k = 0, 1, …, n_terms-1.

    Parameters
    ----------
    phi:
        The argument in radians.
    n_terms:
        Number of terms to compute (including the k=0 constant term ``1``).

    Returns
    -------
    numpy array of shape ``(n_terms,)`` with complex dtype.

    Examples
    --------
    >>> terms = euler_series_terms(0.0, n_terms=5)
    >>> terms[0]   # constant term
    (1+0j)
    >>> terms[1]   # first-order term  i·0 = 0
    0j
    """
    return np.array(
        [(1j * phi) ** k / math.factorial(k) for k in range(n_terms)],
        dtype=complex,
    )


def euler_series_points(phi: float, n_terms: int = 30) -> npt.NDArray[np.complexfloating]:
    """Return the partial-sum endpoints of the Taylor series of exp(i·φ).

    The k-th element is ``sum_{j=0}^{k} (i·φ)^j / j!``.

    Parameters
    ----------
    phi:
        The argument in radians.
    n_terms:
        Number of terms; the returned array has length ``n_terms``.

    Returns
    -------
    numpy array of shape ``(n_terms,)`` with complex dtype.
    The last element is the best partial-sum approximation to ``exp(i·φ)``.
    """
    terms = euler_series_terms(phi, n_terms)
    return np.cumsum(terms)


# ---------------------------------------------------------------------------
# Plotly figure factory
# ---------------------------------------------------------------------------


def make_euler_spiral_plotly(
    phi: float = 1.0,
    n_terms: int = 30,
    show_projections: bool = True,
):
    """Build a Plotly figure showing the power-series construction of exp(i·φ).

    The figure contains:

    * The unit circle.
    * The exact target point ``exp(i·φ)`` on the unit circle.
    * The vector chain: each Taylor term laid head-to-tail.
    * The partial-sum endpoint (where the chain ends).
    * Optional dashed projection lines from the endpoint to both axes.

    Parameters
    ----------
    phi:
        Argument of ``exp(i·φ)`` in radians.  Controls where the target
        point sits on the unit circle.
    n_terms:
        How many Taylor terms to draw (including the ``k=0`` constant ``1``).
    show_projections:
        If True, draw dashed lines from the partial-sum endpoint to the
        real and imaginary axes, labelling ``cos φ`` and ``sin φ``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    terms = euler_series_terms(phi, n_terms)
    points = euler_series_points(phi, n_terms)
    exact = np.exp(1j * phi)
    endpoint = points[-1]

    # Unit circle
    t = np.linspace(0, 2 * np.pi, 360)
    circle_x = np.cos(t)
    circle_y = np.sin(t)

    traces: list = []

    # ---- unit circle -------------------------------------------------------
    traces.append(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line=dict(color="steelblue", width=1.5, dash="dot"),
            name="Unit circle",
            hoverinfo="skip",
        )
    )

    # ---- vector chain (arrows via annotations, bodies via scatter) ---------
    # We draw the bodies of the vectors as a connected scatter for performance,
    # then add a single representative annotation arrow tip for the last vector.
    chain_x: list[Optional[float]] = []
    chain_y: list[Optional[float]] = []
    for k, term in enumerate(terms):
        tail = points[k - 1] if k > 0 else 0 + 0j
        head = tail + term
        chain_x += [tail.real, head.real, None]
        chain_y += [tail.imag, head.imag, None]

    traces.append(
        go.Scatter(
            x=chain_x,
            y=chain_y,
            mode="lines",
            line=dict(color="crimson", width=2),
            name="Series terms",
            hoverinfo="skip",
        )
    )

    # Dots at each partial-sum point (excluding the first trivial 1+0j)
    traces.append(
        go.Scatter(
            x=points.real,
            y=points.imag,
            mode="markers",
            marker=dict(size=5, color="crimson", opacity=0.7),
            name="Partial sums",
            hovertemplate="S_%{pointIndex}: %{x:.3f}+%{y:.3f}i<extra></extra>",
        )
    )

    # ---- exact point on unit circle ----------------------------------------
    traces.append(
        go.Scatter(
            x=[exact.real],
            y=[exact.imag],
            mode="markers+text",
            marker=dict(size=14, color="steelblue", symbol="x", line=dict(width=2)),
            text=[f"e<sup>i·{phi:.2f}</sup> = {exact.real:.3f}+{exact.imag:.3f}i"],
            textposition="top right",
            name="exp(i·φ)",
            hoverinfo="text",
        )
    )

    # ---- partial-sum endpoint -----------------------------------------------
    traces.append(
        go.Scatter(
            x=[endpoint.real],
            y=[endpoint.imag],
            mode="markers+text",
            marker=dict(size=12, color="darkorange", symbol="circle"),
            text=[f"S<sub>{n_terms}</sub>"],
            textposition="bottom left",
            name=f"Partial sum S_{n_terms}",
            hovertemplate=f"S_{n_terms}: {endpoint.real:.4f}+{endpoint.imag:.4f}i<extra></extra>",
        )
    )

    # ---- projection lines --------------------------------------------------
    annotations = []
    if show_projections:
        # horizontal to Im axis
        traces.append(
            go.Scatter(
                x=[exact.real, exact.real, 0],
                y=[0, exact.imag, exact.imag],
                mode="lines",
                line=dict(color="gray", width=1, dash="dash"),
                name="Projections",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # cos φ label on real axis
        annotations.append(
            dict(
                x=exact.real / 2,
                y=-0.12,
                text=f"cos φ = {exact.real:.3f}",
                showarrow=False,
                font=dict(size=11, color="gray"),
            )
        )
        # sin φ label on imaginary axis
        annotations.append(
            dict(
                x=-0.22,
                y=exact.imag / 2,
                text=f"sin φ = {exact.imag:.3f}",
                showarrow=False,
                font=dict(size=11, color="gray"),
                textangle=-90,
            )
        )

    # ---- origin dot --------------------------------------------------------
    traces.append(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(size=8, color="black"),
            name="Origin",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # ---- layout ------------------------------------------------------------
    pad = 0.3
    ax_range = [-1 - pad, 1 + pad]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(
                f"Euler's formula: e<sup>iφ</sup> via power series"
                f"<br><sup>φ = {phi:.3f} rad,  {n_terms} terms,  "
                f"error = {abs(endpoint - exact):.2e}</sup>"
            ),
            x=0.5,
        ),
        xaxis=dict(
            title="Re",
            range=ax_range,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="lightgray",
            scaleanchor="y",
        ),
        yaxis=dict(
            title="Im",
            range=ax_range,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="lightgray",
        ),
        legend=dict(x=1.02, y=1, xanchor="left"),
        template="plotly_white",
        width=680,
        height=620,
        annotations=annotations,
        margin=dict(l=60, r=180, t=80, b=60),
    )

    return fig


def make_euler_spiral_interactive(
    phi_default: float = 1.0,
    n_terms_default: int = 30,
):
    """Return a Plotly figure with sliders for φ and n_terms.

    The sliders are implemented as Plotly *frames* + *sliders* so the result
    is fully self-contained HTML — no Python kernel required.

    Parameters
    ----------
    phi_default:
        Starting value of φ.
    n_terms_default:
        Starting number of terms shown.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    phi_values = np.round(np.arange(0, 2 * np.pi + 0.01, 0.1), 3)
    list(range(2, 41))

    # Build one frame per (phi, n_terms) pair for the phi slider,
    # keeping n_terms fixed at n_terms_default.
    frames = []
    for phi in phi_values:
        terms = euler_series_terms(float(phi), n_terms_default)
        points = euler_series_points(float(phi), n_terms_default)
        exact = np.exp(1j * float(phi))
        endpoint = points[-1]
        error = abs(endpoint - exact)

        # chain bodies
        chain_x: list[Optional[float]] = []
        chain_y: list[Optional[float]] = []
        for k, term in enumerate(terms):
            tail = points[k - 1] if k > 0 else 0 + 0j
            head = tail + term
            chain_x += [tail.real, head.real, None]
            chain_y += [tail.imag, head.imag, None]

        t = np.linspace(0, 2 * np.pi, 360)
        frames.append(
            go.Frame(
                name=str(phi),
                data=[
                    # 0: circle (static — redrawn to preserve trace order)
                    go.Scatter(x=np.cos(t), y=np.sin(t)),
                    # 1: chain bodies
                    go.Scatter(x=chain_x, y=chain_y),
                    # 2: partial-sum dots
                    go.Scatter(x=points.real, y=points.imag),
                    # 3: exact point
                    go.Scatter(
                        x=[exact.real],
                        y=[exact.imag],
                        text=[f"e<sup>iφ</sup>={exact.real:.3f}+{exact.imag:.3f}i"],
                    ),
                    # 4: endpoint
                    go.Scatter(
                        x=[endpoint.real],
                        y=[endpoint.imag],
                        text=[f"S error={error:.2e}"],
                    ),
                ],
                layout=go.Layout(
                    title=dict(
                        text=(
                            f"Euler's formula: e<sup>iφ</sup> via power series"
                            f"<br><sup>φ = {phi:.2f} rad,  "
                            f"{n_terms_default} terms,  error = {error:.2e}</sup>"
                        )
                    )
                ),
            )
        )

    # Base figure at phi_default
    base = make_euler_spiral_plotly(phi_default, n_terms_default, show_projections=False)

    base.frames = frames
    base.update_layout(
        sliders=[
            dict(
                active=int(np.argmin(np.abs(phi_values - phi_default))),
                steps=[
                    dict(
                        args=[
                            [str(phi)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                            ),
                        ],
                        label=f"{phi:.1f}",
                        method="animate",
                    )
                    for phi in phi_values
                ],
                currentvalue=dict(prefix="φ = ", suffix=" rad", visible=True),
                pad=dict(t=50),
                x=0,
                len=1,
            )
        ]
    )

    return base


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------


def export_euler_spiral_html(
    path: str | Path = "docs/assets/plots/euler_spiral.html",
    phi: float = 1.0,
    n_terms: int = 30,
    interactive: bool = True,
) -> Path:
    """Write a self-contained HTML file of the Euler spiral figure.

    Parameters
    ----------
    path:
        Destination file.  Parent directories are created if needed.
    phi:
        Default φ value for the figure.
    n_terms:
        Number of Taylor terms to show.
    interactive:
        If True (default), export the slider-enabled interactive figure.
        If False, export a static figure at the given phi.

    Returns
    -------
    Resolved ``Path`` of the written file.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if interactive:
        fig = make_euler_spiral_interactive(phi_default=phi, n_terms_default=n_terms)
    else:
        fig = make_euler_spiral_plotly(phi=phi, n_terms=n_terms)

    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    return out
