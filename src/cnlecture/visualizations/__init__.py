"""Interactive visualisations for cnlecture."""

from cnlecture.visualizations.euler import (
    euler_series_points,
    euler_series_terms,
    export_euler_spiral_html,
    make_euler_spiral_plotly,
)
from cnlecture.visualizations.quadrilateral_squares import (
    default_quadrilateral_vertices,
    export_quadrilateral_squares_html,
    make_quadrilateral_squares_bokeh,
    opposite_center_identity,
    opposite_center_segments,
    quadrilateral_half_sides,
    square_centers,
    square_vertices,
)
from cnlecture.visualizations.triangle_midpoint import (
    default_triangle_vertices,
    export_triangle_midpoint_html,
    make_triangle_midpoint_bokeh,
    triangle_half_sides,
    triangle_square_centers,
    triangle_square_vertices,
    triangle_third_side_midpoint,
)

__all__ = [
    "euler_series_terms",
    "euler_series_points",
    "make_euler_spiral_plotly",
    "export_euler_spiral_html",
    "default_quadrilateral_vertices",
    "quadrilateral_half_sides",
    "square_vertices",
    "square_centers",
    "opposite_center_segments",
    "opposite_center_identity",
    "make_quadrilateral_squares_bokeh",
    "export_quadrilateral_squares_html",
    "default_triangle_vertices",
    "triangle_half_sides",
    "triangle_square_vertices",
    "triangle_square_centers",
    "triangle_third_side_midpoint",
    "make_triangle_midpoint_bokeh",
    "export_triangle_midpoint_html",
]
