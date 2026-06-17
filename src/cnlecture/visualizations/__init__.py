"""Interactive visualisations for cnlecture."""

from cnlecture.visualizations.cotes_theorem import (
    cotes_distance_product,
    cotes_distances,
    cotes_quadratic_coefficients,
    cotes_vertices,
    export_cotes_theorem_html,
    make_cotes_theorem_bokeh,
)
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
from cnlecture.visualizations.spiral_velocity import (
    SpiralStepGeometry,
    angle_between,
    export_spiral_velocity_html,
    make_spiral_velocity_bokeh,
    spiral_curve,
    spiral_point,
    spiral_step_geometry,
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
    "cotes_vertices",
    "cotes_distances",
    "cotes_distance_product",
    "cotes_quadratic_coefficients",
    "make_cotes_theorem_bokeh",
    "export_cotes_theorem_html",
    "SpiralStepGeometry",
    "spiral_point",
    "spiral_curve",
    "angle_between",
    "spiral_step_geometry",
    "make_spiral_velocity_bokeh",
    "export_spiral_velocity_html",
    "default_triangle_vertices",
    "triangle_half_sides",
    "triangle_square_vertices",
    "triangle_square_centers",
    "triangle_third_side_midpoint",
    "make_triangle_midpoint_bokeh",
    "export_triangle_midpoint_html",
]
