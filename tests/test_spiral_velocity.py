"""Tests for the spiral velocity geometry helpers."""

import math

import pytest

from cnlecture.visualizations.spiral_velocity import (
    COMPONENT_COLORS,
    M_COLOR,
    _bokeh_sources,
    angle_between,
    spiral_curve,
    spiral_point,
    spiral_step_geometry,
)


def test_spiral_point_on_unit_circle_when_a_zero():
    point = spiral_point(a=0.0, b=1.0, t=math.pi / 2)

    assert point.real == pytest.approx(0.0)
    assert point.imag == pytest.approx(1.0)


def test_spiral_curve_shape():
    curve = spiral_curve(n_points=17)

    assert curve.shape == (17,)


def test_spiral_curve_requires_two_points():
    with pytest.raises(ValueError, match="at least 2"):
        spiral_curve(n_points=1)


def test_angle_between_complex_vectors():
    assert angle_between(1 + 0j, 1j) == pytest.approx(90.0)
    assert angle_between(1 + 0j, 1 + 0j) == pytest.approx(0.0)


def test_exact_step_decomposes_movement():
    geometry = spiral_step_geometry(delta=0.35)

    assert geometry.a_exact + geometry.b_exact == pytest.approx(geometry.m)


def test_linear_step_decomposes_model_movement():
    geometry = spiral_step_geometry(delta=0.35)

    assert geometry.a_linear + geometry.b_linear == pytest.approx(geometry.m_linear)


def test_perpendicularity_improves_as_delta_shrinks():
    large = spiral_step_geometry(delta=0.8)
    small = spiral_step_geometry(delta=0.02)

    assert small.perpendicular_error_degrees < large.perpendicular_error_degrees
    assert small.perpendicular_error_degrees < 1.0


def test_linear_model_improves_as_delta_shrinks():
    large = spiral_step_geometry(delta=0.8)
    small = spiral_step_geometry(delta=0.02)

    assert abs(small.m - small.m_linear) < abs(large.m - large.m_linear)


def test_triangle_component_colors_are_consistent():
    sources = _bokeh_sources(a=0.18, b=1.0, t=1.25, delta=0.22)

    assert sources["overview_step"]["color"] == COMPONENT_COLORS
    assert sources["overview_origin_model"]["color"] == COMPONENT_COLORS
    assert sources["overview_rotated_model"]["color"] == COMPONENT_COLORS
    assert sources["overview_b_chord"]["color"] == [COMPONENT_COLORS[1]]
    assert sources["finite_exact"]["color"] == COMPONENT_COLORS
    assert sources["finite_ideal"]["color"] == COMPONENT_COLORS
    assert sources["overview_points"]["color"][3] == M_COLOR
    assert sources["finite_points"]["color"][-1] == M_COLOR
    assert "ideal_lines" not in sources


def test_triangle_shade_sources_have_three_vertices():
    sources = _bokeh_sources(a=0.18, b=1.0, t=1.25, delta=0.22)

    assert len(sources["overview_local_shade"]["x"]) == 3
    assert len(sources["overview_local_shade"]["y"]) == 3
    assert len(sources["overview_origin_shade"]["x"]) == 3
    assert len(sources["overview_origin_shade"]["y"]) == 3
    assert len(sources["overview_rotated_shade"]["x"]) == 3
    assert len(sources["overview_rotated_shade"]["y"]) == 3
    assert len(sources["finite_shade"]["x"]) == 3
    assert len(sources["finite_shade"]["y"]) == 3


def test_finite_zoom_keeps_sequential_b_segment():
    sources = _bokeh_sources(a=0.18, b=1.0, t=1.25, delta=0.22)
    exact = sources["finite_exact"]
    ideal = sources["finite_ideal"]

    assert exact["xs"][1][0] == pytest.approx(exact["xs"][0][1])
    assert exact["ys"][1][0] == pytest.approx(exact["ys"][0][1])
    assert ideal["xs"][1][0] == pytest.approx(ideal["xs"][0][1])
    assert ideal["ys"][1][0] == pytest.approx(ideal["ys"][0][1])


def test_overview_step_constructs_a_from_z_and_b_on_next_radius_circle():
    a = 0.18
    b = 1.0
    t = 1.25
    delta = 0.22
    sources = _bokeh_sources(a=a, b=b, t=t, delta=delta)
    step = sources["overview_step"]
    guides = sources["overview_guides"]
    b_chord = sources["overview_b_chord"]
    geometry = spiral_step_geometry(a=a, b=b, t=t, delta=delta)
    expected_turn_point = abs(geometry.z) * complex(
        math.cos(b * (t + delta)),
        math.sin(b * (t + delta)),
    )

    assert step["xs"][0][0] == pytest.approx(geometry.z.real)
    assert step["ys"][0][0] == pytest.approx(geometry.z.imag)
    assert step["xs"][0][1] == pytest.approx(geometry.radial_point.real)
    assert step["ys"][0][1] == pytest.approx(geometry.radial_point.imag)
    assert step["xs"][1][0] == pytest.approx(geometry.radial_point.real)
    assert step["ys"][1][0] == pytest.approx(geometry.radial_point.imag)
    assert step["xs"][1][1] == pytest.approx(geometry.z_next.real)
    assert step["ys"][1][1] == pytest.approx(geometry.z_next.imag)
    assert step["xs"][2][0] == pytest.approx(geometry.z.real)
    assert step["ys"][2][0] == pytest.approx(geometry.z.imag)
    assert step["xs"][2][1] == pytest.approx(geometry.z_next.real)
    assert step["ys"][2][1] == pytest.approx(geometry.z_next.imag)
    assert math.hypot(guides["xs"][2][0], guides["ys"][2][0]) == pytest.approx(abs(geometry.z))
    assert math.hypot(guides["xs"][3][0], guides["ys"][3][0]) == pytest.approx(abs(geometry.z_next))
    assert b_chord["xs"][0][0] == pytest.approx(geometry.z.real)
    assert b_chord["ys"][0][0] == pytest.approx(geometry.z.imag)
    assert b_chord["xs"][0][1] == pytest.approx(expected_turn_point.real)
    assert b_chord["ys"][0][1] == pytest.approx(expected_turn_point.imag)


def test_overview_labels_omit_component_names():
    sources = _bokeh_sources(a=0.18, b=1.0, t=1.25, delta=0.22)

    assert "A" not in sources["overview_labels"]["label"]
    assert "B" not in sources["overview_labels"]["label"]
    assert "M" not in sources["overview_labels"]["label"]


def test_origin_model_uses_actual_a_plus_ib_endpoint():
    a = 0.32
    b = 0.71
    sources = _bokeh_sources(a=a, b=b, t=1.25, delta=0.22)
    origin_model = sources["overview_origin_model"]

    assert origin_model["xs"][2][1] == pytest.approx(a)
    assert origin_model["ys"][2][1] == pytest.approx(b)
    assert math.hypot(origin_model["xs"][2][1], origin_model["ys"][2][1]) != pytest.approx(1.0)
