"""Tests for the triangle midpoint square construction."""

import pytest

from cnlecture.visualizations.triangle_midpoint import (
    default_triangle_vertices,
    triangle_half_sides,
    triangle_square_centers,
    triangle_square_vertices,
    triangle_third_side_midpoint,
)


def test_half_side_definitions():
    vertices = default_triangle_vertices()
    a, b = triangle_half_sides(vertices)

    assert 2 * a == pytest.approx(vertices[1])
    assert 2 * a + 2 * b == pytest.approx(vertices[2])


def test_square_center_formulas():
    vertices = default_triangle_vertices()
    a, b = triangle_half_sides(vertices)
    p_center, s_center = triangle_square_centers(vertices)

    assert p_center == pytest.approx(a + 1j * a)
    assert s_center == pytest.approx(2 * a + b + 1j * b)


def test_third_side_midpoint_formula():
    vertices = default_triangle_vertices()
    a, b = triangle_half_sides(vertices)
    midpoint = triangle_third_side_midpoint(vertices)

    assert midpoint == pytest.approx(a + b)


def test_midpoint_center_identity():
    vertices = default_triangle_vertices()
    p_center, s_center = triangle_square_centers(vertices)
    midpoint = triangle_third_side_midpoint(vertices)

    assert (midpoint - p_center) + 1j * (s_center - midpoint) == pytest.approx(0 + 0j)


def test_square_paths_are_closed():
    squares = triangle_square_vertices(default_triangle_vertices())

    assert squares.shape == (2, 5)
    assert squares[:, 0] == pytest.approx(squares[:, -1])
