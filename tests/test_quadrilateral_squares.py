"""Tests for the quadrilateral square-center geometry."""

import numpy as np
import pytest

from cnlecture.visualizations.quadrilateral_squares import (
    default_quadrilateral_vertices,
    opposite_center_identity,
    opposite_center_segments,
    quadrilateral_half_sides,
    square_centers,
    square_vertices,
)


def test_half_sides_sum_to_zero():
    halves = quadrilateral_half_sides(default_quadrilateral_vertices())

    assert np.sum(halves) == pytest.approx(0 + 0j)


def test_square_center_formulas():
    vertices = default_quadrilateral_vertices()
    a, b, c, d = quadrilateral_half_sides(vertices)
    centers = square_centers(vertices)

    expected = np.asarray(
        [
            a + 1j * a,
            2 * a + b + 1j * b,
            2 * a + 2 * b + c + 1j * c,
            -d + 1j * d,
        ],
        dtype=complex,
    )

    assert centers == pytest.approx(expected)


def test_square_paths_are_closed():
    squares = square_vertices(default_quadrilateral_vertices())

    assert squares.shape == (4, 5)
    assert squares[:, 0] == pytest.approx(squares[:, -1])


def test_opposite_center_identity():
    identity = opposite_center_identity(default_quadrilateral_vertices())

    assert identity == pytest.approx(0 + 0j)


def test_opposite_center_segments_have_equal_length_and_are_perpendicular():
    a_segment, b_segment = opposite_center_segments(default_quadrilateral_vertices())

    assert abs(a_segment) == pytest.approx(abs(b_segment))
    assert (a_segment / b_segment).real == pytest.approx(0.0)
