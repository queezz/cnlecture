"""Tests for the Cotes theorem distance-product visualization."""

import math

import numpy as np
import pytest

from cnlecture.visualizations.cotes_theorem import (
    _bokeh_sources,
    _factor_terms,
    _un_bokeh_sources,
    cotes_distance_product,
    cotes_distances,
    cotes_quadratic_coefficients,
    cotes_vertices,
)


def test_cotes_vertices_are_roots_of_unity():
    vertices = cotes_vertices(4)

    assert vertices == pytest.approx(np.asarray([1, 1j, -1, -1j], dtype=complex))
    assert vertices**4 == pytest.approx(np.ones(4, dtype=complex))


def test_cotes_distance_product_matches_polynomial_for_x_greater_than_one():
    for n in range(2, 13):
        x = 1.37

        assert cotes_distance_product(n, x) == pytest.approx(x**n - 1)


def test_distances_for_square_include_pythagorean_factors():
    distances = cotes_distances(4, 1.6)

    assert distances[0] == pytest.approx(0.6)
    assert distances[1] == pytest.approx(math.sqrt(1.6**2 + 1))
    assert distances[2] == pytest.approx(2.6)
    assert distances[3] == pytest.approx(math.sqrt(1.6**2 + 1))


def test_quadratic_coefficients_give_real_pair_factors():
    coefficients = cotes_quadratic_coefficients(5)

    assert coefficients == pytest.approx(
        [
            -2 * math.cos(2 * math.pi / 5),
            -2 * math.cos(4 * math.pi / 5),
        ]
    )


def test_bokeh_sources_describe_polygon_and_rays():
    sources = _bokeh_sources(6, 1.5)

    assert len(sources["polygon"]["x"]) == 7
    assert len(sources["rays"]["xs"]) == 6
    assert sources["vertex_labels"]["label"] == ["C1", "C2", "C3", "C4", "C5", "C6"]
    assert sources["point"]["x"] == [1.5]


def test_un_sources_mark_current_polynomial_value():
    sources = _un_bokeh_sources(5, 1.5)

    assert sources["point"]["x"] == [1.5]
    assert sources["point"]["y"] == pytest.approx([1.5**5 - 1])
    assert len(sources["curve"]["x"]) == 240


def test_factor_terms_have_total_degree():
    for n in range(2, 13):
        terms = _factor_terms(n)
        degree = 0
        for term in terms:
            degree += 2 if "x^2" in term else 1

        assert degree == n


def test_layout_exposes_n_x_and_un_panel_controls():
    pytest.importorskip("bokeh")
    from bokeh.models import Checkbox, Slider

    from cnlecture.visualizations.cotes_theorem import make_cotes_theorem_bokeh

    layout = make_cotes_theorem_bokeh()
    sliders = {m.title.split()[0]: m.value for m in layout.references() if isinstance(m, Slider)}
    checkboxes = {m.label: m.active for m in layout.references() if isinstance(m, Checkbox)}

    assert sliders == {"n": 5, "x": 1.65}
    assert checkboxes == {"Show U_n(x) panel": True}
