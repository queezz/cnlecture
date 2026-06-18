"""Tests for the triangle distance locator visualization."""

import numpy as np
import pytest

from cnlecture.visualizations.triangle_distance_locator import (
    apply_direct_motion,
    circle_intersections,
    default_locator_point,
    default_locator_triangle,
    locator_geometry,
    locator_radii,
    oriented_triangle_area,
)


def test_default_triangle_is_non_collinear():
    assert abs(oriented_triangle_area(default_locator_triangle())) > 0.1


def test_direct_motion_preserves_pairwise_distances():
    points = np.asarray([0 + 0j, 1.2 - 0.2j, 0.1 + 1.4j, default_locator_point()])
    moved = apply_direct_motion(points, 37.0, 2.0 - 0.4j)

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            assert abs(points[i] - points[j]) == pytest.approx(abs(moved[i] - moved[j]))


def test_locator_radii_are_distances_to_source_vertices():
    triangle = default_locator_triangle()
    point = default_locator_point()

    assert locator_radii(triangle, point) == pytest.approx(np.abs(point - triangle))


def test_circle_intersections_for_two_unit_circles():
    intersections = circle_intersections(0 + 0j, 1.0, 1 + 0j, 1.0)

    assert len(intersections) == 2
    assert sorted(z.imag for z in intersections) == pytest.approx(
        sorted([-np.sqrt(3) / 2, np.sqrt(3) / 2])
    )
    assert [z.real for z in intersections] == pytest.approx([0.5, 0.5])


def test_locator_geometry_selects_transformed_point_with_third_circle():
    geometry = locator_geometry(angle_degrees=28.0, translation=1.8 + 0.5j)

    assert len(geometry.candidates) == 2
    assert geometry.candidate_residuals[0] == pytest.approx(0.0, abs=1e-10)
    assert geometry.candidates[0] == pytest.approx(geometry.target_point)
    assert np.abs(geometry.target_point - geometry.target_triangle) == pytest.approx(geometry.radii)


def test_bokeh_sources_include_source_and_target_distance_segments():
    from cnlecture.visualizations.triangle_distance_locator import _bokeh_sources

    geometry = locator_geometry()
    sources = _bokeh_sources(geometry)

    assert len(sources["source_distances"]["xs"]) == 3
    assert len(sources["target_distances"]["xs"]) == 3
    assert sources["source_distances"]["color"] == sources["target_distances"]["color"]


def test_bokeh_sources_show_candidate_points_without_labels():
    from cnlecture.visualizations.triangle_distance_locator import _bokeh_sources

    sources = _bokeh_sources(locator_geometry())

    assert "label" not in sources["candidates"]
    assert len(sources["candidates"]["x"]) == 2


def test_layout_exposes_motion_controls():
    pytest.importorskip("bokeh")
    from bokeh.models import Slider, Toggle

    from cnlecture.visualizations.triangle_distance_locator import (
        make_triangle_distance_locator_bokeh,
    )

    layout = make_triangle_distance_locator_bokeh()
    sliders = {m.title: m.value for m in layout.references() if isinstance(m, Slider)}
    toggles = {m.label: m.active for m in layout.references() if isinstance(m, Toggle)}

    assert sliders == {
        "theta (degrees)": 34.0,
        "translation x": 2.1,
        "translation y": 0.45,
    }
    assert toggles == {
        "theta = 0": False,
        "translation = 0": False,
        "overlap triangles": False,
    }
