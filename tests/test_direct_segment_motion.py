"""Tests for the direct segment motion visualization."""

import numpy as np
import pytest

from cnlecture.visualizations.direct_segment_motion import (
    apply_segment_motion,
    classify_direct_motion,
    default_segment,
    direct_motion_rotation_center,
    rotate_image_back,
    segment_motion_geometry,
)


def test_default_segment_has_distinct_endpoints():
    segment = default_segment()

    assert segment.shape == (2,)
    assert abs(segment[1] - segment[0]) > 0.5


def test_segment_motion_preserves_segment_length():
    segment = np.asarray([-1 - 0.2j, 0.8 + 0.6j])
    moved = apply_segment_motion(segment, 37.0, 1.5 - 0.4j)

    assert abs(moved[1] - moved[0]) == pytest.approx(abs(segment[1] - segment[0]))


def test_zero_angle_classifies_as_translation():
    geometry = segment_motion_geometry(angle_degrees=0.0, translation=1.2 + 0.3j)

    assert classify_direct_motion(0.0) == "translation"
    assert geometry.motion_kind == "translation"
    assert geometry.rotation_center is None
    assert geometry.target_segment == pytest.approx(geometry.source_segment + 1.2 + 0.3j)


def test_nonzero_angle_rotation_center_is_fixed_point():
    angle = 48.0
    translation = 1.8 - 0.25j
    center = direct_motion_rotation_center(angle, translation)
    geometry = segment_motion_geometry(angle_degrees=angle, translation=translation)

    assert center is not None
    assert geometry.motion_kind == "rotation"
    assert geometry.rotation_center == pytest.approx(center)
    assert apply_segment_motion([center], angle, translation)[0] == pytest.approx(center)
    assert apply_segment_motion(geometry.source_segment, angle, translation) == pytest.approx(
        geometry.target_segment
    )


def test_rotate_image_back_recovers_preimage_at_full_fraction():
    angle = 63.0
    translation = 1.1 + 0.45j
    geometry = segment_motion_geometry(angle_degrees=angle, translation=translation)

    assert geometry.rotation_center is not None
    halfway = rotate_image_back(
        geometry.target_segment,
        geometry.rotation_center,
        angle,
        0.5,
    )
    recovered = rotate_image_back(
        geometry.target_segment,
        geometry.rotation_center,
        angle,
        1.0,
    )

    assert rotate_image_back(geometry.target_segment, geometry.rotation_center, angle, 0.0) == pytest.approx(
        geometry.target_segment
    )
    assert recovered == pytest.approx(geometry.source_segment)
    assert halfway != pytest.approx(geometry.target_segment)


def test_bokeh_sources_switch_between_vector_and_rotation_center():
    from cnlecture.visualizations.direct_segment_motion import _bokeh_sources

    translation_sources = _bokeh_sources(segment_motion_geometry(angle_degrees=0.0))
    rotation_sources = _bokeh_sources(segment_motion_geometry(angle_degrees=30.0))

    assert translation_sources["source_segment"]["label"] == ["A", "B"]
    assert "source_labels" not in translation_sources
    assert len(translation_sources["vector"]["x0"]) == 1
    assert len(translation_sources["center"]["x"]) == 0
    assert len(rotation_sources["vector"]["x0"]) == 0
    assert len(rotation_sources["center"]["x"]) == 1
    assert len(rotation_sources["arc"]["x"]) > 0
    assert rotation_sources["back_segment"] == {"x": [], "y": [], "label": []}
    assert rotation_sources["back_radii"] == {"xs": [], "ys": []}


def test_layout_exposes_direct_motion_controls():
    pytest.importorskip("bokeh")
    from bokeh.models import Slider, Toggle

    from cnlecture.visualizations.direct_segment_motion import make_direct_segment_motion_bokeh

    layout = make_direct_segment_motion_bokeh()
    sliders = {m.title: m.value for m in layout.references() if isinstance(m, Slider)}
    slider_ranges = {m.title: (m.start, m.end) for m in layout.references() if isinstance(m, Slider)}
    toggles = {m.label: m.active for m in layout.references() if isinstance(m, Toggle)}

    assert sliders == {
        "theta (degrees)": 42.0,
        "translation x": 1.85,
        "translation y": 0.55,
        "rotate image back (%)": 0,
    }
    assert slider_ranges["theta (degrees)"] == (-180, 180)
    assert slider_ranges["rotate image back (%)"] == (0, 100)
    assert toggles == {}
