# Triangle Distance Locator

## Interactive figure

[Open the full interactive Bokeh figure](../assets/plots/triangle_distance_locator.html){ .md-button .md-button--primary target="_blank" }

Drag the point \(P\), or use the sliders to move the target triangle
\(A'B'C'\).  The zero toggles temporarily remove rotation, translation, or
both so the target triangle can be compared with the original one.

## What this visualisation shows

Needham notes that a motion is determined once we know where any
non-collinear triangle goes.  This figure shows the local reason for that
claim.

Start with a triangle \(A,B,C\), a corresponding target triangle \(A',B',C'\),
and a point \(P\).  Since motions preserve distance, the image point \(P'\)
must lie on all three circles

\[
|z-A'|=|P-A|,\qquad |z-B'|=|P-B|,\qquad |z-C'|=|P-C|.
\]

The first two circles usually meet in two points.  The third circle selects
the correct one.  Matching colors show the source distances \(AP,BP,CP\) and
their target counterparts \(A'P',B'P',C'P'\).

## Reproducing the figure

```python
from cnlecture.visualizations import (
    export_triangle_distance_locator_html,
    make_triangle_distance_locator_bokeh,
)

layout = make_triangle_distance_locator_bokeh()

export_triangle_distance_locator_html("triangle_distance_locator.html")
```
