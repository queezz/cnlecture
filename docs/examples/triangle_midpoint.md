# Triangle Side Squares and the Third-Side Midpoint

## Interactive figure

[Open the full interactive Bokeh figure](../assets/plots/triangle_midpoint.html){ .md-button .md-button--primary target="_blank" }

Drag the two red vertices.  The origin remains fixed.

## What this visualisation shows

This is the two-side piece of the quadrilateral picture.  Take the triangle
with vertices

\[
0,\quad 2a,\quad 2a+2b.
\]

Build squares on the sides \(2a\) and \(2b\).  Their centers are

\[
p=a+ia,
\qquad
s=2a+b+ib.
\]

The midpoint of the third side is

\[
m=a+b.
\]

The segments from the square centers to \(m\) are drawn directly.  In general
these are not the same as either of the full quadrilateral center-connecting
segments \(A\) or \(B\); they are local to this triangle.

## Reproducing the figure

```python
from cnlecture.visualizations import (
    export_triangle_midpoint_html,
    make_triangle_midpoint_bokeh,
)

layout = make_triangle_midpoint_bokeh()

export_triangle_midpoint_html("triangle_midpoint.html")
```
