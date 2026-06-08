# Squares on the Sides of a Quadrilateral

## Interactive figure

[Open the full interactive Bokeh figure](../assets/plots/quadrilateral_squares.html){ .md-button .md-button--primary target="_blank" }

Drag the three red vertices.  The origin remains fixed.

## What this visualisation shows

Fix one vertex of a quadrilateral at the origin and write its oriented sides as

\[
2a,\quad 2b,\quad 2c,\quad 2d.
\]

Because the sides close up,

\[
2a + 2b + 2c + 2d = 0,
\qquad\text{so}\qquad
a+b+c+d=0.
\]

Build a square on each side, using the same left-turn convention for every
oriented side.  The square centers are then

\[
a+ia,\quad
2a+b+ib,\quad
2a+2b+c+ic,\quad
-d+id.
\]

Now connect the centers of opposite squares.  Let \(A\) run from the center of
the square on \(2a\) to the center of the square on \(2c\), and let \(B\) run
from the center of the square on \(2d\) to the center of the square on \(2b\).
Then

\[
A+iB=0.
\]

So the two connecting segments remain perpendicular and equal in length as the
quadrilateral changes.

!!! note "Outward-facing squares"
    The figure uses the left-turn square convention, matching the formulas
    above.  For clockwise quadrilaterals these squares face outward.  If the
    dragged vertices are moved through a self-intersection or orientation
    reversal, the algebraic identity still holds, while "outward" becomes a
    convention rather than a visual side of a simple polygon.

## Reproducing the figure

```python
from cnlecture.visualizations import (
    export_quadrilateral_squares_html,
    make_quadrilateral_squares_bokeh,
)

layout = make_quadrilateral_squares_bokeh()

export_quadrilateral_squares_html("quadrilateral_squares.html")
```
