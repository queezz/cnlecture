# Spiral Velocity Geometry

## Interactive figure

[Open the full interactive Bokeh figure](../assets/plots/spiral_velocity.html){ .md-button .md-button--primary target="_blank" }

Four sliders drive the figure: \(a\) (growth rate), \(b\) (angular speed),
\(t\) (which point \(Z(t)\) the construction sits on), and \(\delta\) (the
finite step).  The first panel auto-scales to keep the whole construction in
view as you change them.

Move the \(\delta\) slider.  For a visible finite step, the radial part \(A\)
and the circular chord \(B\) are not exactly perpendicular.  The zoomed plot
normalises the step by \(|Z|\delta\), so the picture stays readable even when
\(\delta\) is small.  As \(\delta\) shrinks, the finite triangle approaches the
right-triangle picture used to read

\[
M \sim (a+ib)Z\delta,
\qquad
V = \frac{dZ}{dt} = (a+ib)Z.
\]

## What this visualisation shows

The spiral is

\[
Z(t)=e^{at}e^{ibt}.
\]

The finite movement from \(t\) to \(t+\delta\) is

\[
M=Z(t+\delta)-Z(t).
\]

The first panel follows textbook Figure 16.  The bold arrow is the velocity
\(V=(a+ib)\,Z\), the tangent to the spiral at \(Z(t)\); the finite chord \(M\)
approaches the direction of \(V\) as \(\delta\to 0\).

Near the origin, the triangle marks \(a+ib\) at its actual endpoint \((a,b)\), so
its length is \(\sqrt{a^2+b^2}\), not artificially one.  The dashed copy is
\(e^{ibt}(a+ib)\): the same shape rotated by the angle of \(Z(t)\), hence
parallel to \(V\).  The point \(1\) on the unit circle and the small \(b\delta\)
wedge between the two radii fix the scale and the angular step of one move.

At \(Z(t)\), the defining chord for \(B\) is shown on the smaller arc through
\(Z(t)\).  The top green edge is kept as the corresponding edge of the shaded
finite triangle at the radius of \(Z(t+\delta)\).  The orange side \(A\) starts
at \(Z(t)\) and runs radially to that top edge, and the pink side \(M\) is the
actual finite movement from \(Z(t)\) to \(Z(t+\delta)\).  The full dashed circle
is the unit circle; the two short dashed arcs show the circles through \(Z(t)\)
and \(Z(t+\delta)\) only where they are relevant.

The second panel decomposes \(M\) into a radial change \(A\) and a circular
chord \(B\), then divides all lengths by \(|Z|\delta\).  The shaded triangle
keeps the finite step visible, while the dashed triangle shows the infinitesimal
model:

\[
A \sim aZ\delta,
\qquad
B \sim ibZ\delta.
\]

## Reproducing the figure

```python
from cnlecture.visualizations import (
    export_spiral_velocity_html,
    make_spiral_velocity_bokeh,
)

layout = make_spiral_velocity_bokeh()

export_spiral_velocity_html("spiral_velocity.html")
```
