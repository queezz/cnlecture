# Cotes' Theorem and Roots of Unity

## Interactive figure

[Open the full interactive Bokeh figure](../assets/plots/cotes_theorem.html){ .md-button .md-button--primary target="_blank" }

The \(n\) slider changes the regular polygon inscribed in the unit circle.  The
\(x\) slider moves \(P\) along the ray through \(C_1=1\), keeping \(x>1\) so the
distance product is positive:

\[
U_n(x)=x^n-1=PC_1\,PC_2\cdots PC_n.
\]

The first panel draws the \(n\)-gon and the distance segments from \(P\) to the
vertices.

The second panel can be hidden.  It stays in the complex plane: the point
\(z=x+iy\) is joined to each root \(C_k\), so each segment represents the
complex factor \(z-C_k=R_ke^{i\phi_k}\).  The magenta vector from the origin
shows the product

\[
\prod_{k=1}^n (z-C_k)=z^n-1.
\]

For large \(n\) or large \(|z|\), this product may be too long to draw at true
scale.  In that case the arrow keeps the correct direction and is labeled with
its display scale, while the summary reports the true complex value.

## Real factors from conjugate vertices

Taking \(O\) as the origin and \(C_1=1\), the vertices are the roots of unity

\[
C_{k+1}=e^{2\pi i k/n}.
\]

The complex factorization is

\[
z^n-1=(z-C_1)(z-C_2)\cdots(z-C_n).
\]

For real \(x\), conjugate vertices combine into real quadratic factors:

\[
(x-e^{2\pi i k/n})(x-e^{-2\pi i k/n})
  = x^2-2x\cos\left(\frac{2\pi k}{n}\right)+1.
\]

## Reproducing the figure

```python
from cnlecture.visualizations import (
    export_cotes_theorem_html,
    make_cotes_theorem_bokeh,
)

layout = make_cotes_theorem_bokeh(n=5, x=1.65)

export_cotes_theorem_html("cotes_theorem.html")
```
