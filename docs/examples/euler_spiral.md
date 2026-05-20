# Euler's Formula — Power Series Construction

## What this visualisation shows

This is **not** a generic spiral animation.

It is a geometric proof of Euler's formula using the Taylor series of \(e^{i\varphi}\):

\[
e^{i\varphi} \;=\; \sum_{k=0}^{\infty} \frac{(i\varphi)^k}{k!}
\;=\; 1 + i\varphi + \frac{(i\varphi)^2}{2!} + \frac{(i\varphi)^3}{3!} + \cdots
\]

Each term \(\dfrac{(i\varphi)^k}{k!}\) is a complex number.  When laid
**head-to-tail** in the complex plane they form a vector chain whose partial
sums spiral inward and converge to the exact point \(e^{i\varphi}\) on the
unit circle.

## Key observations

| Feature | Meaning |
|---------|---------|
| Blue ×  | Exact target \(e^{i\varphi} = \cos\varphi + i\sin\varphi\) |
| Red chain | Taylor terms laid head-to-tail |
| Orange dot | Partial sum \(S_n = \sum_{k=0}^{n-1} \frac{(i\varphi)^k}{k!}\) |
| Dashed circle | Unit circle \(\|z\|=1\) |

As \(n \to \infty\), the orange dot converges to the blue ×.

## Interactive figure

Use the **φ slider** to change the target angle.
The chain reshapes itself in real time.

<div style="width:100%; aspect-ratio:680/620; max-width:720px; margin:auto;">
  <iframe
    src="{{ base_url }}/assets/plots/euler_spiral.html"
    style="width:100%; height:100%; border:none; border-radius:6px;"
    loading="lazy"
    title="Euler spiral — power-series construction">
  </iframe>
</div>

!!! note "Offline / local use"
    The embedded figure loads Plotly from a CDN.  If you are viewing this
    offline, open `examples/EulersSpiral.ipynb` in JupyterLab instead —
    it uses the local `ipywidgets` interactive version.

## Reproducing the figure

```python
from cnlecture.visualizations import make_euler_spiral_plotly, export_euler_spiral_html

# Static figure at a fixed φ
fig = make_euler_spiral_plotly(phi=1.5, n_terms=25)
fig.show()

# Interactive slider version exported to HTML
export_euler_spiral_html("euler_spiral.html", phi=1.0, n_terms=30)
```

## Mathematical note

Multiplying by \(i\) rotates a complex number by \(90°\).
So each successive term \(\dfrac{(i\varphi)^k}{k!}\) is a rotated and
scaled copy of the previous one.  The shrinking magnitude and perpetual
\(90°\) rotation is why the chain curves — and why it converges to a
point **on the unit circle** for any real \(\varphi\).
