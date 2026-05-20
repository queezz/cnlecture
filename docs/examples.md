# Examples

All notebooks live in `examples/` and are also rendered here in the documentation.

---

## Primitive Roots of Unity

Visualise the \(n\)-th roots of unity on the unit circle and highlight the **primitive** roots.
A primitive root \(\omega\) generates all others: \(\{\omega, \omega^2, \ldots, \omega^n = 1\}\).

```python
from cnlecture import functions as fc
fc.primitive_roots(n=7)
```

---

## Modular Surface

Plot the surface \(|w| = |z^n|\) over the unit disc.  
As \(n\) increases, the surface sprouts more and more "petals".

```python
from cnlecture import functions as fc
fig = fc.modular_surface(power=4)
```

---

## Domain Colouring

Encode the **argument** of \(f(z)\) as hue and the **modulus** as brightness.
Poles appear as points where all colours converge; zeros appear dark.

```python
import numpy as np
from cnlecture import cplotting as cp

n = 400
x = y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

cp.domain_coloring_illuminated(X, Y, Z**2 - 1)
```

---

## Interactive Demonstrations

The `examples/` folder also contains:

- **`_interactive_notebook.ipynb`** — ipywidgets slider controlling plot parameters
- **`interactive_demo.ipynb`** — Plotly-based rotating complex number

See the [Notebooks](notebooks/interactive_demo.ipynb) section for the rendered versions.
