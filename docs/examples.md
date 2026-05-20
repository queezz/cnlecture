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

## Notebooks

All notebooks live in [`examples/`](https://github.com/queezz/cnlecture/tree/master/examples)
on GitHub.  Clone the repo and open them in JupyterLab:

```bash
git clone https://github.com/queezz/cnlecture.git
cd cnlecture
pip install -e ".[dev]"
jupyter lab
```

| Notebook | Topic |
|----------|-------|
| [`PrimitiveRoots.ipynb`](https://github.com/queezz/cnlecture/blob/master/examples/PrimitiveRoots.ipynb) | Roots of unity, primitive roots |
| [`ModularSurface.ipynb`](https://github.com/queezz/cnlecture/blob/master/examples/ModularSurface.ipynb) | Modular surface \(\|w\| = \|z^n\|\) |
| [`EulersSpiral.ipynb`](https://github.com/queezz/cnlecture/blob/master/examples/EulersSpiral.ipynb) | Euler's formula power-series construction |
| [`PowerSeries.ipynb`](https://github.com/queezz/cnlecture/blob/master/examples/PowerSeries.ipynb) | Power series convergence |
| [`interactive_demo.ipynb`](https://github.com/queezz/cnlecture/blob/master/examples/interactive_demo.ipynb) | Plotly + ipywidgets demo |
