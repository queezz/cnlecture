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

---

## Squares on the Sides of a Quadrilateral

Drag three vertices of a quadrilateral while one vertex stays fixed at the
origin.  Squares are built on the sides \(2a,2b,2c,2d\), and the figure tracks
the complex identity \(A+iB=0\) for the segments connecting opposite square
centers.

See the interactive page: [Squares on the Sides of a Quadrilateral](examples/quadrilateral_squares.md).

---

## Triangle Side Squares and the Third-Side Midpoint

Look at the two-side triangle piece separately: squares on \(2a\) and \(2b\),
their centers \(p\) and \(s\), and the midpoint \(m=a+b\) of the third side.

See the interactive page: [Triangle Side Squares and the Third-Side Midpoint](examples/triangle_midpoint.md).

---

## Spiral Velocity Geometry

Compare the finite step \(M=Z(t+\delta)-Z(t)\) on
\(Z(t)=e^{at}e^{ibt}\) with the infinitesimal right triangle
\((a+ib)Z\delta\).  A \(\delta\) slider makes the failure and recovery of
perpendicularity visible.

See the interactive page: [Spiral Velocity Geometry](examples/spiral_velocity.md).

---

## Cotes' Theorem and Roots of Unity

Move \(P=x\) on the real axis while a regular \(n\)-gon supplies the roots of
unity \(C_1,\ldots,C_n\).  The figure tracks Cotes' distance product
\(PC_1PC_2\cdots PC_n=x^n-1\) and the corresponding real factor grouping.

See the interactive page: [Cotes' Theorem and Roots of Unity](examples/cotes_theorem.md).
