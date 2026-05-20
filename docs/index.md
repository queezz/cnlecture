# cnlecture

**Complex-number visualisation tools for lecture use.**

This package provides a small, focused set of Python utilities for building
visual intuition about complex analysis: domain colouring, modular surfaces,
streamplots, primitive roots, and more.

---

## What is this?

Complex numbers are easiest to understand through pictures.  
This project collects the plotting infrastructure used in lecture demonstrations so that examples are:

- **reproducible** — install once, run anywhere
- **interactive** — tweak parameters, see the geometry move
- **readable** — minimal code, maximum insight

---

## Quick example

```python
import numpy as np
from cnlecture import functions as fc

fc.modular_surface(power=3)
```

Or for domain colouring:

```python
import numpy as np
from cnlecture import cplotting as cp

n = 300
x = np.linspace(-2, 2, n)
y = np.linspace(-2, 2, n)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

f = (Z - 1) / (Z + 1)          # Möbius transformation
cp.domain_coloring(X, Y, f)
```

---

## Mathematics covered

| Topic | Notebook |
|-------|----------|
| Primitive roots of unity | [PrimitiveRoots](notebooks/PrimitiveRoots.ipynb) |
| Modular surfaces \( \|w\| = \|z^n\| \) | [ModularSurface](notebooks/ModularSurface.ipynb) |
| Euler's spiral (Cornu spiral) | [EulersSpiral](notebooks/EulersSpiral.ipynb) |
| Power series convergence | [PowerSeries](notebooks/PowerSeries.ipynb) |
| Interactive parameter demo | [interactive_demo](notebooks/interactive_demo.ipynb) |

---

## Installation

```bash
python -m venv ~/.venvs/cnlecture
source ~/.venvs/cnlecture/bin/activate      # Windows: Activate.ps1
pip install --upgrade pip
pip install -e ".[dev]"
```

See [Getting Started](getting_started.md) for the full workflow.
