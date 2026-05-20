# Getting Started

## Prerequisites

- Python 3.9 or newer
- A virtual environment (recommended)

---

## Create a virtual environment

=== "macOS / Linux"

    ```bash
    python -m venv ~/.venvs/cnlecture
    source ~/.venvs/cnlecture/bin/activate
    pip install --upgrade pip
    ```

=== "Windows PowerShell"

    ```powershell
    python -m venv $HOME\.venvs\cnlecture
    & "$HOME\.venvs\cnlecture\Scripts\Activate.ps1"
    python -m pip install --upgrade pip
    ```

---

## Install the package

Clone the repository and install in editable mode with all development tools:

```bash
git clone https://github.com/queezz/cnlecture.git
cd cnlecture
pip install -e ".[dev]"
```

The `[dev]` extras include JupyterLab, Plotly, ipywidgets, MkDocs, pytest, and Ruff.

---

## Launch Jupyter Lab

```bash
jupyter lab
```

Navigate to `examples/` and open any notebook.

---

## Run the tests

```bash
pytest
```

---

## Build the documentation locally

```bash
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## Project layout

```
cnlecture/
├── src/
│   └── cnlecture/          # importable package
│       ├── __init__.py
│       ├── functions.py    # geometry helpers, primitive roots, modular surface
│       └── cplotting.py    # domain coloring, streamplots, 3-D surfaces
├── examples/               # Jupyter notebooks (lecture examples)
├── docs/                   # MkDocs source
│   ├── notebooks/          # copies of notebooks for documentation
│   └── ...
├── tests/                  # pytest suite
├── pyproject.toml
└── mkdocs.yml
```

---

## Importing the package

```python
from cnlecture import functions as fc
from cnlecture import cplotting as cp
```

Both modules are available after `pip install -e .`.
