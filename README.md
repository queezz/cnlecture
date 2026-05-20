# cnlecture

[![Tests](https://github.com/queezz/cnlecture/actions/workflows/python-package.yml/badge.svg)](https://github.com/queezz/cnlecture/actions/workflows/python-package.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://queezz.github.io/cnlecture/)

Complex-number visualisation tools for lecture use — domain colouring, modular
surfaces, streamplots, primitive roots, and interactive parameter controls.

## Quick start

```bash
python -m venv ~/.venvs/cnlecture
source ~/.venvs/cnlecture/bin/activate   # Windows: Activate.ps1
pip install --upgrade pip
git clone https://github.com/queezz/cnlecture.git && cd cnlecture
pip install -e ".[dev]"
```

## Usage

```python
from cnlecture import functions as fc
from cnlecture import cplotting as cp

fc.primitive_roots(7)          # roots of unity
fc.modular_surface(power=3)    # 3-D surface |w| = |z³|
```

Open the notebooks:

```bash
jupyter lab
```

Build the docs locally:

```bash
mkdocs serve
```

Run the tests:

```bash
pytest
```

## Documentation

Full documentation at <https://queezz.github.io/cnlecture/>.

## License

MIT
