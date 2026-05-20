# Known Issues and Pending Work

This file tracks unresolved issues found during the infrastructure migration.

---

## Fixed during migration

| Issue | Status | Notes |
|-------|--------|-------|
| `sys.path.insert(0, "../")` in `PrimitiveRoots.ipynb` and `ModularSurface.ipynb` | **Fixed** | Replaced with `from cnlecture import functions as fc` — requires editable install |
| `from mpl_toolkits.mplot3d import Axes3D` in `cplotting.py` | **Fixed** | Import was unused; removed (Axes3D registers automatically in modern Matplotlib) |
| `logb` function in `cplotting.py` always returned 1 | **Fixed** | Was `np.log(base)/np.log(base)`; corrected to `np.log(arg)/np.log(base)` |
| `pyproject.toml` used `hatchling` but package had no `src/` layout | **Fixed** | Migrated to `setuptools` with `src/` layout |

---

## Known limitations

### `plot3dexample.py` — pyqtgraph dependency

`src/cnlecture/plot3dexample.py` depends on `pyqtgraph` and `PyOpenGL`, which
are not in the standard dependencies.  The file is kept for reference but is
**not imported** by the package `__init__`.  It requires a desktop Qt
environment and cannot run in a headless CI or JupyterHub.

**Resolution options:**
- Port the example to Plotly / Vispy for web-native rendering
- Keep as an optional desktop script, document separately

### `examples/cplot.ipynb` — third-party `cplot` package

This notebook uses the [`cplot`](https://github.com/nschloe/cplot) package,
which is not bundled here.  Install it separately if needed:

```bash
pip install cplot
```

### `examples/pyqtgraph3d.ipynb` — desktop Qt required

Requires `pyqtgraph` and a running Qt event loop.
Does not work in standard JupyterLab without additional Qt backend setup.

---

## Pending improvements

- [ ] Port `plot3dexample.py` to Plotly / Vispy
- [ ] Add `nbval` or `nbmake` CI step to smoke-test notebooks on every push
- [ ] Extend pytest suite to cover `cplotting` functions (needs `matplotlib` Agg backend)
- [ ] Add type annotations to public API
- [ ] Consider `mkdocs-jupyter` `execute: true` once notebooks are dependency-clean
