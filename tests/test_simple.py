"""Basic smoke tests — run with: pytest"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

from cnlecture import functions as fc  # noqa: E402


def test_polygon():
    xs, ys = fc.polygon(6)
    assert len(xs) == 7  # n+1 points (closed polygon)


def test_cneq():
    assert fc.cneq(1 + 1j, 1 + 1j)
    assert not fc.cneq(1 + 1j, 1 + 2j)


def test_cnunique():
    seq = [1 + 0j, 1 + 0j, 0 + 1j]
    result = fc.cnunique(seq)
    assert len(result) == 2


def test_primitive_roots(tmp_path):
    import matplotlib.pyplot as plt

    fc.primitive_roots(5)
    plt.close("all")


def test_modular_surface():
    import matplotlib.pyplot as plt

    fig = fc.modular_surface(2)
    assert fig is not None
    plt.close("all")
