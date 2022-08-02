import matplotlib.pyplot as plt
import numpy as np


def polygon(n=6):
    """
    Calculate vertices for an n-gone, inscirbed in a unit circle.
    First vertice at 1.
    """
    t = np.arange(0, np.pi * 2 * (n + 1) / n, 2 * np.pi / n)
    return (np.cos(t), np.sin(t))


def vec(a, b, **kws):
    plt.plot([a.real, b.real], [a.imag, b.imag], "-", **kws)


def plot_circ(r=1, o=0 + 0j, **kws):
    """
    Plot a circle.
    """
    cx, cy = polygon(100)
    c = kws.get("c", "k")
    ls = kws.get("ls", "-")
    for i in ["c", "ls"]:
        try:
            kws.pop(i)
        except:
            pass
    plt.plot(cx * r + o.real, cy * r + o.imag, c=c, ls=ls, **kws)


def roots_of_one(power=3):
    """
    Plot roots of 1. 
    """
    plot_circ()

    x, y = polygon(power)
    [vec(0, i + 1j * j) for i, j in zip(x, y)]
    plt.plot(x, y, "-o")

    plt.plot([0], [0], "ko")

    plt.gca().set_aspect("equal")
    plt.axis("off")


def cneq(a, b, threshold=1e-14):
    """
    Compares complex numbesr to find if they are equal.
    This is needed because of the floating point precision.
    """
    return abs(a - b) < threshold


def cnin(a, seq):
    """
    Check if a complex number `a` is in a sequenc `seq`
    """
    for z in seq:
        if cneq(a, z):
            return True
    return False


def cnunique(seq):
    """
    Remove duplicates from a sequence of complex numbers.
    """
    return [z for n, z in enumerate(seq) if not cnin(z, seq[:n])]


def primitive_roots(n):
    """
    Plot primitive roots for an n-gone in red. 
    Other roots are indicated in gray.
    """
    roots_of_one(n)

    x, y = polygon(n)
    vs = [i + 1j * j for i, j in zip(x, y)]

    ax = plt.gca()

    for k, v in enumerate(vs[0 : n // 2 + 1]):
        r = 1 + 0.2 * (k + 1)
        d = cnunique([vs[k] ** d for d in range(1, n + 1)])
        if len(d) == n:
            plot_circ(r=r, c="r")
            c, marker = ["r", "o"]
            col = "C3"
            vec(0, v, c=col)
            vec(0, v.conjugate(), c=col)
            plt.plot(v.real, v.imag, "o", c=col)
            plt.plot(v.real, -v.imag, "o", c=col)
        else:
            plot_circ(r=r, c="gray", ls="--")
            c, marker = ["gray", "s"]

        [plt.plot(a.real * r, a.imag * r, marker=marker, c=c) for a in d]
        ax.text(r, 0.1, k)

    ax.text(0.3, 0.1, "$m\\frac{2\\pi}{n}$")
    ax.text(1.15, 0.3, "$m$")


def modular_surface(power):
    """
    Plot modular surface of a polynomial w = z ^ {power}
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Create the mesh in polar coordinates and compute corresponding Z.
    r = np.linspace(0, 1.0, 100)
    p = np.linspace(0, 2 * np.pi, 50)
    R, P = np.meshgrid(r, p)
    W = R ** power  # Modulus of W

    # Express the mesh in the cartesian system.
    X, Y = R * np.cos(P), R * np.sin(P)

    # Plot the surface.
    # ax.contour3D(X, Y, W, cmap=plt.cm.viridis)
    ax.plot_surface(X, Y, W, cmap=plt.cm.viridis)

    # Tweak the limits and add latex math labels.
    ax.set_zlim(0, 1)
    ax.set_xlabel(r"$\mathrm{Re}(z)$")
    ax.set_ylabel(r"$\mathrm{Im}(z)$")
    ax.set_zlabel(r"$|w|$")

    return fig
