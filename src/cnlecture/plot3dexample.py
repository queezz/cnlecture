"""
This example demonstrates the use of GLSurfacePlotItem.
"""

import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore

## Create a GL View widget to display data
app = pg.mkQApp("GLSurfacePlot Example")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle("pyqtgraph example: GLSurfacePlot")
w.setCameraPosition(distance=10)

## Add a grid to the view
g = gl.GLGridItem()
# g.scale(1, 1, 1)
g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
w.addItem(g)


## Complex plot
n = 100
b = 3
y = np.linspace(-b * 1j, b * 1j, n)
x = np.linspace(-b, b, n)
X, Yc = np.meshgrid(x, y)
X, Y = np.meshgrid(x, y.imag)
Z = X + Yc
f = (Z - 1 - 1j) / (Z + 1 + 1j)
# f = np.sin(Z)
# f = Z ** 2
# f = Z
# W = abs(np.cos(np.angle(f)))
W = abs(f)

shader = ["balloon", "heightColor", "shaded", "normalColor"]
p2 = gl.GLSurfacePlotItem(x=x, y=y.imag, z=W, shader="normalColor")
p2.translate(b, -b, 0)
w.addItem(p2)

# another plot
W1 = abs(np.cos(np.angle(f)))
p3 = gl.GLSurfacePlotItem(x=x, y=y.imag, z=W1, shader="normalColor")
# p3.scale(16./49., 16./49., 1.0)
p3.translate(-b, -b, 0)
w.addItem(p3)

# another plot
W2 = np.angle(f)
p4 = gl.GLSurfacePlotItem(x=x, y=y.imag, z=W2, shader="normalColor")
p4.translate(-b, b, 0)
w.addItem(p4)

# another plot
W3 = np.cos(np.angle(f))
p4 = gl.GLSurfacePlotItem(x=x, y=y.imag, z=W3, shader="normalColor")
p4.translate(b, b, 0)
w.addItem(p4)


if __name__ == "__main__":
    pg.exec()
