import numpy as np

from regression import *
from error import *

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


N = 30
M = 0
S = 20

dx = np.linspace(-20, 20, N) + np.random.normal(M, S, N)
dy = np.linspace(-20, 20, N) + np.random.normal(M, S, N)
dz = 26 + 2.3 * dx - 4.8 * dy + np.random.normal(M, S, N)

reg = multilinear_regression(dz, np.array([dx, dy]).T)

print(np.array([26, 2.3, -4.8]))
print(reg)

ax: Axes3D
fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    subplot_kw={'projection': '3d'}
)

ax.scatter3D(xs=dx, ys=dy, zs=dz)
gx, gy = np.meshgrid(np.linspace(-80, 80, N), np.linspace(-80, 80, N))
gz = reg[0] + reg[1] * gx + reg[2] * gy

ax.plot_surface(X=gx, Y=gy, Z=gz, cmap="rainbow", alpha=0.5)
plt.show()