# Pogram : To show different types of sigmoid functions
# Author : Anant Shah
# Date : 23-1-2018

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plot for function h11() #
start = -1
stop = 1
x = np.linspace(start,stop,num=1000)
h11 = 1.0/(1.0+np.exp(-(500*x+30)))
h12 = 1.0/(1.0+np.exp(-(500*x-30)))
h21 = h11 - h12

plt.figure(1)
plt.plot(x,h11)
plt.grid(True)
plt.xlabel(r'$x$')
plt.ylabel(r'$h_{11}$')
plt.title(r'$h_{11} = \frac{1}{1+e^{-(500x+30)}}$')
plt.show()

plt.figure(2)
plt.plot(x,h12)
plt.grid(True)
plt.xlabel(r'$x$')
plt.ylabel(r'$h_{12}$')
plt.title(r'$h_{12} = \frac{1}{1+e^{-(500x-30)}}$')
plt.show()

plt.figure(3)
plt.plot(x,h21)
plt.grid(True)
plt.xlabel(r'$x$')
plt.ylabel(r'$h_{21}$')
plt.title(r'$h_{21} = h_{11} - h_{12}$')
plt.show()

# Plot the surface plots #
start_a = -5
stop_a = 5
x1 = x2 = np.linspace(start_a, stop_a, num=100)
X1, X2 = np.meshgrid(x1, x2)

h11_a = 1.0/(1.0 + np.exp(-(X1+50*X2+100)))
h12_a = 1.0/(1.0 + np.exp(-(X1+50*X2-100)))
h13_a = 1.0/(1.0 + np.exp(-(50*X1+X2+100)))
h14_a = 1.0/(1.0 + np.exp(-(50*X1+X2-100)))
h21_a = h11_a - h12_a
h22 = h13_a - h14_a
h31 = h21_a + h22
f = 1.0/(1.0 + np.exp(-(100*h31-200)))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, h11_a, rstride=1, cstride=1, cmap=None, linewidth=0)
ax.set_xlabel(r'$x_{1}$')
ax.set_ylabel(r'$x_{2}$')
ax.set_zlabel(r'$h_{11}$')
ax.set_title(r'$h_{11} = \frac{1}{1+e^{-(x_{1}+50x_{2}+100)}}$')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, h12_a, rstride=1, cstride=1, cmap=None, linewidth=0)
ax.set_xlabel(r'$x_{1}$')
ax.set_ylabel(r'$x_{2}$')
ax.set_zlabel(r'$h_{12}$')
ax.set_title(r'$h_{12} = \frac{1}{1+e^{-(x_{1}+50x_{2}-100)}}$')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, h13_a, rstride=1, cstride=1, cmap=None, linewidth=0)
ax.set_xlabel(r'$x_{1}$')
ax.set_ylabel(r'$x_{2}$')
ax.set_zlabel(r'$h_{13}$')
ax.set_title(r'$h_{13} = \frac{1}{1+e^{-(50x_{1}+x_{2}+100)}}$')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, h14_a, rstride=1, cstride=1, cmap=None, linewidth=0)
ax.set_xlabel(r'$x_{1}$')
ax.set_ylabel(r'$x_{2}$')
ax.set_zlabel(r'$h_{14}$')
ax.set_title(r'$h_{14} = \frac{1}{1+e^{-(50x_{1}+x_{2}-100)}}$')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, h21_a, rstride=1, cstride=1, cmap=None, linewidth=0)
ax.set_xlabel(r'$x_{1}$')
ax.set_ylabel(r'$x_{2}$')
ax.set_zlabel(r'$h_{21}$')
ax.set_title(r'$h_{21} = h_{11} - h_{12}$')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, h22, rstride=1, cstride=1, cmap=None, linewidth=0)
ax.set_xlabel(r'$x_{1}$')
ax.set_ylabel(r'$x_{2}$')
ax.set_zlabel(r'$h_{22}$')
ax.set_title(r'$h_{22} = h_{13} - h_{14}$')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, h31, rstride=1, cstride=1, cmap=None, linewidth=0)
ax.set_xlabel(r'$x_{1}$')
ax.set_ylabel(r'$x_{2}$')
ax.set_zlabel(r'$h_{31}$')
ax.set_title(r'$h_{31} = h_{21} + h_{22}$')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, f, rstride=1, cstride=1, cmap=None, linewidth=0)
ax.set_xlabel(r'$x_{1}$')
ax.set_ylabel(r'$x_{2}$')
ax.set_zlabel(r'$f$')
ax.set_title(r'$f = \frac{1}{1+e^{-(100h_{31}-200)}}$')
plt.show()
