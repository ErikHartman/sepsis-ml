import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import random

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

x = np.arange(-1,1, .1)
y = np.arange(-1,1, .1)

X, Y = np.meshgrid(x, y)
Z = X**2+Y**2

Z = Z +np.random.rand(20,20)/6


# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.

ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
plt.axis('off')
plt.grid(b=None)
# Add a color bar which maps values to colors.
plt.savefig('src/background/loss.png', dpi=500
            )