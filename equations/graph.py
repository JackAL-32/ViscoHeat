import numpy as __np
import matplotlib.pyplot as __plt
from matplotlib.colors import Normalize
from matplotlib import cm
from equations.parameters import *
import time

__plt.jet()

R, theta = __np.meshgrid(r, tha1)

X = R * __np.cos(theta)
Y = R * __np.sin(theta)

cmap = __plt.get_cmap('jet',10000)

fig, ax = __plt.subplots(subplot_kw=dict())

def graph_eq(sig, name = str(time.time())[7:-4], factor = 10e6, dir = "./pictures/", colorbar_ticks = "0.5 or 1, 0.2 for T"):
    values = __np.abs(sig)
    ax.contourf(X, Y, values)

    vmax = __np.floor(__np.amax(values)/factor)
    vmin = __np.floor(__np.amin(values)/factor)

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    #colorbar(sm, ticks=arange(vmin,vmax,1),boundaries=arange(vmin,vmax,.001))

    __plt.gca().set_aspect("equal")
    __plt.xlabel("z (mm)")
    __plt.ylabel("x (mm)")
    fig.savefig(dir + name + ".png")

def graph_temp_gr(values, shift = 21, name = str(time.time())[7:-4], dir = "./pictures"):
    ax.contourf(X, Y, values - shift)
    vmax = __np.floor(__np.amax(__np.abs(values)))
    vmin = __np.floor(__np.amin(__np.abs(values)))

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    #colorbar(sm, ticks=arange(vmin,vmax,10),boundaries=arange(vmin,vmax,.001))
    
    __plt.gca().set_aspect("equal")
    fig.savefig(dir + name + ".png")
    
