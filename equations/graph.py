import numpy as __np
import matplotlib.pyplot as __plt
from matplotlib.colors import Normalize, Colormap
from matplotlib import cm
from equations.parameters import *
import time

__plt.jet()

R, theta = __np.meshgrid(r, tha1)

X = R * __np.cos(theta)
Y = R * __np.sin(theta)

cmap = __plt.get_cmap('jet')

# Should be 10e6? Need to figure out why/if 10x difference.
def graph_eq(sig, name = str(time.time())[7:-4], factor = 10e5, dir = "./pictures/"):
    fig, ax = __plt.subplots()
    values = __np.abs(sig)
    ax.contourf(X, Y, values, 256)

    vmax = __np.ceil(__np.amax(values)/factor)
    vmin = __np.floor(__np.amin(values)/factor)

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # colorbar_ticks = "0.5 or 1, 0.2 for T"
    # From last year ^
    if(name == "rR"): tickstep = 1
    else: tickstep = .5

    if (name == "q1"): title = "Heat "
    else: title = "Stress "

    fig.colorbar(sm, ticks=__np.arange(0,vmax,tickstep), boundaries=__np.arange(0,vmax,.001), ax=ax)

    __plt.gca().set_aspect("equal")
    __plt.xlabel("z (mm)")
    __plt.ylabel("x (mm)")
    __plt.title(title + name)
    fig.savefig(dir + name + ".png")

# shift is T0?
def graph_temp_gr(values, shift = 21, name = str(time.time())[7:-4], dir = "./pictures/"):
    fig, ax = __plt.subplots()
    ax.contourf(X, Y, values - shift, 256)

    vmax = __np.ceil(__np.amax(__np.abs(values))) - shift # T0?
    vmin = __np.floor(__np.amin(__np.abs(values))) - shift # T0?

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    tickstep = .2 # From last year

    fig.colorbar(sm, ticks=__np.arange(0,vmax,tickstep), boundaries=__np.arange(0,vmax,.001), ax=ax)
    
    __plt.gca().set_aspect("equal")
    __plt.title(name)
    fig.savefig(dir + name + ".png")
    
