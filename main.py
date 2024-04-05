# #Define Input
# io = start_io(sys.argv[1])
# # Sys Config
# P1 = get_io("SysConfig", "Amplitude")
# a = get_io("SysConfig", "CrystalRadius")
# f = get_io("SysConfig", "Omega")
# N = get_io("SysConfig", "VariableN")
# 
# # MechProp
# v1 = get_io("MechProp", "LongWaveSpeed")
# X1 = get_io("MechProp", "LongWaveAttenuation")
# v2 = get_io("MechProp", "ShearWaveSpeed")
# p = get_io("MechProp", "Density1")
# p2 = get_io("MechProp", "Density2")
# 
# # ThermalProp
# k1 = get_io("HeatProperties", "ThermalConductivity")
# gamma = get_io("HeatProperties", "ThermalDiffusivity")
# h = get_io("HeatProperties", "ConvectionCoef")
# #T0 = get_io("HeatProperties", "InitTemp")


P1 = 0.000001
a = 0.00025
f = 500000
N = 30

v1 = 1100
X1 = 0.024
v2 = 570
p = 1030
p2 = 1910

k1 = 0.27
gamma = 0.000000102
h = 5
T0 = 21

import equations.parameters as param
param.init([P1, a, f, N, v1, X1, v2, p, p2, k1, gamma, h, T0])

from equations.parameters import *
# Library
import re
#import Rappture
import numpy as np
import pylab as labplt
from colorsys import hls_to_rgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as scisp
# from io_rap import *
from equations.scipy_funcs import *
from equations.stress import *
from equations.graph import * #Need to fix this
from equations.temperature import *
import multiprocessing
import time

begin = time.time()

def worker(func, result_dict):
    result_dict[func.__name__] = func()

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    sigmas = manager.dict()

    funcs = [get_sigrr, get_sigphiphi, get_sigthatha, get_sigrtha]

    processes = []
    for func in funcs:
        process = multiprocessing.Process(target=worker, args=(func, sigmas))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    sigrr = sigmas["get_sigrr"]
    sigphiphi = sigmas["get_sigphiphi"]
    sigthatha = sigmas["get_sigthatha"]
    sigrtha = sigmas["get_sigrtha"]

    # Stress Graphs
    graph_eq(sigrr, name = "rR"); print("sigrr graph done...")

    graph_eq(sigphiphi, name = "phiPhi"); print("sigpp graph done...")

    graph_eq(sigthatha, name = "thetaTheta"); print("sigtt graph done...")

    graph_eq(sigrtha, name = "rTheta"); print("sigrt graph done...")

    # Volumetric Heat Generation Graph
    start = time.time()
    q1 = get_q1(sigrr, sigphiphi, sigthatha, sigrtha)
    stop = time.time()
    print(f"q1 calculated in {stop - start:.2f} seconds...")

    graph_eq(q1, "q1"); print("q1 graph done...")

    # Input for Tempurature Gradient

    #Initial Tempurature
    T0 = 21 #degC

    #Theta increment
    dtha = tha[1]-tha[0]
    
    #r increment
    dr = a

    #t array and increment
    dt = 0.01*(dr**2/gamma * 0.2)
    t = np.arange(0,0.5,dt)

    q1 = dt*(gamma/k1)*q1

    # Temperature Gradient
    start = time.time()
    #Iteration Matrices
    Hmat = H(r,tha,gamma,dt,dr,dtha)
    Hbmat = Hb(r,tha,gamma,dt,dr,dtha) # Should go away

    #Temperature Array
    T1 = Tnew_t(Hmat,T0,Hbmat,q1,r,t,dt).round(2)

    #Filling in Gap
    T = FillGap(T1,r,tha1,t)
    stop = time.time()
    print(f"temp matrix calculated in {stop - start:.2f} seconds...")

    # Graph Temperature Gradient

    graph_temp_gr(T[:,:,(t.size-1)], shift = 21, name = "tempGradient"); print("temp graph done...")

    end = time.time()
    print(f"program completed in {end - begin:.2f} seconds!")

    # #Output IO
    # put_io("outi", "phiPhi", "Phi Stress on the Phi Face Image")
    # put_io("outj", "rR", "Radial Stress on the Radius Face Image")
    # put_io("outk", "thetaTheta", "Theta Stress on the Theta Face Image")
    # put_io("outl", "rTheta", "Radial Stress on the Theta Face Image")
    # put_io("outm", "q1", "Volumetric Heat Generation Image")
    # put_io("outn", "T", "Temperature Gradient Image")
    # 
    # close_io()