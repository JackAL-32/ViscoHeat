# Library
import re
import Rappture
import sys
from numpy import *
import numpy as np
from numpy import pi
import pylab as plt
from colorsys import hls_to_rgb
from matplotlib import pyplot  as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import *
#import pandas as pd
#import seaborn as sns
from matplotlib.pyplot import *

#Define Input
io = Rappture.PyXml(sys.argv[1])
# Sys Config
P1 = float(io['input.group(InputParameters).group(SysConfig).number(Amplitude).current'].value)
a = float(io['input.group(InputParameters).group(SysConfig).number(CrystalRadius).current'].value)
f  = float(io['input.group(InputParameters).group(SysConfig).number(Omega).current'].value)

# MechProp
v1 = float(io['input.group(InputParameters).group(MechProp).number(LongWaveSpeed).current'].value)
X1 = float(io['input.group(InputParameters).group(MechProp).number(LongWaveAttenuation).current'].value)
v2 = float(io['input.group(InputParameters).group(MechProp).number(ShearWaveSpeed).current'].value)
# X2 = float(io['input.group(InputParameters).group(MechProp).number(ShearWaveAttenuation).current'].value)
p = float(io['input.group(InputParameters).group(MechProp).number(Density1).current'].value)
p2 = float(io['input.group(InputParameters).group(MechProp).number(Density2).current'].value)

# ThermalProp
k1 = float(io['input.group(InputParameters).group(HeatProperties).number(ThermalConductivity).current'].value)
gamma = float(io['input.group(InputParameters).group(HeatProperties).number(ThermalDiffusivity).current'].value)
h = float(io['input.group(InputParameters).group(HeatProperties).number(ConvectionCoef).current'].value)

# Parameter Equations



#-- Plot... ------------------------------------------------
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
ax.contourf(theta, R, transpose(values))
fig.savefig("q1.png")


io['output.image(outi).about.label']="Phi Stress on the Phi Face Image"
io.put('output.image(outi).current',"phiPhi.png",type='file',compress=True)


io['output.image(outj).about.label']="Radial Stress on the Radius Face Image"
io.put('output.image(outj).current',"rR.png",type='file',compress=True)
#io['output.image(outj).note.contents'] = htmltext


io['output.image(outk).about.label']="Theta Stress on the Theta Face Image"
io.put('output.image(outk).current',"thetaTheta.png",type='file',compress=True)
#io['output.image(outk).note.contents'] = htmltext


io['output.image(outl).about.label']="Radial Stress on the Theta Face Image"
io.put('output.image(outl).current',"rTheta.png",type='file',compress=True)
#io['output.image(outl).note.contents'] = htmltext

# idk what this graph is called
io['output.image(outm).about.label']="Heat Image"
io.put('output.image(outm).current',"q1.png",type='file',compress=True)

io.close()
