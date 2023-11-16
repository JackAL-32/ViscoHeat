#Imports

#Import entire numpy library
from numpy import *

#Import entire math library
from math import *

#Import scipy.special library
from scipy.special import *

#Import entire pandas library as pd
import pandas as pd

#Import entire matplotlib
import matplotlib as plt

#This is used for variable display in the GUI
import sys
#Define Input

#rho: Desity of the Medium
p = 1

#Phi_0: Amplitude of the Incident Wave
P = 1

#Betta_1: Complex Shear Wave Number
B1 = 1

#alpha_1 = Complex Longitudinal Wave Number
a1 = 1

#a: radius of the chrystal
a = 1

#omega: angular frequency
w = 1
#Parameter Equations

#N_p
def Np(n):
    return (4 * n**2) + (n+1) * (B1/a1)**2 

#A_n
def An(n):
    p1 = 1
    p2 = 1
    
    if n == 0:
        return -1/3 * 1j * P * (a1*a)**3
        
    if n == 1:
        return -1/3 * 1j * P * (1 - (p2/p1)) * (a1*a)**3
        
    if n > 1:
        return (1j)**(n-1) * P * ( (2**n * factorial(n))/factorial(2*n) )**2 * Np(n)*(a1*a)**(2*n-1)

#B_n
def Bn(n):
    if n == 0:
        return 1
    
    if n > 0:
        return -(B1/a1)**(n+1) * (An(n)/n)

#mu_1
u1 = p * w**2 / B1

#P_n: Legendre Pollynomials
def Pn(n):
    return legendre(n)
#n: int
#Degree of the polynomial.

#h_n: Spherical Hankel Function of the first kind
def hn(n,z):
    return hankel1(n,z)
#n: array_like
#Order (float)

#z: array_like
#Argument (float or complex)
#Basically whatever is in the parentheses


#j_n: Spherical Bessel Function of the first kind
def jn(n,z):
    return spherical_jn(n,z)
#n: int, array_like
#Order of the Bessel function (n >= 0)

#z: complex or float, array_like
#Argument of the Bessel function
#Basically whatever is in the parentheses

#==========================================================================================================
#code below works currently

# Implementation of matplotlib function
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.close('all')
arr = np.arange(100).reshape((10, 10))
fig = plt.figure(figsize =(4, 4))

im = plt.imshow(arr,
                interpolation ="none",
                cmap ="plasma")

divider = make_axes_locatable(fig.gca())
cax = divider.append_axes("left","15 %",pad ="30 %")

plt.colorbar(im, cax = cax)

fig.suptitle('Stress', fontweight ="bold")

plt.show()

#=========================================================================================================
#code below is barely broken
# libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
 
# Get the data (csv file is hosted on the web)
url = 'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/volcano.csv'
data = pd.read_csv(url)
 
# Transform it to a long format
df=data.unstack().reset_index()
df.columns=["X","Y","Z"]
 
# And transform the old column name in something numeric
df['X']=pd.Categorical(df['X'])
df['X']=df['X'].cat.codes
 
# Make the plot
fig = plt.figure()
ax = fig.gca('3d')
ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
plt.show()
 
# to Add a color bar which maps values to colors.
fig = plt.figure()
ax = fig.gca('3d')
surf=ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar( surf, shrink=0.5, aspect=5)
plt.show()
 
# Rotate it
fig = plt.figure()
ax = fig.gca()
surf=ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
ax.view_init(30, 45)
plt.show()
 
# Other palette
fig = plt.figure()
ax = fig.gca()
ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.jet, linewidth=0.01)
plt.show()