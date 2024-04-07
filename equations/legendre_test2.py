#Date: 3 March 2024
#Author: Paige Doughman
#legendre_test2.py: Legendre functions for Visco. Heat. project

# Adding to gitignore

#Need to evaluate every "x-value" for every n-value

import numpy as __np
import scipy.special as __scisp
import matplotlib.pyplot as plt

#Finding the Legendre Polynomial in terms of cos(theta)
def Pn_cos(tha,n):
    x = __np.cos(tha)

    leg = __scisp.legendre(n)
    val = leg(x)

    return val


#First derivative of Legendre Poly. 
def dPndTheta_1(n,tha):
    #When n = 0 derivative formula would be undefined (solved by hand)
    if n == 0:
        value = 1

    #When n > 0 use general derivative formula
    else:
        #Solving for n = i and n = i - 1 polynomial values
        val1 = Pn_cos(tha,n)
        val2 = Pn_cos(tha,n-1)

        #Solving for the value of the first derivative of the Legendre Polynomial
        value = ( -(n/2) * __np.sin(2*tha) * val1 ) + ( n * __np.sin(tha) * val2 )

    return value


#Second derivative of Legendre Poly.
def dPndTheta_2(n,tha):
    #When n = 0 derivative formula would be undefined (solved by hand)
    if n == 0:
        value = 0

    #When n = 1 derivative formula would be undefined (solved by hand)
    elif n == 1:
        value = -0.25 * (-__np.cos(tha) + 9*__np.cos(3*tha))

    else:
        #Solving for n = i and n = i - 1 polynomial values
        val1 = Pn_cos(tha,n)
        val2 = Pn_cos(tha,n-1)

        #Solving for the value of second derivative of the Legendre Polynomial
        value = ( (-n/2) * ( 2*__np.cos(2*tha) * val1 + __np.sin(2*tha) * dPndTheta_1(n,tha) )
                + n * ( __np.cos(tha) * val2 + __np.sin(tha) * dPndTheta_1(n-1,tha)) )

    return value


#Graphing functions to test outputs
#Number of points
n = 85

#Creating array for theta values
#Step has n+2 so zeros are not hit for tan values
step = __np.pi / (n+2)
start = step
stop = __np.pi - step
tha = __np.arange(start,stop,step)

#Initializing
dP1_tot_n = 0
dP2_tot_n = 0

dP1_tot = __np.zeros_like(tha)
dP2_tot = __np.zeros_like(tha)


#Need to make separate loops for first derivative and second derivative (for testing purposes)
#Loop for all theta values
for i in range(0,n):
    #Loop for all n-values for each theta value
    for j in range(0,n):
        dP1 = dPndTheta_1(j,tha[i])
        dP1_tot_n += dP1

        if j == n-1:
            dP1_tot[i] = dP1_tot_n

#Look for all theta values
for i in range(0,n):
    #Loop for all n-values for each theta value
    for j in range(0,n):
        dP2 = dPndTheta_2(j,tha[i])
        dP2_tot_n += dP2

        if j == n-1:
            dP2_tot[i] = dP2_tot_n
        


plt.plot(tha,dP1_tot)
plt.plot(tha,dP2_tot)
plt.show()