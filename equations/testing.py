#Author: Paige Doughman
#Date: 4 April 2024
#testing.py.py: Testing derivative for Legendre Poly.

import scipy.special as __scisp
import numpy as __np

def cot(tha):
    if tha % __np.pi == 0: #if tha % __np.pi == 0: is probably better
        return 0
    else:
        return 1/__np.tan(tha)

#Finding the Legendre Polynomial in terms of cos(theta)
def Pn_cos(tha,n):
    x = __np.cos(tha)

    leg = __scisp.legendre(n)
    val = leg(x)

    return val

tha = __np.pi / 6
n = 1

if n == 0:
    value = 0

elif n == 1:
    value = -__np.sin(tha)

#When n > 1 use general derivative formula
else:
    #Solving for n = i and n = i - 1 polynomial values
    val1 = Pn_cos(tha,n)
    val2 = Pn_cos(tha,n-1)

    #Solving for the value of the first derivative of the Legendre Polynomial
    value = (n * cot(tha) * val1) - (n * (1/__np.sin(tha)) * val2)

print(value)