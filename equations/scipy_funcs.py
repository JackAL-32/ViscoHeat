import scipy.special as __scisp
import numpy as __np
#----------------------------------------------------------------------------------------
#Imported scipy functions
#========================================================================================

#Cotangent function 
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


#First derivative of Legendre Poly. 
def dPndTheta_1(n,tha):
    #When n = 0 derivative formula would be undefined (solved by hand)
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

    return value


#Second derivative of Legendre Poly.
def dPndTheta_2(n,tha):
    #When n = 0 derivative formula would be undefined (solved by hand)
    if n == 0:
        value = 0

    #When n = 1 derivative formula would be undefined (solved by hand)
    elif n == 1:
        value = -__np.cos(tha)

    else:
        #Solving for n = i and n = i - 1 polynomial values
        val1 = Pn_cos(tha,n)
        val2 = Pn_cos(tha,n-1)

        #Solving for the value of second derivative of the Legendre Polynomial
        value = n * ( ( (-(1/__np.sin(tha))**2 * val1) + (cot(tha) * dPndTheta_1(n,tha)) ) +
                     ( ((cot(tha)/__np.sin(tha)) * val2) - ((1/__np.sin(tha)) * dPndTheta_1(n-1,tha)) ) )

    return value

#h_n: Spherical Hankel Function of the first kind
def hn(n,z):
    return __np.sqrt( __np.pi/(2*z) ) * __scisp.hankel1(n+0.5,z)
#n: array_like
#Order (float)

#z: array_like
#Argument (float or complex)
#Basically whatever is in the parentheses


#j_n: Spherical Bessel Function of the first kind
def jn(n,z):
    return __np.sqrt( __np.pi/(2*z) ) * __scisp.jv(n+0.5,z)
    #return spherical_jn(n,z)
#n: int, array_like
#Order of the Bessel function (n >= 0)

#z: complex or float, array_like
#Argument of the Bessel function
#Basically whatever is in the parentheses
