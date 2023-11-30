import scipy.special as __scisp
import numpy as __np
#----------------------------------------------------------------------------------------
#Imported scipy functions
#========================================================================================

#Cotangent function 
def cot(tha):
    if tha == 0 or tha == __np.pi: #if tha % __np.pi == 0: is probably better
        return 0
    else:
        return 1/__np.tan(tha)

#P_n: Legendre Polynomials
def Pn(n):
    return __scisp.legendre(n)
#n: int
#Degree of the polynomial.


def Pndev1(n,x):
    return -(n+1)*(x*Pn(n)(x)-Pn(n+1)(x))/(x**2 - 1)


#d(P_n)/dtheta: derivatives of Legendre Polynomials
def dPndtha(m,n,x,tha):
    if n == 0:
        return 0
        
    elif n == 1:
        if m == 1:
            return -__np.sin(tha)*Pndev1(n,x)
        elif m == 2:
            return -__scisp.lpmv(1,n,x)*(cot(tha))
        else:
            raise TypeError("Invalid input. The first term should be a 1 or a 2")
        
    elif n > 1:
        if m == 1:
            return -__np.sin(tha)*Pndev1(n,x)
        elif m == 2:
            return __scisp.lpmv(2,n,x) - __scisp.lpmv(1,n,x)*(cot(tha))
        else:
            raise TypeError("Invalid input. The first term should be a 1 or a 2")

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
