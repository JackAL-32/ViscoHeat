import numpy as __np
from equations.scipy_funcs import hn, jn

#----------------------------------------------------------------------------------------
#Coefficient for the Longitudinal and Shear Wave Potentials respectivly
#========================================================================================

#A_n and B_n (this is a test comment)
def An_Bn(n):
    from equations.parameters import a1, a, B1, P, eta
    if n == 1:
        e11 = (1-eta)*hn(1,a1*a) - a1*a*hn(2,a1*a)
        e21 = (1-eta)*hn(1,a1*a)
        e12 = -2*(1-eta)*hn(1,B1*a)
        e22 = -2*(1-eta)*hn(1,B1*a) + B1*a*hn(2,B1*a)
        e1  = -3*1j*((1-eta)*jn(1,a1*a) - a1*a*jn(2,a1*a))
        e2  = -3*1j*(1-eta)*jn(1,a1*a)
        
        mat = __np.array([[e11,e12],[e21,e22]])
        av = P*__np.array([[e1],[e2]])
        x = __np.linalg.solve(mat,av)
    
        An = x[0][0]
        Bn = x[1][0]
    
    else:
        E11 = n*hn(n,a1*a) - a1*a*hn(n+1,a1*a)
        E12 = -n*(n+1)*hn(n,B1*a)
        E21 = hn(n,a1*a)
        E22 = -(n+1)*hn(n,B1*a) + B1*a*hn(n+1,B1*a)
        E1  = -1j**n*(2*n+1)*(n*jn(n,a1*a) - a1*a*jn(n+1,a1*a))
        E2  = -1j**n*(2*n+1)*jn(n,a1*a)
    
        mat = __np.array([[E11,E12],[E21,E22]])
        av = P*__np.array([[E1],[E2]])
        x = __np.linalg.solve(mat,av)
    
        An = x[0][0]
        Bn = x[1][0]
    return An, Bn
