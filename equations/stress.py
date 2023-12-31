from numpy import cos as __cos
import numpy as __np
from equations.scipy_funcs import Pn, dPndtha, hn, jn, cot
from equations.An_Bn import An_Bn


#----------------------------------------------------------------------------------------
# This is the stress equation for R R
#========================================================================================
def sigrr_sum(r,tha):
    sigsum = 0
    from equations.parameters import B1, P, a1, u1
    for n in range(0, 85):
        An, Bn = An_Bn(n)
        sigsum += ( 
            Pn(n)(__cos(tha)) * 
            ( 
                P * (1j)**n * (2*n + 1) * 
                ( 
                    (n**2 - n - B1**2 * r**2/2) * ( jn(n,(a1*r)) ) + 2*a1*r*jn(n+1,(a1*r)) 
                ) + 
                An * 
                ( 
                    (n**2 - n - B1**2 * r**2/2) * ( hn(n,(a1*r)) ) + 2*a1*r*hn(n+1,(a1*r)) 
                ) - 
                Bn * 
                ( 
                    n*(n+1)*(n-1)*hn(n,(B1*r)) - n*(n+1)*B1*r*hn(n+1,(B1*r)) 
                ) 
            )
        )
    return sigsum

def get_sigrr(): #!These are repetitive. Need an elif chain giving a function
    #sigma_rr
    from equations.parameters import r, tha, u1
    sigrr = __np.zeros( (r.size,tha.size),dtype = complex )

    for i in range(0, r.size):
        for j in range(0, tha.size):
            sigrr[i,j] = 2*u1/r[i]**2 * sigrr_sum(r[i],tha[j])
            
    #Filling in gap
    sigrr = __np.ndarray.tolist(__np.transpose(sigrr))
    sigrr.append(sigrr[0])
    sigrr = __np.array(sigrr)

    return sigrr

#----------------------------------------------------------------------------------------
# This is the stress equation for Phi Phi
#========================================================================================
def sigphiphi_sum(r,tha):
    sigsum = 0
    from equations.parameters import B1, P, a1, u1
    for n in range(0, 85):
        An, Bn = An_Bn(n)
        sigsum += (
            ( 
                Pn(n)(__cos(tha)) * 
                ( 
                    P * (1j)**n * ( 2*n + 1 ) * 
                    ( 
                        (n + (a1**2) * (r**2) - (B1**2 * r**2/2)) * (jn(n,(a1*r))) - a1*r*jn(n+1,(a1*r)) 
                    ) +  
                    An * 
                    ( 
                        (n + (a1**2) * (r**2) - (B1**2) * (r**2)/2) * hn(n,(a1*r)) - a1*r*hn(n+1,(a1*r))
                    ) - 
                    Bn * n * (n + 1) * hn(n,(B1*r))
                ) + 
                cot(tha)*dPndtha(1,n,__cos(tha),tha) * 
                (
                    P * (1j)**n * (2*n + 1) * jn(n,(a1*r)) + 
                    An * hn(n,(a1*r)) - 
                    Bn *
                    (
                        (n+1)*hn(n,(B1*r)) - B1*r*hn(n+1,(B1*r))
                    )
                )
            )
        )
    return sigsum

def get_sigphiphi():
    #sigma_phiphi
    from equations.parameters import r, tha, u1
    sigphiphi = __np.zeros( (r.size,tha.size),dtype = complex )

    for i in range(0, r.size):
        for j in range(0, tha.size):
            sigphiphi[i,j] = 2*u1/r[i]**2 * sigphiphi_sum(r[i],tha[j])
            

    #Filling in gap
    sigphiphi = __np.ndarray.tolist(__np.transpose(sigphiphi))
    sigphiphi.append(sigphiphi[0])
    sigphiphi = __np.array(sigphiphi)

    return sigphiphi
    
#----------------------------------------------------------------------------------------
# This is the stress equation for Theta Theta
#========================================================================================
def sigthatha_sum(r,tha):
    sigsum = 0
    from equations.parameters import B1, P, a1, u1
    for n in range(0, 85):
        An, Bn = An_Bn(n)
        sigsum += (
            (
                Pn(n)(__cos(tha)) * 
                (
                    P * (1j)**n * ( 2*n + 1 ) * 
                    ( 
                        (n + (a1**2) * (r**2) - (B1**2 * r**2)/2) * jn(n,(a1*r)) - a1*r*jn(n+1,(a1*r)) 
                    ) + 
                    An * 
                    (
                        (n + (a1**2) * (r**2) - (B1**2 * r**2)/2) * hn(n,(a1*r)) - a1*r*hn(n+1,(a1*r))
                    ) - 
                    Bn * n * (n + 1) * hn(n,(B1*r))
                ) +
                dPndtha(2,n,__cos(tha),tha) * 
                (
                    P * (1j)**n * (2*n + 1) * jn(n,(a1 * r)) + An * hn(n,(a1*r)) - 
                    Bn * 
                    (
                        (n+1) * hn(n,(B1*r)) - B1*r*hn(n+1,(B1*r))
                    )
                )
            )
        )
    return sigsum

def get_sigthatha():
    #sigma_thetatheta
    from equations.parameters import r, tha, u1
    sigthatha = __np.zeros( (r.size,tha.size),dtype = complex )

    for i in range(0, r.size):
        for j in range(0, tha.size):
            sigthatha[i,j] = 2*u1/r[i]**2 * sigthatha_sum(r[i],tha[j])
            
    #Filling in gap
    sigthatha = __np.ndarray.tolist(__np.transpose(sigthatha))
    sigthatha.append(sigthatha[0])
    sigthatha = __np.array(sigthatha)

    return sigthatha

#----------------------------------------------------------------------------------------
# This is the stress equation for R Theta
#========================================================================================
def sigrtha_sum(r,tha):
    sigsum = 0
    from equations.parameters import B1, P, a1, u1
    for n in range(0, 85):
        An, Bn = An_Bn(n)
        sigsum += ( 
            dPndtha(1,n,__cos(tha),tha) * 
            (
                P * (1j)**n * (2*n + 1) * 
                ( 
                    (n-1) * jn(n,(a1*r)) - a1*r * jn(n+1,(a1*r)) 
                ) + 
                An * 
                 ( 
                     (n - 1) * hn(n,(a1*r)) - a1*r * hn(n+1,(a1*r))
                 ) -
                Bn * 
                     ( 
                         (n**2 - 1 - (B1**2 * r**2)/2) * hn(n,(B1*r)) + (B1*r) * hn(n+1,(B1*r)) 
                     )
                )
            )
    return sigsum

def get_sigrtha():
    #sigma_rtheta
    from equations.parameters import r, tha, u1
    sigrtha = __np.zeros( (r.size,tha.size),dtype = complex )

    for i in range(0, r.size):
        for j in range(0, tha.size):
            sigrtha[i,j] = 2*u1/r[i]**2 * sigrtha_sum(r[i],tha[j])
            
    #Filling in gap
    sigrtha = __np.ndarray.tolist(__np.transpose(sigrtha))
    sigrtha.append(sigrtha[0])
    sigrtha = __np.array(sigrtha)

    return sigrtha
