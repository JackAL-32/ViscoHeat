import numpy as __np
from equations.parameters import *
from numpy import sin, cos

def get_q1(sigrr, sigphiphi, sigthatha, sigrtha):
    from numpy import angle, abs
    # This is the equation for Delta R R
    drr = (angle(E1)) + (angle(sigrr)) - (angle((sigrr) - ((nu1) * ((sigthatha) + (sigphiphi)))))
    
    # This is the equation for Delta theta theta
    dthatha = (angle(E1)) + (angle(sigthatha)) - (angle((sigthatha) - ((nu1) * ((sigrr) + (sigphiphi)))))
    
    # This is the equation for Delta phi phi
    dphiphi = (angle(E1)) + (angle(sigphiphi)) - (angle((sigphiphi) - ((nu1) * ((sigrr) + (sigthatha)))))
    
    # This is the equation for Delta R theta
    drtha = (angle(E1)) - (angle(1 + nu1))
    
    # This is the equation for Xi R R
    xirr = (__np.sin(drr)) * (abs(sigrr)) * (abs((sigrr) - ((nu1) * ((sigthatha) + (sigphiphi)))))
    
    # This is the equation for Xi theta theta
    xithatha = (__np.sin(dthatha)) * (abs(sigthatha)) * (abs((sigthatha) - ((nu1) * ((sigrr) + (sigphiphi)))))
    
    # This is the equation for Xi phi phi
    xiphiphi = (__np.sin(dphiphi)) * (abs(sigphiphi)) * (abs((sigphiphi) - ((nu1) * ((sigrr) + (sigthatha)))))
    
    # This is the equation for Xi R theta
    xirtha = (__np.sin(drtha)) * (abs(1 + nu1)) * ((abs(sigrtha)) * (abs(sigrtha)))
    
    # This is the equation for q1
    q1 = ((w)/(2 * (abs(E1)))) * (xirr + xithatha + xiphiphi + xirtha)

    return q1

#----------------------------------------------------------------------------------------
# Parameter Equations for Tempurature Gradient
#========================================================================================

def get_T(r, tha, time, gamma, dt, dr, dtha):
    T = __np.fill( shape=(r.size, tha.size, time.size), fill_value=T0 )
    dTdt = __np.zeros( (r.size, tha.size, time.size) )

    for t in range(time.size()):
        for i in range(r.size()):
            for j in range(tha.size()):
                if i == 0: dTdt[i,j,t] = 0 #Eqn1
                if i == a: dTdt[i,j,t] = 0 #Eqn3
                if j == (tha.size()-1): term = 0 #Eqn6
                if j == 0: term = 0 #Eqn5
                else: term = 0 #Eqn0


                




        # r = 0, Eqn 1
        # r = a, Eqn 3 L or R (particle/binder boundary)
        # tha = last tha, Eqn 6
        # tha = first tha, Eqn 5
