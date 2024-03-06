import numpy as __np
from equations.parameters import *
from numpy import sin, cos

def get_T(r, tha, time, gamma, dt, dr, dtha):
    
    T = __np.fill( shape=(r.size+2, tha.size+2, time.size), fill_value=T0 ) # include necessary padding area, and fill with T0.
    dTdt = __np.zeros((r.size, tha.size, time.size)) 
    T34 = __np.zeros((r.size+2, tha.size+2, time.size))
    dTdt34 = __np.zeros((r.size, tha.size, time.size))

    # Initialize first timestep
    T[r.size+1, :, 1] = 21; # Tinf for rmax boundary
    # At initial timestep all other boundary conds are T0.

    for t in range(time.size()):
        # Interior cases:
        ptsInt = __np.where()
        # Boundary cases:
        # tha max

        # tha min
        # r max
        # r min
        # r a
        







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
