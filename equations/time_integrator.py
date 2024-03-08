import numpy as __np
from equations.parameters import *
from numpy import sin, cos

def get_T(r, tha, time, gamma, dt, dr, dtha): # NEEDS "a"
    
    T = __np.full(shape=(r.size+2, tha.size+2, time.size), fill_value=T0 ) # Maybe just +1 now???
    dTdt = __np.zeros((r.size, tha.size, time.size)) 
    T34 = __np.zeros((r.size+2, tha.size+2, time.size)) # Maybe just +1 now???
    dTdt34 = __np.zeros((r.size, tha.size, time.size))

    r_exp = __np.tile(r, (T.shape[1],1)).T # Dupes r vector across columns
    tha_exp = __np.tile(tha, (T.shape[0],1)) # Dupes tha vector across rows

    # Mask the internal points
    ptsInt = __np.full(shape=(r.size, tha.size), fill_value=True)
    ptsInt[:, tha.size-1] = False # tha max
    ptsInt[:, 0] = False # tha min
    ptsInt[r.size-1, :] = False # r max
    ptsInt[0, :] = False # r min
    ptsInt[a, :] = False # r a

    # Initialize first timestep
    T[r.size+1, :, 1] = 21; # Tinf for rmax boundary
    # Everything else already T0

    # Every step except for 1st
    for t in range(1, time.size()):

        # Interior case:
        Tu = __np.roll(T,  1, axis=0)
        Td = __np.roll(T, -1, axis=0)
        Tr = __np.roll(T, -1, axis=1)
        Tl = __np.roll(T,  1, axis=1)
        dTdt[ptsInt, t] =  gamma * (1/r_exp)

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
