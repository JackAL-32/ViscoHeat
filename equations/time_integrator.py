import numpy as __np
from equations.parameters import *

def cot(tha):
    return 1/__np.tan(tha)

def get_dTdt(r, tha, dr, dtha, a_ind, ptsInt, T, q):

    dTdt = __np.zeros((r.size, tha.size))

    # Roll in with the padding, truncate garbage to correct dimensions
    Ti1j = __np.roll(T, -1, axis=0)[:-1, 1:-1]
    Tin1j = __np.roll(T, 1, axis=0)[:-1, 1:-1]
    Tij1 = __np.roll(T, -1, axis=1)[:-1, 1:-1]
    Tijn1 = __np.roll(T, 1, axis=1)[:-1, 1:-1]
    Ti2j = __np.roll(T, -2, axis=0)[:-1, 1:-1]
    Ti3j = __np.roll(T, -3, axis=0)[:-1, 1:-1]
    Tij2 = __np.roll(T, -2, axis=1)[:-1, 1:-1]
    Tij3 = __np.roll(T, -3, axis=1)[:-1, 1:-1]

    # Interior case:
    dTdt[ptsInt] =  gamma * (1/(r**2)) * (
            (((r/dr) + ((r**2)/dr**2)) * Ti1j)
        + ((((r**2)/dr**2) - (r/dr)) * Tin1j)
        + (((cot(tha)/(2*dtha)) + (1/(dtha**2))) * Tij1)
        + (((cot(tha)/(2*dtha)) + (1/(dtha**2))) * Tijn1)
        - ((((2*(r**2))/dr**2) - (2/(dtha**2))) * T[:-1, 1:-1])) # Needs trailing const

    # Boundary cases:
    # tha max: Equation 6
    dTdt[:, -1] = gamma * (1/(r**2)) * (
            ((((-2*(r**2))/(dr**2)) + (cot(tha)/(2*dtha)) - (1/(dtha**2))) * T[:-1, 1:-1]) 
        + (((r/dr) + (r**2/(dr**2))) * Ti1j)
        + (((r**2/(dr**2)) - (r/dr)) * Tin1j)
        + (((1/(dtha**2)) - (cot(tha)/(2*dtha))) * Tijn1)) # Needs trailing const

    # tha min: Equation 5
    dTdt[:, 0] = gamma * (1/(r**2)) * (
        - ((((2*r**2)/(dr**2)) + (cot(tha)/(2*dtha)) + (1/(dtha**2))) * T[:-1, 1:-1])
        + (((r/dr) + (r**2/(dr**2))) * Ti1j)
        + (((r**2/(dr**2)) - (r/dr)) * Tin1j)
        + (((cot(tha)/(2*dtha)) + (1/(dtha**2))) * Tij1)) # Needs trailing const

    # r a: Equation 3 right
    dTdt[a_ind, :] = gamma * (1/(r**2)) * (
            ((((-3*r)/dr) + ((2*r)/dr**2) - ((3*cot(tha))/(2*dtha)) + (2/(dtha**2))) * T[:-1, 1:-1])
        + ((((4*r)/dr) - ((5*r**2)/dr**2)) * Ti1j)
        + ((((4*r**2)/(dr**2)) - (r/dr)) * Ti2j)
        - (((r**2)/(dr**2)) * Ti3j)
        + ((((2*cot(tha))/dtha) - (5/(dtha**2))) * Tij1)
        + (((4/(dtha**2)) - (cot(tha)/(2*dtha))) * Tij2)
        - ((1/(dtha**2)) * Tij3)) # Needs trailing const

    # r max: Should be handled as an internal point with Tinf padding
    # r min: Derivative estimate? Emailed Dr.Baker

    return dTdt

def get_T(r, tha, time, gamma, dt, dr, dtha): # NEEDS "a" , "T0"

    # Append points to r
    # Need to be careful with input sanitization to get exactly 0 and exactly a in the linspace
    step_size = r[1] - r[0]
    num_additional_points = int(a/step_size)
    extra_points = __np.linspace(0, a-step_size, num_additional_points)
    r = __np.concatenate((extra_points, r))

    # Declare necessary arrays for Ralston's Method
    T = __np.full(shape=(r.size+1, tha.size+2, time.size), fill_value=T0 )
    dTdt = __np.zeros((r.size, tha.size, time.size))
    T34 = __np.zeros((r.size+1, tha.size+2, time.size))
    dTdt34 = __np.zeros((r.size, tha.size, time.size))

    # Tiling r and tha for matix operations
    r_exp = __np.tile(r, (dTdt.shape[1],1)).T # Dupes r vector across columns
    tha_exp = __np.tile(tha, (dTdt.shape[0],1)) # Dupes tha vector across rows

    # Mask the internal points
    ptsInt = __np.full(shape=(r.size, tha.size), fill_value=True)
    ptsInt[:, -1] = False  # tha max: might just be treated as internal
    ptsInt[:, 0] = False  # tha min
    ptsInt[-1, :] = False  # r max
    ptsInt[0, :] = False  # r min
    a_ind = __np.where(r == a)[0][0] # find r = a
    ptsInt[a_ind, :] = False # r = a

    for t in range(0, time.size-1):

        # Pad with reflection of itself at tha = pi,0 boundaries
        T[:,0,t] = T[:,1,t]
        T[:,-1,t] = T[:,-2,t]

        # Pad with Tinf at medium-air boundary (r max padding)
        T[-1,:,t] = 21

        # r min padding? Derivative estimate? Emailed Dr.Baker. May go in get_dTdt

        # Calculate dTdt
        dTdt[:,:,t] = get_dTdt(r_exp, tha_exp, dr, dtha, a_ind, ptsInt, T[:,:,t], q)

        # Estimate T34, include padding
        T34[:-1, 1:-1,t] = T[:-1, 1:-1,t] + dTdt * dt * (3/4)
        T34[:,0,t] = T34[:,1,t]
        T34[:,-1,t] = T34[:,-2,t]

        # Calculate dTdt34
        dTdt34[:,:,t] = get_dTdt(r_exp, tha_exp, dr, dtha, a_ind, ptsInt, T34[:,:,t], q)

        # Ralston's method to estimate T at next timestep
        T[:-1, 1:-1,t+1] = T[:-1, 1:-1,t] + (dt/3) * (dTdt + (2*dTdt34))

    # Strip the padding
    T = T[:-1, 1:-1,:]

    # Mirror the values over tha = 0
    T = __np.concatenate((__np.flip(T[1:, :], axis=0), T), axis=0)

    return T
