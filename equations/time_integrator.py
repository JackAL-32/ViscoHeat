import numpy as __np
from equations.parameters import *

def cot(tha):
    return 1/__np.tan(tha)

def calculate_average_per_timestep(T):
    # Calculate the average value for each timestep along the last axis (axis=-1)
    average_per_timestep = __np.mean(T, axis=(0,1))
    return average_per_timestep

def get_dTdt(r, tha, dr, dtha, a, ptsInt, T, q, k, c, p):

    dTdt = __np.zeros(r.shape)

    # Roll in with the padding, truncate to correct dimensions
    Ti1j = __np.roll(T, -1, axis=0)[:-1, 1:-1]
    Tin1j = __np.roll(T, 1, axis=0)[:-1, 1:-1]
    Tij1 = __np.roll(T, -1, axis=1)[:-1, 1:-1]
    Tijn1 = __np.roll(T, 1, axis=1)[:-1, 1:-1]
    Ti2j = __np.roll(T, -2, axis=0)[:-1, 1:-1]
    Ti3j = __np.roll(T, -3, axis=0)[:-1, 1:-1]
    Tij2 = __np.roll(T, -2, axis=1)[:-1, 1:-1]
    Tij3 = __np.roll(T, -3, axis=1)[:-1, 1:-1]

    # Interior case:
    dTdt[ptsInt] =  (k[ptsInt]/(p[ptsInt]*c[ptsInt])) * (1/(r[ptsInt]**2)) * (
            (((r[ptsInt]/dr) + ((r[ptsInt]**2)/(dr**2))) * Ti1j[ptsInt])
        + ((((r[ptsInt]**2)/(dr**2)) - (r[ptsInt]/dr)) * Tin1j[ptsInt])
        + (((cot(tha[ptsInt])/(2*dtha)) + (1/(dtha**2))) * Tij1[ptsInt])
        + (((cot(tha[ptsInt])/(2*dtha)) + (1/(dtha**2))) * Tijn1[ptsInt])
        - ((((2*(r[ptsInt]**2))/(dr**2)) - (2/(dtha**2))) * T[:-1, 1:-1][ptsInt])) + (q[ptsInt]/(p[ptsInt]*c[ptsInt]))

    # Boundary cases:
    # tha max: Equation 6
    dTdt[:, -1] = (k[:,-1]/(p[:,-1]*c[:,-1])) * (1/(r[:,-1]**2)) * (
            ((((-2*(r[:,-1]**2))/(dr**2)) + (cot(tha[:,-1])/(2*dtha)) - (1/(dtha**2))) * T[:-1,1:-1][:,-1]) 
        + (((r[:,-1]/dr) + (r[:,-1]**2/(dr**2))) * Ti1j[:,-1])
        + (((r[:,-1]**2/(dr**2)) - (r[:,-1]/dr)) * Tin1j[:,-1])
        + (((1/(dtha**2)) - (cot(tha[:,-1])/(2*dtha))) * Tijn1[:,-1])) + (q[:,-1]/(p[:,-1]*c[:,-1]))

    # tha min: Equation 5
    dTdt[:, 0] = (k[:,0]/(p[:,0]*c[:,0])) * (1/(r[:,0]**2)) * (
        - ((((2*r[:,0]**2)/(dr**2)) + (cot(tha[:,0])/(2*dtha)) + (1/(dtha**2))) * T[:-1,1:-1][:,0])
        + (((r[:,0]/dr) + (r[:,0]**2/(dr**2))) * Ti1j[:,0])
        + (((r[:,0]**2/(dr**2)) - (r[:,0]/dr)) * Tin1j[:,0])
        + (((cot(tha[:,0])/(2*dtha)) + (1/(dtha**2))) * Tij1[:,0])) + (q[:,0]/(p[:,0]*c[:,0]))

    # r a: Equation 3 right
    dTdt[a, :] = (k[a,:]/(p[a,:]*c[a,:])) * (1/(r[a,:]**2)) * (
            ((((-3*r[a,:])/dr) + ((2*r[a,:])/(dr**2)) - ((3*cot(tha[a,:]))/(2*dtha)) + (2/(dtha**2))) * T[:-1, 1:-1][a,:])
        + ((((4*r[a,:])/dr) - ((5*r[a,:]**2)/(dr**2))) * Ti1j[a,:])
        + ((((4*r[a,:]**2)/(dr**2)) - (r[a,:]/dr)) * Ti2j[a,:])
        - (((r[a,:]**2)/(dr**2)) * Ti3j[a,:])
        + ((((2*cot(tha[a,:]))/dtha) - (5/(dtha**2))) * Tij1[a,:])
        + (((4/(dtha**2)) - (cot(tha[a,:])/(2*dtha))) * Tij2[a,:])
        - ((1/(dtha**2)) * Tij3[a,:])) + (q[a, :]/(p[a, :]*c[a,:]))
    
    # r a and tha min: 

    # r max: Should be handled as an internal point with Tinf padding
    
    # r min: Derivative estimate? Emailed Dr.Baker
    dTdt[0, :] = __np.average(dTdt[1,:])

    return dTdt

def get_T(r, tha, time, dr, dtha, dt, q1):

    # Binder Properties
    k1 = .27 # W/(m*K)
    p1 = 1030 # kg/m**3
    # c1 = k1/(p1*v1) Directly from Mares
    c1 = 2570 # J/(kg * K)

    # Crystal Properties
    k2 = .4184 # W/(m*K)
    p2 = 1910 # kg/m**3
    c2 = 1015 # J/(kg * K)

    # Append points to r
    # Need to be careful with input sanitization to get exactly step_size and exactly a in the linspace
    step_size = r[1] - r[0]
    num_additional_points = int(a/step_size)
    extra_points = __np.linspace(step_size, a-step_size, num_additional_points)
    r = __np.concatenate((extra_points, r))

    # Find the last index that is considered particle
    a_ind = __np.where(r == a)[0][0] # find r = a

    # Declare necessary arrays for Ralston's Method
    T = __np.full(shape=(r.size+1, tha.size+2, time.size), fill_value=T0 )
    T34 = __np.zeros((r.size+1, tha.size+2, time.size))
    dTdt = __np.zeros((r.size, tha.size, time.size))
    dTdt34 = __np.zeros((r.size, tha.size, time.size))

    # Tiling r, and tha for matix operations
    r_exp = __np.tile(r, (dTdt.shape[1],1)).T # Dupes r vector across columns
    tha_exp = __np.tile(tha, (dTdt.shape[0],1)) # Dupes tha vector across rows
    
    # Prepare material properties for particle (x2) and medium (x1)
    k_exp = __np.full((r.size, tha.size), k1)
    k_exp[:a_ind+1, :] = k2
    p_exp = __np.full((r.size, tha.size), p1)
    p_exp[:a_ind+1, :] = p2
    c_exp = __np.full((r.size, tha.size), c1)
    c_exp[:a_ind+1, :] = c2

    # Mask the internal points
    ptsInt = __np.full(shape=(r.size, tha.size), fill_value=True)
    ptsInt[:, -1] = False # tha max WRONG - depends how we handle the 2 pts that are both boundaries
    ptsInt[:, 0] = False  # tha min WRONG - depends how we handle the 2 pts that are both boundaries
    # r max should be treated as internal as long as we pad with Tinf
    ptsInt[0, :] = False  # r min
    ptsInt[a_ind, :] = False # r = a

    # Handle strangeness with the q matrix
    q = __np.transpose(q1[:-1,:])
    extra_points = __np.zeros((num_additional_points, tha.size))
    q = __np.concatenate((extra_points, q))

    for t in range(0, time.size-1):

        # Pad with reflection of itself at tha = pi,0 boundaries
        T[:,0,t] = T[:,1,t]
        T[:,-1,t] = T[:,-2,t]

        # Pad with Tinf at medium-air boundary (r max padding)
        T[-1,:,t] = 21

        # Calculate dTdt
        dTdt[:,:,t] = get_dTdt(r_exp, tha_exp, dr, dtha, a_ind, ptsInt, T[:,:,t], q, k_exp, c_exp, p_exp)

        # Estimate T34, include padding
        T34[:-1, 1:-1,t] = T[:-1, 1:-1,t] + dTdt[:, :, t] * dt * (3/4)
        T34[:,0,t] = T34[:,1,t]
        T34[:,-1,t] = T34[:,-2,t]

        # Calculate dTdt34
        dTdt34[:,:,t] = get_dTdt(r_exp, tha_exp, dr, dtha, a_ind, ptsInt, T34[:,:,t], q, k_exp, c_exp, p_exp)

        # Ralston's method to estimate T at next timestep
        T[:-1, 1:-1,t+1] = T[:-1, 1:-1,t] + (dt/3) * (dTdt[:, :, t] + (2*dTdt34[:, :, t]))

    # Strip the padding
    T = T[:-1, 1:-1,:]

    # Mirror the values over tha = 0
    T = __np.concatenate((__np.flip(T[1:, :], axis=0), T), axis=0) # Probably need to dupe first and last cols before mirror

    print(T[:,:,2])
    print(T[:,:,100])
    print(T[:,:,150])
    print(T[:,:,200])
    print(T[:,:,250])
    print(T[:,:,-1])
    print(T.shape)
    print(__np.amax(T))

    # Example usage:
    # Assuming T is your temperature array with shape (num_r_points, num_tha_points, num_time_steps)
    avg_per_timestep = calculate_average_per_timestep(T)

    # Print the average temperature for each timestep
    for i, avg_temp in enumerate(avg_per_timestep):
        print(f"Average temperature at timestep {i}: {avg_temp}")

    return T, r, tha
