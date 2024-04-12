import numpy as __np

#!Global Variables
P1, a, f, N, v1, X1, v2, p, p2, k1, gamma, h, r, tha, tha1, P, B1, a1, u1, T0 = [0]*20

def init(lis):
    global P1, a, f, N, v1, X1, v2, p, p2, k1, gamma, h, T0
    P1, a, f, N, v1, X1, v2, p, p2, k1, gamma, h, T0 = lis
    __set_variables()

#----------------------------------------------------------------------------------------
# Parameter Equations
#========================================================================================
        
def __set_variables():
    #Xi2: Shear wave attenuation
    X2 = X1*v1/v2 #dB/Mhz/m

    #Ratio of densities
    global eta
    eta = p/p2

    #Tangents
    tan1 = X1*v1/(__np.pi*8.686*10**6)
    tan2 = X2*v2/(__np.pi*8.686*10**6)

    #mus
    u_e = v2**2*p
    u_v = u_e*tan2
    global u1
    u1 = u_e + u_v* 1j

    #omega: angular frequency
    global w
    w = 2*__np.pi*f

    #lambdas
    x_e = v1**2*p-2*u_e
    x_v = x_e * tan1
    x = x_e + x_v * 1j

    #
    global E1
    E1 = (u1*(3*x + 2*u1))/(x + u1)

    #
    global nu1
    nu1 = x/(2*(x + u1))

    #alpha_1 = Complex Longitudinal Wave Number
    global a1
    a1 = __np.sqrt(w**2*p/(x + 2 * u1))

    #Betta_1: Complex Shear Wave Number
    global B1
    B1 = __np.sqrt(p*w**2/u1)

    #Phi0: Wave Potential Amplitude
    global P
    P = P1*__np.exp(1j*a1*a*20)/a1

    #----------------------------------------------------------------------------------------
    #Spacial Arrays
    #========================================================================================

    #R array
    global r
    r = __np.linspace(a, 20*a, N)
    #i

    #Theta array
    global tha
    tha = __np.linspace(__np.pi/N, 2*__np.pi-__np.pi/N, N)
    #j

    #Filling in gap
    dtha = tha[1]-tha[0]
    global tha1
    tha1 = __np.ndarray.tolist(tha)
    tha1.append(tha1[-1]+dtha)
    tha1 = __np.array(tha1)
