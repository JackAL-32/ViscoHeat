import numpy as __np
from equations.parameters import *

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

#For Boundary
def B(r,dt):
    return dt*gamma*(2*h/(r*k1) - (r**2 * h**2 / (k1**2)))*T0

def delta(i,j):
    if i==j:
        return 1
    return 0

#Matrix that is built from the diffusion equation that evolves the tempurature.
def H(r,tha,gamma,dt,dr,dtha):
    a = gamma
    from numpy import cos
    mat = __np.zeros( (r.size,)*4 )
    for i in range(r.size):
        for j in range(tha.size):
            for k in range(r.size):
                for m in range(r.size):
                    value = a*dt*(
                        ( 1/(r[i]**2 * dtha**2) - cos(tha[j])/(2 * r[i]**2 * __np.sin(tha[j]) * dtha) ) * delta(i,k)*delta(j-1,m) +
                        
                        ( 1/(dr**2) - 1/(r[i]*dr) ) * delta(i-1,k)*delta(j,m) -
                        
                        ( 2/(dr**2) + 2/(r[i]**2 * dtha**2) - 1/(a*dt) ) * delta(i,k)*delta(j,m) +
                        
                        ( 1/(dr**2) + 1/(r[i]*dr) ) * delta(i+1,k)*delta(j,m) +
                        
                        ( cos(tha[j])/(2 * r[i]**2 * __np.sin(tha[j]) * dtha) + 1/(r[i]**2 * dtha**2) ) * delta(i,k)*delta(j+1,m)
                        )
                    
                    if i == 0:
                        value += a*dt*( ( 1/(dr**2) - 1/(r[i]*dr) ) * delta(i+1,k)*delta(j,m) )
                    
                    if j == 0:
                        value += a*dt*( ( 1/(r[i]**2 * dtha**2) - cos(tha[j])/(2 * r[i]**2 * __np.sin(tha[j]) * dtha) ) * delta(i,k)*delta((tha.size-1),m) )
                        
                    if j == (tha.size-1):
                        value += a*dt*( ( cos(tha[j])/(2 * r[i]**2 * __np.sin(tha[j]) * dtha) + 1/(r[i]**2 * dtha**2) ) * delta(i,k)*delta(0,m) )
                    
                    mat[k,m,i,j] = value
    return mat

#This is similar to the matrix above, except it is for the edge boundry conditions
def Hb(r,tha,gamma,dt,dr,dtha):
    a = gamma
    from numpy import cos
    mat = __np.zeros( (r.size,)*4 )
    for i in range(r.size):
        for j in range(tha.size):
            for k in range(r.size):
                for m in range(r.size):
                    value = a*dt*(
                        ( 1/(r[i]**2 * dtha**2) - cos(tha[j])/(2 * r[i]**2 * __np.sin(tha[j]) * dtha) ) * delta(i,k)*delta(j-1,m) -
                        
                        ( 2/(r[i]**2 * dtha**2) + 2*h/(r[i]*k1) - (r[i]**2 * h**2 / (k1**2)) - 1/(a*dt) ) * delta(i,k)*delta(j,m) +
                        
                        ( cos(tha[j])/(2 * r[i]**2 * __np.sin(tha[j]) * dtha) + 1/(r[i]**2 * dtha**2) ) * delta(i,k)*delta(j+1,m)
                        
                        )
                    
                    if j == 0:
                        value += a*dt*( ( 1/(r[i]**2 * dtha**2) - cos(tha[j])/(2 * r[i]**2 * __np.sin(tha[j]) * dtha) ) * delta(i,k)*delta((tha.size-1),m) )
                        
                    if j == (tha.size-1):
                        value += a*dt*( ( cos(tha[j])/(2 * r[i]**2 * __np.sin(tha[j]) * dtha) + 1/(r[i]**2 * dtha**2) ) * delta(i,k)*delta(0,m) )
                    
                    mat[k,m,i,j] = value
    return mat

#Matrix that is built from the diffusion equation that evolves the tempurature.
def H_new(r,tha,gamma,dt,dr,dtha):
    a = gamma
    from numpy import cos
    H = __np.zeros( (r.size,tha.size,r.size,r.size) )
    ig,jg,kg,mg = __np.mgrid[0:r.size, 0:tha.size, 0:r.size, 0:r.size]
    #Delta inds then add to equation in 2 lines. a*dt at the end
    #For boundaries just set to the value 
    
    # Original H
    inds = __np.where((ig == kg) & (jg-1 == mg))
    H[inds] += 1/(r[ig][inds]**2 * dtha**2) - cos(tha[jg][inds])/(2 * r[ig][inds]**2 * __np.sin(tha[jg][inds]) * dtha)
    inds = __np.where((ig-1 == kg) & (jg == mg))
    H[inds] += 1/(dr**2) - 1/(r[ig][inds]*dr)
    inds = __np.where((ig == kg) & (jg == mg))
    H[inds] -= 2/(dr**2) + 2/(r[ig][inds]**2 * dtha**2) - 1/(a*dt) #-= is not a typo
    inds = __np.where((ig+1 == kg) & (jg == mg))
    H[inds] += 1/(dr**2) + 1/(r[ig][inds]*dr)
    inds = __np.where((ig == kg) & (jg+1 == mg))
    H[inds] += cos(tha[jg][inds])/(2 * r[ig][inds]**2 * __np.sin(tha[jg][inds]) * dtha) + 1/(r[ig][inds]**2 * dtha**2)

    # Original if-blocks in H
    inds = __np.where((ig == 0) & (kg == 1) & (jg == mg)) # ig + 1 == kg?
    H[inds] += 1/(dr**2) - 1/(r[ig][inds]*dr)
    inds = __np.where((jg == 0) & (ig == kg) & (tha.size-1 == mg))
    H[inds] += 1/(r[ig][inds]**2 * dtha**2) - cos(tha[jg][inds])/(2 * r[ig][inds]**2 * __np.sin(tha[jg][inds]) * dtha)
    inds = __np.where((jg == tha.size-1) & (mg == 0) & (ig == kg))
    H[inds] += cos(tha[jg][inds])/(2 * r[ig][inds]**2 * __np.sin(tha[jg][inds]) * dtha) + 1/(r[ig][inds]**2 * dtha**2)

    '''
    # Original Hb
    inds = __np.where((ig == kg) & (jg-1 == mg))
    inds = __np.where((ig == kg) & (jg == mg))
    inds = __np.where((ig == kg) & (jg+1 == mg))

    # Original if-blocks in Hb
    inds = __np.where((ig == kg) & ((tha.size-1) == mg))
    inds = __np.where((ig == kg) & (mg == 0))

    k/rho cp is alpha/gamma
    '''

    H *= a*dt

    return H


def H_final(r, tha, time, gamma, dt, dr, dtha):

    # np.roll()
    # gamma is alpha? k/rhocp

    from numpy import sin, cos
    H = __np.zeros( (r.size, tha.size, time.size) )
    H[:,:,0] = T0 # first timestep

    for t in range(time.size()-1):
        for i in range(r.size()):
            for j in range(tha.size()):
                # if boundary:

                # else:
                    term = 0
                    term -= ((2*r[i]**2)/(dr**2))-(2/(dtha**2))                 # contribution from H[i , j , t] if interior
                    term += (r[i]/dr)+((r[i]**2)/(dr**2))                       # contribution from H[i+1, j, t] if interior
                    term += ((r[i]**2)/(dr**2))-(r[i]/dr)                       # contribution from H[i-1, j, t] if interior
                    term += ((cos(tha[i])/sin(tha[i]))/(2*dtha))+(1/(dtha**2))  # contribution from H[i, j+1, t] if interior
                    term += ((cos(tha[j])/sin(tha[j]))/(2*dtha))+(1/(dtha**2))  # contribution from H[i, j-1, t] if interior

                    H[:,:,t+1] = term

#Sums the elements in the inputed matrix
# This is arr.sum()
def matsum(mat,x):
    add = 0
    for i in range(x.size):
        for j in range(x.size):
            add += mat[i,j]
    return add

#Creates new matrix T based on matrix H and the previous matrix T
def Tnew(mat1,mat2,mat3,q,x,dt):
    T = __np.zeros( (x.size,x.size) )
    
    for i in range(x.size):
        for j in range(x.size):

            # Hb stuff
            if i == (x.size-1):
                A = __np.multiply(mat3[:,:,i,j],mat2)
                add = matsum(A,x)
                T[i,j] = add + q[j,i] + B(x,dt)[i]
                
            # H stuff
            else:
                
                #if 0 == i and 0 <= j and j <= 2:
                #    print(i,j)
                #    print(mat1[:,:,i,j])
                    
                A = __np.multiply(mat1[:,:,i,j],mat2)
                add = matsum(A,x)
                T[i,j] = add + q[j,i]
    return T

#Creates a group of matrecies that is an array of T arrays where each matrix is a different time step
def Tnew_t(mat1,mat2,mat3,q,x,t,dt):
    T = __np.zeros( (x.size,x.size,t.size) )
    T[:,:,0] = mat2
    
    for i in range(1,t.size):
        T[:,:,i] = Tnew(mat1,T[:,:,i-1],mat3,q,x,dt)
    return T

#Fills in the gap in the graph
def FillGap(T,r,tha,t):
    T1 = __np.zeros( (tha1.size,r.size,t.size) )
    for i in range(t.size):
        T2 = __np.ndarray.tolist(__np.transpose(T[:,:,i]))
        T2.append(T2[0])
        T1[:,:,i] = __np.array(T2)

    return T1
