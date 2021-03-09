import numpy as np
from numba import jit

def leapfrogPush(x,v,E,dt,qm):
    vnew = v + dt*qm*E
    xnew = x + dt*v
    return xnew,vnew

def gatherE(E1,E2,x):
    '''
    x is the normalized distance within a cell 0<x<1
    E1 is the E field at left cell boundary, E2 at right
    operates on all particles in a cell
    '''
    return E1*(1.0-x) + E2*x

def depositJ(xnew,xold,ncells):
    '''
    x is ncell long list of particle arrays
    '''

    dx = [xn-xo for xn,xo in zip(xnew,xold)]
    J = np.zeros(ncells)
    J[0] = .5*(dx[-1]).sum(axis=-1)+.5*dx[0].sum(axis=-1)
    J[ncells-1] = .5*(dx[ncells-2]).sum(axis=-1)+.5*dx[ncells-1].sum(axis=-1)
    for c in range(1,ncells-1):
        J[c] = .5*(dx[c-1]).sum(axis=-1)+.5*dx[c].sum(axis=-1)
    return J

def exchangeParticles(xnew,xold,v,ncells):
    linds = []
    rinds = []
    sinds = []
    xfluxold = []
    xfluxnew = []
    vflux = []
    for c in range(ncells):
        linds.append(np.nonzero(xnew[c]<0)[0])
        rinds.append(np.nonzero(xnew[c]>=1)[0])
        sinds.append(np.nonzero(np.logical_and(xnew[c]>=0,xnew[c]<1))[0])

        xfluxold.append(np.concatenate((xold[c][linds[c]],xold[c][rinds[c]])))
        xfluxnew.append(np.concatenate((np.zeros(len(linds[c])),np.ones(len(rinds[c])))))

    J = depositJ(xfluxnew,xfluxold,ncells)
    vflux.append(np.concatenate((v[-1][rinds[-1]],v[0][sinds[0]],v[1][linds[1]])))
    xfluxnew[0] = np.concatenate((xnew[-1][rinds[-1]]-1,xnew[0][sinds[0]],1+xnew[1][linds[1]]))
    xfluxold[0] = np.concatenate((np.zeros(len(rinds[-1])),xold[0][sinds[0]],np.ones(len(linds[1]))))
    xfluxnew[ncells-1] = np.concatenate((xnew[ncells-2][rinds[ncells-2]]-1,xnew[ncells-1][sinds[ncells-1]],1+xnew[0][linds[0]]))
    xfluxold[ncells-1] = np.concatenate((np.zeros(len(rinds[ncells-2])),xold[ncells-1][sinds[ncells-1]],np.ones(len(linds[0]))))
    for c in range(1,ncells-1):
        xfluxnew[c] = np.concatenate((xnew[c-1][rinds[c-1]]-1,xnew[c][sinds[c]],1+xnew[c+1][linds[c+1]]))
        vflux.append(np.concatenate((v[c-1][rinds[c-1]],v[c][sinds[c]],v[c+1][linds[c+1]])))
        xfluxold[c] = np.concatenate((np.zeros(len(rinds[c-1])),xold[c][sinds[c]],np.ones(len(linds[c+1]))))
    
    vflux.append(np.concatenate((v[ncells-2][rinds[ncells-2]],v[ncells-1][sinds[ncells-1]],v[0][linds[0]])))

    return xfluxnew,xfluxold,vflux,J

epsilon = 8.854e-11
def stepE(E,J,dt):
    return E-J*dt/epsilon

def step(x,xnew,v,E,ncells,dt,dx,q,m):
    for c in range(ncells-1):
        Einterp = gatherE(E[c]/dx,E[c+1]/dx,x[c])
        xnew[c],v[c] = leapfrogPush(x[c],v[c],Einterp,dt,q/m)
    Einterp = gatherE(E[ncells-1]/dx,E[ncells-1]/dx,x[ncells-1])
    xnew[ncells-1],v[ncells-1] = leapfrogPush(x[ncells-1],v[ncells-1],Einterp,dt,q/m)
    xnew,x,v,J = exchangeParticles(xnew,x,v,ncells)
    J = q*dx/dt*J + q*dx/dt*depositJ(xnew,x,ncells)
    E = stepE(E,J,dt)
    return xnew,v,J,E

qe = -1.602e-19
me =  9.11e-31

ncells = 100
dx = .001
n0 = 1e13
wpe = np.sqrt((qe*qe*n0/(me*epsilon)))
dt = .01/wpe

nppc = 100
nscale = n0/nppc
x = [np.linspace(0,1.0,nppc) for i in range(ncells)]
xnew = x.copy()
Te = 1.0
vte = np.sqrt(-qe*Te/me)
v = [np.random.randn(nppc)*vte/dx for i in range(ncells)]
E = np.zeros(ncells)
E[30:70] = 0.0
nsteps = 1000

densrec = np.zeros((nsteps,ncells))
Erec = np.zeros((nsteps,ncells))
Jrec = np.zeros((nsteps,ncells))
Temprec = np.zeros((nsteps,ncells))

for i in range(nsteps):
    x,v,J,E = step(x,xnew,v,E,ncells,dt,dx,qe*nscale,me*nscale)
    densrec[i] = [len(x[j]) for j in range(ncells)]
    Temprec[i] = [(-me/(2.0*qe)*dx*dx*vs*vs).mean() for vs in v]
    Erec[i] = E
    Jrec[i] = J