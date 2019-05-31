#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:17:30 2019

@author: watkins35
"""
from __future__ import print_function,division
import numpy as np
import matplotlib.pyplot as py
from scipy.special import ellipk, ellipe


class BField:
    """ calculates the components of the magetic field of a given loop """
    def Br(self,I,R,Z,cR,rvec):
        # magnetic field in r direction
        r = rvec[0] - cR 
        z = rvec[1] - Z
        k = np.sqrt((4*R*r)/((R+r)**2+z**2))
    
        return I/(2*np.pi) * z/(r*((R+r)**2+z**2)**(1/2)) * (-ellipk(k)+(R**2+r**2+z**2)/((R-r)**2+z**2)*ellipe(k))

    def Bz(self,I,R,Z,cR,rvec):
        # magnetic field in z direction
        r = rvec[0] - cR
        z = rvec[1] - Z
        k = np.sqrt((4*R*r)/((R+r)**2+z**2))
    
        return I/(2*np.pi) * 1/( (R+r)**2+z**2) **(1/2) * (ellipk(k)+(R**2-r**2-z**2)/((R-r)**2+z**2) * ellipe(k))


class StreamFunction:
    """ calculates the stream function of a given loop """
    def Aphi(self,I,R,Z,cR,rvec):
        # vector potential. scalar
        r = rvec[0] - cR
        z = rvec[1] - Z
        k = np.sqrt((4*R*r)/((R+r)**2+z**2))
        
        return I/k * (R/r)**(1/2) * ((1-1/2*k**2)*ellipk(k)-ellipe(k))

    def Psi(self,I,R,Z,cR,rvec):
        # psi function, flow lines tangential to B field
        r = rvec[0] - cR
        return r * self.Aphi(I,R,Z,cR,rvec)


class MagneticLoop:
    """ We want to have a subroutine for the general case magnetic loop. 
    This class will calculate and visualize magnetic field of a circular current loop"""

    def __init__(self,spec,dim,mu=np.array([0.0, 1., 0.0])):
        """ for dipole, we have an optional parameter. This way the magnetic moment
        can be passed in only if we want to use this approximation """
        self.mu = mu # magnetic moment-- r,z,phi
        # for current loop. Unpack the specifications vector
        self.R = spec['R'] # radius of loop
        self.I = spec['I'] # current in loop
        self.cZ = spec['z'] # z coordinate of the center of the loop
        self.cR = spec['r'] # r coordinate of the center of the loop

        # unpack the dimensions of the grid
        self.nr = dim['nr'] # number of grid points in r direction
        self.nz = dim['nz'] # number of grid points in z direction

        # linspace takes beginning point, end point, and number of steps
        r = np.linspace(dim['rmin'],dim['rmax'],dim['nr'])
        z = np.linspace(dim['zmin'],dim['zmax'],dim['nz'])
        self.r,self.z = np.meshgrid(r,z)
        self.br,self.bz = np.meshgrid(r,z)
        self.psi = np.zeros_like(self.r)
        
        self.B = BField()
        self.PSI = StreamFunction()


    def calcMagField(self,dipole=False):
    	""" 
    	Calculate the magnetic field.
    	Here we have the dipole (far field) approximation as an option, with 
        the default set as the loop.
    	"""

    	for ir in range(self.nr):
            for jz in range(self.nz):
                # vector from the dipole to given r,z location
                rvec=np.array([self.r[jz,ir],self.z[jz,ir], 0.0])
        
                if dipole:
                    # this is the approximation
                    self.br[jz,ir]= (3*rvec[0]*np.dot(self.mu,rvec))/(np.linalg.norm(rvec))**5
                    self.bz[jz,ir]= (3*rvec[0]*np.dot(self.mu,rvec))/(np.linalg.norm(rvec))**5 - self.mu[1]/(np.linalg.norm(rvec))**3
    
                else:
                    # exact form
                    # magnetic field compenents
                    self.br[jz,ir] = self.B.Br(self.I,self.R,self.cZ,self.cR,rvec)
                    self.bz[jz,ir] = self.B.Bz(self.I,self.R,self.cZ,self.cR,rvec)
                # psi function
                self.psi[jz,ir] = self.PSI.Psi(self.I,self.R,self.cZ,self.cR,rvec)
                # current density
                self.J = Laplacian(self.psi,self.r,self.z)
                
                
def Laplacian(psi,r,z):
    """ 
    calculate the current distribution for a given stream function psi
    dell^2 =  1/r d/dr (r dt/ds) + d^2t/dz^2
    centered difference = f(x+h)-f(x-h) / 2h
    2nd centered difference = f(x+h)-2f(x)+f(x-h) / h^2 
    
    assumes that psi and r are nxn, or rows and columns have the same
    dimensions. Returns an array of size n-2 x n-2.
    """
    dr = abs(r[0,0]-r[0,1]) # step sizes
    dz = abs(z[0,0]-z[1,0])

    
    # derivative with respect to r
    dp1 = (psi[1:,:]-psi[:-1,:])/(2*dr)
    dp2 = r[1:,:] * dp1
    dp3 = (dp2[1:,:]-dp2[:-1,:])/(2*dr)
    dp4 = 1/r[1:-1,:]*dp3
    
    # derivative with respect to z
    d2psi = (psi[:,2:]-2*psi[:,1:-1]+psi[:,:-2])/dz**2
    
    return dp4[:,1:-1] + d2psi[1:-1,:]


def plot(loop,psi,J):
    py.close()
    py.subplot(1,2,1) # row, column, position
    py.contour(loop.r,loop.z,psi,30) # stream function
    #py.plot(loop.cR+loop.R,loop.cZ,'ro') # mark the center of the loop
    
    py.axis('equal')
    py.xlabel("R")
    py.ylabel("Z")
    py.title(r'$\psi_\phi$ field')

    
    # display J on a separate plot
    py.subplot(1,2,2)
    py.contour(loop.r[1:-1,1:-1],loop.z[1:-1,1:-1],J,30)
    #py.plot(loop.cR+loop.R,loop.cZ,'ro') # mark the center of the loop
    
    py.axis('equal')
    py.xlabel("R")
    py.ylabel("Z")
    py.title(r'$J_\phi$')
    
    py.colorbar()
    
    py.show()


def ResultingStream(loops):
    l = loops[0] # need the dimensions and each loop has the same
    
    psi = np.zeros_like(l.r) # store the overall stream function
    for loop in loops:
        psi += loop.psi # superposition of psi
    
    # find the current distribution
    J = Laplacian(psi,l.r,l.z)
    return psi,J


def main():
    # dipole moment (located at the origin r=z=0)
#    mu=np.array([0.0, 1., 0.0]) # r,z,phi
    

    # grid dimensions
    dim = {'rmin':0.2,
           'rmax':4.0,
           'zmin':-3.0,
           'zmax':3.0,
           'nr':50,
           'nz':80} # nr/nz is number of steps
    
    #accepts Radius, Current I, center (r,z) - called spec in the class
    loop = MagneticLoop({'R':2.5,'I':-1,'r':0,'z':0},dim)
    loop.calcMagField()
    
    loop2 = MagneticLoop({'R':1,'I':-1,'r':0,'z':2},dim)
    loop2.calcMagField()
    
    loop3 = MagneticLoop({'R':1,'I':-1,'r':0,'z':-2},dim)
    loop3.calcMagField()
    
    loop4 = MagneticLoop({'R':1.5,'I':-1,'r':0,'z':1.5},dim)
    loop4.calcMagField()
    
    loop5 = MagneticLoop({'R':1.5,'I':-1,'r':0,'z':-1.5},dim)
    loop5.calcMagField()
    
    loop6 = MagneticLoop({'R':2,'I':3,'r':0,'z':.75},dim)
    loop6.calcMagField()
    
    loop7 = MagneticLoop({'R':2,'I':-1,'r':0,'z':-.75},dim)
    loop7.calcMagField()
    
    loop8 = MagneticLoop({'R':.5,'I':-1,'r':0,'z':2.5},dim)
    loop8.calcMagField()
    
    loop9 = MagneticLoop({'R':.5,'I':5,'r':0,'z':-2.5},dim)
    loop9.calcMagField()
    
    
    plasma = MagneticLoop({'R':1,'I':5,'r':0,'z':0},dim)
    plasma.calcMagField()
    
    psi,J = ResultingStream([loop,loop2,loop3,loop4,loop5,
                             loop6,loop7,loop8,loop9,plasma])
    
    plot(loop,psi,J)


if __name__ == "__main__":
    main()