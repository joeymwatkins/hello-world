#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 22:56:47 2019

@author: Joey
"""
import numpy as np
import matplotlib.pyplot as py
from mpl_toolkits.mplot3d import Axes3D

class Poisson:
    
    def __init__(self,xDomain,yDomain,Nx,Ny):
        
        
        # generate mesh grid right away
        x,self.dx = np.linspace(xDomain[0],xDomain[1],Nx,retstep = True)
        y,self.dy = np.linspace(yDomain[0],yDomain[1],Ny,retstep = True)
        
        # make the grid
        self.X, self.Y = np.meshgrid(x,y)
        self.V = np.zeros_like(self.X)
        
        # step sizes
        self.Nx = Nx
        self.Ny = Ny
        
        
    def optimalOmega(self):
        """ this time we will also check what the optimum value is """
        R = (self.dy**2*np.cos(np.pi/self.Nx) + self.dx**2*np.cos(np.pi/self.Ny))/(self.dx**2 + self.dy**2)
        self.omega = 2/(1+np.sqrt(1-R**2))
        
    
    def setBoundaryConditions(self):
        # regular function bc V_{j,k}
        self.V[:,-1] = 1 # last row is V(2,y) = 2
        self.V[:,0] = -1 # first row is V(-2,y) = -2

    def makeMask(self):
        """ this function will create a mask for the potential that will
        act as holes allowing some fields to leak in. The set up here is to 
        enable us to change the number of grid points and still have the 
        holes at the same general locations """
        # need to create a mask
        mask = np.ones([self.Nx,self.Ny])
        # set the elements we don't want to change to zero
        x = np.arange(2*self.Nx//6,5*self.Nx//6,4) # list of the x coordinates we want
        zeros = np.ones([2*len(x),2]) # initialize 2d array, each coordinate
        
        for i in range(len(x)):
            # replace each element with coordinate we want
            zeros[i,:] = [x[i],1*self.Ny//4] # first row of holes
            zeros[i+len(x),:] = [x[i],3*self.Ny//4] # second row of holes
        # this resulted in the list of coordinates for the holes
        zeros = zeros.astype(int) # convert to integers
        
        for i,j in zeros:
            # now we go through the mask and poke holes in it
            mask[i,j] = 0
        
        return mask

    
    def relax(self,movie = True):
        """ over relax because we go one step farther than given.
        We have the option of watching a movie as the error lessons and we 
        converge on the solution.
        Or, we can set movie to false and only look at a plot of the
        final state."""
        
        fig = py.figure(1)
        counter = 0 # for the movie
        
        # create the mask
        mask = self.makeMask()
            
        shouldContinue = True
        while shouldContinue and counter < 1000:
            
            shouldContinue = False # to ensure accuracy of every point
            
            # loop over both dimensions
            for j in range(1,self.Nx-1):
                for k in range(1,self.Ny-1):
                    # this possions equations, solved for v_{j,k} (eqn21.3)
                    # rho and epsilon are gone because we are doing a 
                    # charge free problem
                    rhs = ((self.V[j+1,k]+self.V[j-1,k])/self.dx**2 +
                           (self.V[j,k+1]+self.V[j,k-1])/self.dy**2)/\
                           (2/self.dx**2 + 2/self.dy**2)
                    error = abs(self.V[j,k] - rhs)
                    # eq 21.15. This includes the overrelaxation part
                    self.V[j,k] = self.omega * rhs + (1-self.omega)*self.V[j,k]
                    self.V[j,k] *= mask[j,k] # maintain the holes in the potential
                    
                    if error > 1e-4:
                        # finish going over the whole grid, then iterate again
                        # we need every point to be good
                        shouldContinue = True
                    
            # edges held at zero
            self.V[0,:] = 0
            self.V[-1,:] = 0
            
            
            # wait until the whole cell updates
            if counter % 10 == 0 and movie:
                self.plotMovie(fig)
            counter += 1
        
        if movie:
            py.close()
            
        if not movie:
            self.plotFinal(fig)
    
    def plotMovie(self,fig):
        """ called in the animate function.
        lists the settings for our movie """
        ax = fig.gca(projection = '3d')
        ax.plot_surface(self.X,self.Y,self.V,cmap = py.cm.ocean) # set the colorscheme
        ax.set_zlim(-1,1)
        ax.view_init(elev = 30, azim = 230) # viewing angle
        py.draw()
        py.pause(1e-5)
        py.clf()
    
    
    def plotFinal(self,fig):
        """ skips the error demonstration and shows us the final configuration.
        the advantage is that we can rotate and interact with the plot after. """
        ax = fig.gca(projection = '3d')
        ax.plot_surface(self.X,self.Y,self.V,cmap = py.cm.ocean) # set the colorscheme
        ax.set_zlim(-1,1)
        ax.view_init(elev = 30, azim = 230) # viewing angle
        py.show()


def main():
    # no free charge, but now we have some holes
    xBound = [-0.1,0.1]
    yBound = [0,0.4]
    Nx = 40 # 40 is nicer than any of the other choices.
    Ny = 40 # the others take longer
    
    # make the thing
    p = Poisson(xBound,yBound,Nx,Ny)
    p.optimalOmega()
    p.setBoundaryConditions()
    p.relax(movie = False)

        
            

if __name__ == "__main__":
    main()