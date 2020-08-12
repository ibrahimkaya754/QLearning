# Import Modules
import numpy as np
import pygame
import random
import math
from matplotlib import pyplot as plt

class swarm():
    def __init__(self, screensize, number_of_particles=6, display=False, dim=2):
        self.screensize   = screensize
        self.nop          = number_of_particles
        self.display      = display
        self._vel_max     = 20.0
        self._vel_min     = -20.0
        self.member       = {}
        self.iteration_no = 0
        self.dim          = 2 # 2 dimensional motion
        self.__coefficients_()
        
        if self.display:
            self.screen       = pygame.display.set_mode((self.screensize[0],self.screensize[1]))
            pygame.display.set_caption("Swarm System")
            self.screen.fill(self.WHITE)
                    
        for ii in range(self.nop):
            self.member[str(ii)] = {'center':(0,0)}
            #self.member[str(ii)]['best_value'] = 10**8
            
            self.member[str(ii)]['position'] = {str(jj): np.random.random()*(self.screensize[jj]-200)+100 \
                                                for jj in range(self.dim)}
            self.member[str(ii)]['velocity'] = {str(jj): np.random.random() for jj in range(self.dim)}
            self.member[str(ii)]['center']   = (int(np.round(self.member[str(ii)]['position']['0'])),
                                                     int(np.round(self.member[str(ii)]['position']['1'])))
            self.member[str(ii)]['algo']     = {'rulebased': True,
                                                'rl'       : False}                  
        self._distance()
    
    def __coefficients_(self):
        self.params = [19.087,77.570,74.741,49.385,50.461,31.121,715.096,
                       87.658,-8.527,9.441,-1.126,7.908,5.326,-3.060]
            
    def _distance(self):
        '''calculates the distances between the members of the swarm'''
        for ii in range(self.nop):
            self.member[str(ii)]['distance'] = {str(jj): {str(kk): self.member[str(ii)]['position'][str(kk)]-self.member[str(jj)]['position'][str(kk)] \
                for kk in range(self.dim)} for jj in range(self.nop)}
            
            self.member[str(ii)]['distance_sorted'] = {str(jj):sorted(np.abs(self.member[str(ii)]['distance'][str(kk)][str(jj)]) for kk in range(self.nop)) \
                for jj in range(self.dim)}
            
            self.member[str(ii)]['abs_distance'] = {str(jj): np.sqrt(math.fsum(self.member[str(ii)]['distance'][str(jj)][str(kk)]**2 for kk in \
                range(self.dim))) for jj in range(self.nop)}
            
            self.member[str(ii)]['abs_distance_sorted'] = [sorted(self.member[str(ii)]['abs_distance'].items(), key = lambda kv:(kv[1], kv[0]))[jj][1] \
                for jj in range(self.nop)]
            
            self.member[str(ii)]['closest_neighbours'] = [sorted(self.member[str(ii)]['abs_distance'].items(), key = lambda kv:(kv[1], kv[0]))[jj][0] \
                for jj in range(self.nop)]   

    ### Rule Based Algo Under Development 120820 ###
    def algo(self,particle,axes,position,velocity,closest_particles_abs,
             xtrg,wght,distance_to_target,vflock= 20,delta_t= 0.1,
             vel_min= 10,vel_max= 20.0,TimeConstant=0.01):
        
        r0,r1,r2,D,Cfrict,Cshill,R,d    = [self.params[ii] for ii in range(8)]
        apot, aslp, v1, a1, a6          = [0 for ii in range(5)]
    
        self.particle                   = particle
        self.axes                       = axes
        self.position                   = position
        self.velocity                   = velocity
        self.closest_particles_abs      = closest_particles_abs
        self.xtrg                       = xtrg
        self.wght                       = wght
        self.distance_to_target         = distance_to_target
        self.closestneighbour_number    = 3

        for nb in range(1,self.closestneighbour_number):
            if np.abs(self.position[self.closest_particles_abs[self.particle,nb],self.axes] - self.position[self.particle,self.axes]) < r0:
                apot = apot + D * np.minimum(r1,r0 - np.abs(self.position[self.closest_particles_abs[self.particle,nb],self.axes] - \
                       self.position[self.particle,self.axes])) * (self.position[self.closest_particles_abs[self.particle,nb],self.axes]  - \
                       self.position[self.particle,self.axes]) / np.abs(self.position[self.closest_particles_abs[self.particle,nb],self.axes] - \
                       self.position[self.particle,self.axes])
            else:
                apot = apot + 0

            aslp = aslp + Cfrict * (self.velocity[self.closest_particles_abs[self.particle,nb],self.axes] - \
                   self.velocity[self.particle,self.axes]) / (np.maximum(np.abs(self.position[self.closest_particles_abs[self.particle,nb],self.axes] - \
                   self.position[self.particle,self.axes])-(r0-r2),1))**2

            a1   = a1 + (self.position[self.closest_particles_abs[self.particle,nb],self.axes] - \
                   self.position[self.particle,self.axes]) / self.closestneighbour_number

            a6   = a6 + self.xtrg[self.axes] - self.position[self.particle,self.axes]

            v1   = v1 + (self.velocity[self.closest_particles_abs[self.particle,nb],self.axes] - \
                   self.velocity[self.particle,self.axes]) / self.closestneighbour_number

        vspp   = vflock * self.velocity[self.particle,self.axes] / np.abs(self.velocity[self.particle,self.axes])
        awall  = Cshill * self.smooth_transfer_function(np.abs(self.xtrg[self.axes] - \
                self.position[self.particle,self.axes]),R,d) * (vflock * ((self.xtrg[self.axes] - \
                self.position[self.particle,self.axes]) / np.abs(self.xtrg[self.axes] - \
                self.position[self.particle,self.axes])) - self.velocity[self.particle,self.axes])
        vtrack = 0
        
        self.delta_vel = ((self.wght[5]) * (v1+vspp+vtrack-self.velocity[self.particle,self.axes]) + \
                    (self.wght[0] * apot + self.wght[1] * aslp + self.wght[2] * awall + self.wght[3] * a1 + self.wght[4] * a6))
        
        self.velocity[self.particle,self.axes] = self.velocity[self.particle,self.axes] + self.delta_vel * delta_t
        self.velocity[self.particle,self.axes] = self.distance_to_target/100 * self.velocity[self.particle,self.axes]

        if self.velocity[self.particle,self.axes]>vel_max:
                self.velocity[self.particle,self.axes] = vel_max
        elif self.velocity[self.particle,self.axes]<-vel_max:
            self.velocity[self.particle,self.axes] = -vel_max

        if self.distance_to_target <= 100:
            self.velocity[self.particle,self.axes] = self.distance_to_target/200 * self.velocity[self.particle,self.axes]
        self.position_delta = self.velocity[self.particle,self.axes] * delta_t
    
    ######################### SMOOTH TRANSFER FUNCTION ########################################################################################
    def smooth_transfer_function(self,x,R,d):
        if x<=R:
            res = 0
        elif x>R and x<=R+d:
            res = (np.sin(np.pi/d*(x-R) - np.pi/2) + 1) / 2
        else:
            res = 1
        return res