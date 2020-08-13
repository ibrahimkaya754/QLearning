# Import Modules
import numpy as np
import pygame
import random
import math
from matplotlib import pyplot as plt

class swarm():
    def __init__(self, screensize, target_location, number_of_particles=6, display=False, dim=2):
        self.screensize   = screensize
        self.nop          = number_of_particles
        self.display      = display
        self._vel_max     = 20.0
        self._vflock      = 20.0
        self.member       = {}
        self.iteration_no = 0
        self.dim          = 2 # 2 dimensional motion
        self.trgt_loc     = {str(ii): target_location[ii] for ii in range(self.dim)}
        self.__coefficients_()
        self.wght         = {'follower': [self.params[ii] for ii in range(8,14)],
                             'leader'  : np.array((0.0,0.0,0.0,0.0,10.0,0.0))}
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
            self.member[str(ii)]['deltavel'] = {str(jj): 0 for jj in range(self.dim)}
            self.member[str(ii)]['deltapos'] = {str(jj): 0 for jj in range(self.dim)}
            self.member[str(ii)]['center']   = (int(np.round(self.member[str(ii)]['position']['0'])),
                                                     int(np.round(self.member[str(ii)]['position']['1'])))
            self.member[str(ii)]['algo']     = {'rulebased': True,
                                                'rl'       : False}                
            self.member[str(ii)]['role']     = 'follower'
            self.member[str(ii)]['target']   = 'leader'
        self.leader = str(np.random.randint(self.nop))
        self.member[self.leader]['role']     = 'leader'
        self.member[self.leader]['target']   = 'target'
        self.targetposition                  = {'leader': self.member[self.leader]['position'],
                                                'target': self.trgt_loc}
        self._distance()

        for key in self.member.keys():
            print('particle id: ', key)
            print('role   : ',self.member[key]['role'])
            print('target : ',self.member[key]['target'])
            print('wghts  : ',self.wght[self.member[key]['role']])
            print('dist2wp: ',self.member[key]['distance2target'])
            print('-----------------------')
    
    def __coefficients_(self):
        self.params = [19.087,77.570,74.741,49.385,50.461,31.121,715.096,
                       87.658,-8.527,9.441,-1.126,7.908,5.326,-3.060]
            
    def _distance(self):
        '''calculates the distances between the members of the swarm'''
        for ii in range(self.nop):
            self.member[str(ii)]['distance']            = {str(jj): {str(kk): self.member[str(ii)]['position'][str(kk)]-\
                                                                    self.member[str(jj)]['position'][str(kk)] \
                                                                    for kk in range(self.dim)} \
                                                                    for jj in range(self.nop)}
            
            self.member[str(ii)]['distance_sorted']     = {str(jj): sorted(np.abs(self.member[str(ii)]['distance'][str(kk)][str(jj)])\
                                                                    for kk in range(self.nop)) for jj in range(self.dim)}
            
            self.member[str(ii)]['abs_distance']        = {str(jj): np.sqrt(math.fsum(self.member[str(ii)]['distance'][str(jj)][str(kk)]**2\
                                                                    for kk in range(self.dim))) for jj in range(self.nop)}
            
            self.member[str(ii)]['abs_distance_sorted'] = [sorted(self.member[str(ii)]['abs_distance'].items(), \
                                                           key = lambda kv:(kv[1], kv[0]))[jj][1] for jj in range(self.nop)]
            
            self.member[str(ii)]['closest_neighbours']  = [sorted(self.member[str(ii)]['abs_distance'].items(), \
                                                           key = lambda kv:(kv[1], kv[0]))[jj][0] for jj in range(self.nop)]  

            self.member[str(ii)]['distance2target']     = (lambda x: np.sqrt(x[0]**2+x[1]**2))\
                                                          ([self.member[str(ii)]['position'][str(jj)] -\
                                                          self.targetposition[self.member[str(ii)]['target']][str(jj)] \
                                                          for jj in range(self.dim)])
    
    ### Rule Based Algo Under Development 120820 ###
    def algo(self, delta_t= 0.1,numberofClosestMembers=3,TimeConstant=0.01):
        
        r0,r1,r2,D,Cfrict,Cshill,R,d    = [self.params[ii] for ii in range(0,8)]
        apot, aslp, v1, a1, a6          = [0 for ii in range(5)]
        
        self.delta_t                    = delta_t
        self.numberofClosestMembers     = numberofClosestMembers
        self.TimeConstant               = TimeConstant

        for axis in range(self.dim):
            for particle in self.member.keys():
                for neigbour in self.member[particle]['closest_neighbours'][1:]:
                    if np.abs(self.member[neigbour]['position'][str(axis)] - self.member[particle]['position'][str(axis)]) < r0:
                        apot = apot + D * np.minimum(r1,r0 - np.abs(self.member[neigbour]['position'][str(axis)] - \
                        self.member[particle]['position'][str(axis)])) * (self.member[neigbour]['position'][str(axis)]  - \
                        self.member[particle]['position'][str(axis)]) / np.abs(self.member[neigbour]['position'][str(axis)] - \
                        self.member[particle]['position'][str(axis)])
                    else:
                        apot = apot + 0

                    aslp = aslp + Cfrict * (self.member[neigbour]['velocity'][str(axis)] - \
                           self.member[particle]['velocity'][str(axis)]) / \
                           (np.maximum(np.abs(self.member[neigbour]['position'][str(axis)] - \
                           self.member[particle]['position'][str(axis)]) - (r0-r2),1))**2

                    a1   = a1 + (self.member[neigbour]['position'][str(axis)] - \
                           self.member[particle]['position'][str(axis)]) / self.numberofClosestMembers

                    a6   = a6 + self.targetposition[self.member[particle]['target']][axis] - self.member[particle]['position'][str(axis)]

                    v1   = v1 + (self.member[neigbour]['velocity'][str(axis)] - \
                           self.member[particle]['velocity'][str(axis)]) / self.numberofClosestMembers

                vtrack = 0
                vspp   = self._vflock * self.member[particle]['velocity'][str(axis)] / np.abs(self.member[particle]['velocity'][str(axis)])
                awall  = Cshill * self.smooth_transfer_function(np.abs(self.self.targetposition[self.member[particle]['target']][axis] - \
                         self.member[particle]['position'][str(axis)]),R,d) * (self._vflock * ((self.self.targetposition[self.member[particle]['target']][axis] - \
                         self.member[particle]['position'][str(axis)]) / np.abs(self.self.targetposition[self.member[particle]['target']][axis] - \
                         self.member[particle]['position'][str(axis)])) - self.member[particle]['velocity'][str(axis)])
                
                self.member[particle]['deltavel'][str(axis)] = self.wght[self.member[particle]['role']][5] * (v1+vspp+vtrack-self.member[particle]['velocity'][str(axis)]) + \
                                                               self.wght[self.member[particle]['role']][0] * apot + self.wght[self.member[particle]['role']][1] * aslp + \
                                                               self.wght[self.member[particle]['role']][2] * awall + \
                                                               self.wght[self.member[particle]['role']][3] * a1 + self.wght[self.member[particle]['role']][4] * a6

                self.member[particle]['velocity'][str(axis)] = self.member[particle]['velocity'][str(axis)] + \
                                                               self.member[particle]['delta_vel'][str(axis)] * self.delta_t
                self.member[particle]['velocity'][str(axis)] = self.member[particle]['distance2target']/100 * self.member[particle]['velocity'][str(axis)]

                if self.member[particle]['velocity'][str(axis)] > self._vel_max:
                    self.member[particle]['velocity'][str(axis)] = self._vel_max
                elif self.member[particle]['velocity'][str(axis)] < -self._vel_max:
                    self.member[particle]['velocity'][str(axis)] = -self._vel_max

                if self.member[particle]['distance2target'] <= 100:
                    self.member[particle]['velocity'][str(axis)] = self.member[particle]['distance2target']/200 *self.member[particle]['velocity'][str(axis)]
                self.member[particle]['delta_vel'][str(axis)] = self.member[particle]['velocity'][str(axis)] * self.delta_t
       
    def run(self):
        self.algo()
        self._distance()

    
    ######################### SMOOTH TRANSFER FUNCTION ########################################################################################
    def smooth_transfer_function(self,x,R,d):
        if x<=R:
            res = 0
        elif x>R and x<=R+d:
            res = (np.sin(np.pi/d*(x-R) - np.pi/2) + 1) / 2
        else:
            res = 1
        return res