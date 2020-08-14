'''
Author: ikaya
'''
# Import neccessary modules

import time
import math
import matplotlib.pyplot as plt
import pygame
import random
import numpy as np
from pygame.locals import *
from swarm import *
#from QLearningClass import *

# Simulation Parameters
number_of_particles = 51
number_of_axes      = 2
delta_t             = 0.1
t_final             = 1000

#def actions():
#    act = np.ndarray(shape=(21,2))
#    ctr = 0
#    for ii in range(0,21):
#        act[ii,0] = int(ctr)
#        act[ii,1] = (ii-10)/10
#        ctr = ctr + 1
#    return act

screen_size       = [3000,1800]
xtrg              = [1500,900]
list_min_distance = []
list_ave_distance = []
particles         = swarm(number_of_particles=50, screensize=screen_size,target_location=xtrg,display=True, CommRng=100)
#myagent          = agent(numberofstate=10,numberofaction=62)
clock             = pygame.time.Clock()
keepGoing         = True
iter              = 0
t                 = 0

while keepGoing:
    particles.run(keepGoing=True)
    remainder = iter % 100  
    if remainder == 0:
        print('time = ',t,' s ', ' target_pos = ', xtrg)   
    t = t + delta_t
    if t >= t_final:
        keepGoing = False
            
print('\n------------------------------------')
for key in particles.member.keys():
    print('Particle id       : %s' % (key))
    print('Particle role     : %s' % (particles.member[key]['role']))
    print('Particle color    : ', particles.color[particles.member[key]['role']])
    print('Particle target   : %s' % (particles.member[key]['target']))
    print('particle velocity : %s' % (particles.member[key]['velocity']))
    print('particle position : %s' % (particles.member[key]['position']))
    print('target position   : %s' % (particles.targetposition[particles.member[key]['target']]))
    print('weigths           : %s' % (particles.wght[particles.member[key]['role']]))
    print('particles in rng  : %s' % (particles.member[key]['PrtclsInRng']))
    print('------------------------------------')