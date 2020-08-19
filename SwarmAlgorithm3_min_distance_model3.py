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
from QLearningClass import *

# Simulation Parameters
number_of_particles = 51
number_of_axes      = 2
delta_t             = 0.1
t_final             = 1000

def actions():
   act = np.ndarray(shape=(21,2))
   ctr = 0
   for ii in range(0,21):
       act[ii,0] = int(ctr)
       act[ii,1] = (ii-10)/10
       ctr = ctr + 1
   return act

screen_size       = [1000,600]
xtrg              = [500,300]
list_min_distance = []
list_ave_distance = []
particles         = swarm(number_of_particles=50,screensize=screen_size,target_location=xtrg,
                          display=True,CommRng=100)
rlagent           = [key for key in particles.member.keys() if particles.member[key]['role']=='rlagent'][0]
leader            = particles.leader
numberofneighbour = 5
numberofleader    = 1
clock             = pygame.time.Clock()
keepGoing         = True
iter , t          = 0 , 0

### The multiplayer 2 below is for 'position' and 'velocity' ###
print('----------------------------------------------------------------------------')
print('There will be %s states, %s for relative velocity, %s for relative position' % \
      (particles.dim*(numberofneighbour+numberofleader)*2,\
      particles.dim*(numberofneighbour+numberofleader),\
      particles.dim*(numberofneighbour+numberofleader)))
print('----------------------------------------------------------------------------')
### Some states are from the closest leader ###
print('%s of the states are gathered from the closest leader of the swarm' % (numberofleader*particles.dim*2))
print('----------------------------------------------------------------------------')
action            = actions()  
myagent           = agent(numberofstate=particles.dim*(numberofneighbour+numberofleader)*2,numberofaction=len(action))
#################################################################
### States are appended to the "states list" ###
def stateappend(state):
    state = []
    for relpos,relvel in zip(list(particles.member[rlagent]['relative_position'].values())[0:numberofneighbour],\
                             list(particles.member[rlagent]['relative_velocity'].values())[0:numberofneighbour]):
        for pos,vel in zip(relpos.values(),relvel.values()):
            state.append(pos)
            state.append(vel)

    for relpos,relvel in zip(list(particles.member[rlagent]['distance2leader'].values()),\
        list(particles.member[rlagent]['velocity2leader'].values())):
        state.append(relpos)
        state.append(relvel)
    state = np.array(state)
    return state
myagent.state = stateappend(myagent.state)
###############################################

while keepGoing:
    particles.rulebasedalgo()

    qval = myagent.model['model1']['model_network'].predict(myagent.state.reshape(1,myagent.numberofstate))
    for dim in range(particles.dim):
        if random.random() < myagent.epsilon:
            myagent.action[0,dim] = np.random.randint(0,myagent.numberofaction)
        else:
            myagent.action[0,dim] = int(np.argmax(qval[dim]))
        particles.member[rlagent]['deltavel'][str(dim)] = action[myagent.action[0,dim]][1]
    
    particles.update(keepGoing=keepGoing)
    myagent.newstate = stateappend(myagent.newstate)
    myagent.reward = 0
    myagent.done   = False
    myagent.replay_list()
    myagent.state  = myagent.newstate
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
