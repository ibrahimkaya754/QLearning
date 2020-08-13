# Import neccessary modules
import time
import math
import matplotlib.pyplot as plt
import pygame
import random
import numpy as np
from pygame.locals import *
#from PygameModule import *
#from Swarm_Algorithm_RuleBased import *
from swarm import *
#from QLearningClass import *

# Simulation Parameters
number_of_particles = 51
number_of_axes      = 2
delta_t             = 0.1
t_final             = 1000

#myagent = agent(numberofstate=10,numberofaction=62)

#def actions():
#    act = np.ndarray(shape=(21,2))
#    ctr = 0
#    for ii in range(0,21):
#        act[ii,0] = int(ctr)
#        act[ii,1] = (ii-10)/10
#        ctr = ctr + 1
#    return act


screen_size = [3000, 1300]
xtrg              = [2400,200]
list_min_distance = []
list_ave_distance = []
particles = swarm(screensize=screen_size,target_location=xtrg)
clock = pygame.time.Clock()
time1 = time.process_time()
keepGoing = True
iter      = 0
t         = 0
counter   = 0

#while keepGoing:
#    try:
particles.run(keepGoing=True)

#xtrg      = np.add(xtrg,np.multiply([4,4],0.01))
remainder = iter % 100        
#            
t = t + delta_t
if t >= t_final:
    keepGoing = False
            
