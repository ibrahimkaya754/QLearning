# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:10:32 2017

@author: ikaya
"""
#get_ipython().magic('reset -sf')
# Import neccessary modules
import time
import math
import matplotlib.pyplot as plt
import pygame
import random
import numpy as np
from pygame.locals import *
from PygameModule import *
from Swarm_Algorithm_RuleBased import *
from QLearningClass import *

agent = QLearning(state_number=5, act_num=2, location='./model3files')

# Simulation Parameters
number_of_particles = 51
number_of_axes      = 2
delta_t             = 0.1
t_final             = 1000
wght_leader         = np.array((0.0,0.0,0.0,0.0,10.0,0.0))

# main function
def main(paramaters=[19.087,77.570,74.741,49.385,50.461,31.121,715.096,
                     87.658,-8.527,9.441,-1.126,7.908,5.326,-3.060],TimeConstant=0.1):
    wght_fllwr               = np.array((paramaters[8],paramaters[9],paramaters[10],
                                         paramaters[11],paramaters[12],paramaters[13]),dtype='double')
    distances                = np.zeros((number_of_particles,number_of_particles*number_of_axes))
    distances_abs            = np.zeros((number_of_particles,number_of_particles))
    closest_distances_abs    = np.zeros((number_of_particles,number_of_particles))
    closest_particles_abs    = np.zeros((number_of_particles,number_of_particles),dtype='int32')
    closestneighbours        = np.zeros((number_of_particles),dtype='int16')
    position                 = np.zeros((number_of_particles,number_of_axes)) 
    velocity                 = np.zeros((number_of_particles,number_of_axes)) 
    position_delta           = np.zeros((number_of_particles,number_of_axes))
    dist_twp                 = np.zeros((number_of_particles-1,1))

    screen_size = [3000, 1300]
    screen = pygame.display.set_mode(tuple(screen_size))
    pygame.display.set_caption("Swarm")
    
    xtrg              = [2200.0,650.0] + np.round(np.multiply(np.subtract(screen_size,[2700,1200]),[np.random.random()]))
    #xtrg              = [2400,200]
    list_min_distance = []
    list_ave_distance = []
            
    background = pygame.Surface(screen.get_size())
    background.fill((255, 255, 255))
    screen.blit(background, (0, 0))
    
    particles = np.zeros((number_of_particles),dtype = particle)
    for ii in range(number_of_particles-2):
        particles[ii]  = particle(screen, background, color =0)
        position[ii,0] = particles[ii].positionx
        position[ii,1] = particles[ii].positiony
        velocity[ii,0] = particles[ii].velx
        velocity[ii,1] = particles[ii].vely
    for ii in range(number_of_particles-2,number_of_particles-1):
        particles[ii]  = particle(screen, background, color =1)
        position[ii,0] = particles[ii].positionx
        position[ii,1] = particles[ii].positiony
        velocity[ii,0] = particles[ii].velx
        velocity[ii,1] = particles[ii].vely
    for ii in range(number_of_particles-1,number_of_particles):
        particles[ii]  = particle(screen, background, color =2)
        particles[ii].positionx = xtrg[0]
        particles[ii].positiony = xtrg[1]
        position[ii,0] = particles[ii].positionx
        position[ii,1] = particles[ii].positiony
        velocity[ii,0] = particles[ii].velx
        velocity[ii,1] = particles[ii].vely
    
    dist = distance(population_number=number_of_particles,dimension=number_of_axes,
                    distances=distances,distances_abs=distances_abs,
                    position=position,
                    closest_distances_abs=closest_distances_abs,closest_particles_abs=closest_particles_abs,
                    closestneighbours=closestneighbours)
    
    distances,distances_abs,closest_distances_abs,closest_particles_abs,closestneighbours = dist.find_distances()
    swarm_algo_follower = swarm_algorithm(params=paramaters)
    swarm_algo_leader   = swarm_algorithm(params=paramaters)
    
    allSprites = pygame.sprite.Group(particles[:]) # Grouping the objects to use the uniform method
    time1 = time.process_time()
    keepGoing = True
    iter      = 0
    t         = 0
    counter   = 0

    while keepGoing:
        try:
            xtrg      = np.add(xtrg,np.multiply([4,4],0.01))
            remainder = iter % 100        
            for ii in range(number_of_particles-2):
                dist_to_ldr = dist.dist_to_wypnt(position[ii],position[number_of_particles-2])
                for jj in range(number_of_axes):
                    swarm_algo_follower.algo(particle=ii,axes=jj,position=position,velocity=velocity,
                                             closest_particles_abs=closest_particles_abs,
                                             xtrg=position[number_of_particles-2],TimeConstant=TimeConstant,
                                             wght=wght_fllwr,distance_to_target=dist_to_ldr)
                    position_delta[ii,jj] = swarm_algo_follower.position_delta
                    velocity[ii,jj]       = swarm_algo_follower.velocity[ii,jj]
                
                particles[ii].dx = position_delta[ii,0]
                particles[ii].dy = position_delta[ii,1]
                
                particles[ii].update()
                position[ii,0] = particles[ii].positionx
                position[ii,1] = particles[ii].positiony                    
            
            for ii in range(number_of_particles-2,number_of_particles-1):
                dist_to_trg = dist.dist_to_wypnt(position[ii],xtrg)
                for jj in range(number_of_axes): 
                    swarm_algo_leader.algo(particle=ii,axes=jj,position=position,velocity=velocity,
                                           closest_particles_abs=closest_particles_abs,TimeConstant=TimeConstant, 
                                           xtrg=xtrg,wght=wght_leader,distance_to_target=dist_to_trg)
                    
                    position_delta[ii,jj] = swarm_algo_leader.position_delta  
                    velocity[ii,jj]       = swarm_algo_leader.velocity[ii,jj]
    
                particles[ii].dx = position_delta[ii,0]
                particles[ii].dy = position_delta[ii,1]
                particles[ii].update()
                position[ii,0] = particles[ii].positionx
                position[ii,1] = particles[ii].positiony       
                              
            particles[number_of_particles-1].dx          = 0
            particles[number_of_particles-1].dy          = 0
            particles[number_of_particles-1].update()
            particles[number_of_particles-1].positionx   = xtrg[0]
            particles[number_of_particles-1].positiony   = xtrg[1]                
            particles[number_of_particles-1].rect.center = (particles[number_of_particles-1].positionx, 
                                                            particles[number_of_particles-1].positiony)
            
            position[number_of_particles-1,0]            = particles[number_of_particles-1].positionx
            position[number_of_particles-1,1]            = particles[number_of_particles-1].positiony
            
            distances,distances_abs,closest_distances_abs, \
            closest_particles_abs,closestneighbours = dist.find_distances()
            dist_to_trg = dist.dist_to_wypnt(position[number_of_particles-2],xtrg)
            
            iter = iter + 1
            t    = t + delta_t
            counter = counter + 1
            list_min_distance.append(np.min(closest_distances_abs[0:98,1]))
            list_ave_distance.append(np.average(closest_distances_abs[0:98,1:100]))
            
            if dist_to_trg <= 18.0:
                if np.random.rand() >= 0.9995:
                    xtrg      = [50.0,50.0] + \
                                np.round(np.multiply(np.subtract(screen_size,[100,100]),[np.random.random()]))
            
            if t >= t_final:
                keepGoing = False
            
            if remainder == 0:
                print('time = ',t,' s ', ' target_pos = ', xtrg)
                print('average_min_distance = ', np.average(list_min_distance))
                print('average_group_distance = ', np.average(list_ave_distance))
                print('\n')
                counter = 0
#################################################################################################################################
            allSprites.clear(screen, background)
            allSprites.draw(screen)\

            pygame.display.flip()
        except:
            print('unexpected error --- sorry')
            keepGoing = False
            
    pygame.quit()
    time2 = time.process_time()
    delta_time = time2 - time1
    return position,velocity,distances,closestneighbours,delta_time, \
           closest_particles_abs, closest_distances_abs, list_min_distance, list_ave_distance

# run
if __name__ == "__main__":
    position,velocity,distances,closestneighbours,delta_time, \
    closest_particles_abs,closest_distances_abs, list_min_distance, list_ave_distance = main(TimeConstant=0.001)



'''
###################### Tensorflow Ram Kullanımının optimize edilmesi ####################################################################
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

##################################################################################################################################

######################### SMOOTH TRANSFER FUNCTION ########################################################################################
def smooth_transfer_function(x,R,d):
    if x<=R:
        res = 0
    elif x>R and x<=R+d:
        res = (np.sin(np.pi/d*(x-R) - np.pi/2) + 1) / 2
    else:
        res = 1
    return res
###########################################################################################################################################
    
def act(action):
    if action[0,0] == 0:
        delta_velx = -10.0
    elif action[0,0] == 1:
        delta_velx = -9.0
    elif action[0,0] == 2:
        delta_velx = -8.0
    elif action[0,0] == 3:
        delta_velx = -7.0
    elif action[0,0] == 4:
        delta_velx = -6.0
    elif action[0,0] == 5:
        delta_velx = -5.0
    elif action[0,0] == 6:
        delta_velx = -4.0
    elif action[0,0] == 7:
        delta_velx = -3.0
    elif action[0,0] == 8:
        delta_velx = -2.0
    elif action[0,0] == 9:
        delta_velx = -1.0
    elif action[0,0] == 10:
        delta_velx = -0.8
    elif action[0,0] == 11:
        delta_velx = -0.6
    elif action[0,0] == 12:
        delta_velx = -0.4
    elif action[0,0] == 13:
        delta_velx = -0.2
    elif action[0,0] == 14:
        delta_velx = -0.1
    elif action[0,0] == 15:
        delta_velx = 0.0
    elif action[0,0] == 16:
        delta_velx = 0.1
    elif action[0,0] == 17:
        delta_velx = 0.2
    elif action[0,0] == 18:
        delta_velx = 0.4
    elif action[0,0] == 19:
        delta_velx = 0.6
    elif action[0,0] == 20:
        delta_velx = 0.8
    elif action[0,0] == 21:
        delta_velx = 1.0
    elif action[0,0] == 22:
        delta_velx = 2.0
    elif action[0,0] == 23:
        delta_velx = 3.0
    elif action[0,0] == 24:
        delta_velx = 4.0
    elif action[0,0] == 25:
        delta_velx = 5.0
    elif action[0,0] == 26:
        delta_velx = 6.0
    elif action[0,0] == 27:
        delta_velx = 7.0
    elif action[0,0] == 28:
        delta_velx = 8.0
    elif action[0,0] == 29:
        delta_velx = 9.0
    elif action[0,0] == 30:
        delta_velx = 10.0
    
    if action[0,1] == 0:
        delta_vely = -10.0
    elif action[0,1] == 1:
        delta_vely = -9.0
    elif action[0,1] == 2:
        delta_vely = -8.0
    elif action[0,1] == 3:
        delta_vely = -7.0
    elif action[0,1] == 4:
        delta_vely = -6.0
    elif action[0,1] == 5:
        delta_vely = -5.0
    elif action[0,1] == 6:
        delta_vely = -4.0
    elif action[0,1] == 7:
        delta_vely = -3.0
    elif action[0,1] == 8:
        delta_vely = -2.0
    elif action[0,1] == 9:
        delta_vely = -1.0
    elif action[0,1] == 10:
        delta_vely = -0.8
    elif action[0,1] == 11:
        delta_vely = -0.6
    elif action[0,1] == 12:
        delta_vely = -0.4
    elif action[0,1] == 13:
        delta_vely = -0.2
    elif action[0,1] == 14:
        delta_vely = -0.1
    elif action[0,1] == 15:
        delta_vely = 0.0
    elif action[0,1] == 16:
        delta_vely = 0.1
    elif action[0,1] == 17:
        delta_vely = 0.2
    elif action[0,1] == 18:
        delta_vely = 0.4
    elif action[0,1] == 19:
        delta_vely = 0.6
    elif action[0,1] == 20:
        delta_vely = 0.8
    elif action[0,1] == 21:
        delta_vely = 1.0
    elif action[0,1] == 22:
        delta_vely = 2.0
    elif action[0,1] == 23:
        delta_vely = 3.0
    elif action[0,1] == 24:
        delta_vely = 4.0
    elif action[0,1] == 25:
        delta_vely = 5.0
    elif action[0,1] == 26:
        delta_vely = 6.0
    elif action[0,1] == 27:
        delta_vely = 7.0
    elif action[0,1] == 28:
        delta_vely = 8.0
    elif action[0,1] == 29:
        delta_vely = 9.0
    elif action[0,1] == 30:
        delta_vely = 10.0
       
    return delta_velx,delta_vely     

######################### MAIN FUNCTION ###################################################################################################
epochs                   = 50000
action                   = np.zeros((1,2))
max_score                = 0.0  
max_time                 = 0.0  
number_of_particles      = 100
number_of_particles_RL   = 1
number_of_particles_trg  = 1
number_of_particles_lead = 20
number_of_particles_swrm = number_of_particles - (number_of_particles_RL+number_of_particles_trg+number_of_particles_lead)
number_of_axes           = 2
row                      = int(number_of_particles)
col                      = number_of_axes

########################Q Learning Params##################################
Qlearning = QLearningClass.QLearning(gamma = 0.95, batchSize = 100, buffer = 50000, nghbr = int(5), state_number = int(20),
                                     act_num=2, location='./model3files',annealing=1000)
Qlearning.saved_data(use_saved_data = False, eps = 1.0, saved_replay = [])
###########################################################################

for epch in range(epochs):
    iter    = 0
    timee   = 0
    counter = 0
    score   = 0.0
    ######################### PARTICLE SWARM PARAMETERS #######################################################################################
    delta_t      = 0.1
    t_final      = 10/0.1
    
    r0           = 19.087
    r1           = 77.570
    r2           = 74.741
    D            = 49.385
    Cfrict       = 50.461
    Cshill       = 31.121
    R            = 715.096
    d            = 87.658
    
    vflock       = 20    
    wght         = np.array((-8.527,9.441,-1.126,7.908,5.326,-3.060),dtype='double')
    ###########################################################################################################################################
    distances                       = np.zeros((number_of_particles,number_of_particles*number_of_axes))
    distances_leaders               = np.zeros((number_of_particles,number_of_particles_lead*number_of_axes))
    distances_abs                   = np.zeros((number_of_particles,number_of_particles))
    closest_distances_abs           = np.zeros((number_of_particles,number_of_particles))
    closest_leaders_abs             = np.zeros((number_of_particles,number_of_particles_lead),dtype='int16')
    closest_leaders_distances_abs   = np.zeros((number_of_particles,number_of_particles_lead))
    closest_particles_abs           = np.zeros((number_of_particles,number_of_particles),dtype='int16')
    closestneighbours               = np.zeros((number_of_particles),dtype='int16')
    closestneighbour_number         = np.zeros((number_of_particles),dtype='int16')
    position                        = np.zeros((number_of_particles,number_of_axes)) 
    velocity                        = np.zeros((number_of_particles,number_of_axes)) 
    position_delta                  = np.zeros((number_of_particles,number_of_axes)) 
    
#    lower_boundaries        = np.array([0,0])          # x,y domain için minimum sınırlar
#    upper_boundaries        = np.array([1500,750])            # x,y domain için maximum sınırlar
    vel_min                 = -20                              # particles' minimum velocity
    vel_max                 = 20                               # particles' maximum velocity
    xtrg                    = np.array((600+np.random.random()*600,400+np.random.random()*300),dtype = 'double')
    
    state      = np.zeros((1,Qlearning.state_number),dtype='double')
    newstate   = np.zeros((1,Qlearning.state_number),dtype='double')
    
    screen = pygame.display.set_mode((1800, 1100))
    pygame.display.set_caption("Swarming Particles")
    
    background = pygame.Surface(screen.get_size())
    background.fill((255, 255, 255))
    screen.blit(background, (0, 0))
    
    particles = np.zeros((number_of_particles),dtype = PygameModule.particle)
    rand2 = random.random()
    for ii in range(number_of_particles_swrm):
        particles[ii]  = PygameModule.particle(screen, background, 2, rand2, 0)
        position[ii,0] = particles[ii].positionx
        position[ii,1] = particles[ii].positiony
        velocity[ii,0] = particles[ii].velx
        velocity[ii,1] = particles[ii].vely
    for ii in range(number_of_particles_swrm,number_of_particles_swrm+number_of_particles_lead):
        particles[ii]  = PygameModule.particle(screen, background, 3, rand2, 0)
        position[ii,0] = particles[ii].positionx
        position[ii,1] = particles[ii].positiony
        velocity[ii,0] = particles[ii].velx
        velocity[ii,1] = particles[ii].vely
    for ii in range(number_of_particles_swrm+number_of_particles_lead,number_of_particles_swrm+number_of_particles_lead+number_of_particles_RL):
        particles[ii]  = PygameModule.particle(screen, background, 0, rand2, 1)
        position[ii,0] = particles[ii].positionx
        position[ii,1] = particles[ii].positiony
        velocity[ii,0] = particles[ii].velx
        velocity[ii,1] = particles[ii].vely
    for ii in range(number_of_particles_swrm+number_of_particles_lead+number_of_particles_RL,number_of_particles_swrm+number_of_particles_lead+number_of_particles_RL+number_of_particles_trg):
        particles[ii]  = PygameModule.particle(screen, background, 1, rand2, 0)
        particles[ii].positionx = xtrg[0]
        particles[ii].positiony = xtrg[1]
        position[ii,0] = particles[ii].positionx
        position[ii,1] = particles[ii].positiony
        velocity[ii,0] = particles[ii].velx
        velocity[ii,1] = particles[ii].vely

    
    distances,distances_abs,closest_distances_abs,closest_particles_abs,closestneighbours,distances_leaders,closest_leaders_abs,closest_leaders_distances_abs = find_distances(row,col,distances,distances_abs,position,closest_distances_abs,closest_particles_abs,closestneighbours,number_of_particles_swrm,number_of_particles_lead,closest_leaders_abs,closest_leaders_distances_abs)
    
    allSprites = pygame.sprite.Group(particles[:]) # SONRADAN EKLENEN OBJELERI UNIFORM METOD KULLANMAK ICIN TEK GRUBA ALMA
    
    clock = pygame.time.Clock()
    keepGoing = True

    ####### bu kısım yeni ekleniyor 22/06/18 #######
#    total_neighbour_distance = 0.0
#    for ngb in range(Qlearning.nghbr-1):
#        total_neighbour_distance = total_neighbour_distance + closest_distances_abs[row-2, 1 + ngb]
#    total_neighbour_distance     = total_neighbour_distance + closest_leaders_distances_abs[row-2,0]
#    total_neighbour_distance_avg = total_neighbour_distance / Qlearning.nghbr
#    ################################################
    
    for i in range(Qlearning.nghbr-1):
        state[0,i]                                                  = distances[row-2,closest_particles_abs[row-2,i+1]]
        state[0,i+Qlearning.nghbr]                                  = distances[row-2,closest_particles_abs[row-2,i+1]+row]
        state[0,i+Qlearning.nghbr+Qlearning.nghbr]                  = velocity[closest_particles_abs[row-2,i+1],0]
        state[0,i+Qlearning.nghbr+Qlearning.nghbr+Qlearning.nghbr]  = velocity[closest_particles_abs[row-2,i+1],1]
    
    for i in range(Qlearning.nghbr-1,Qlearning.nghbr):
        state[0,i]                                                  = distances[row-2,number_of_particles_swrm+closest_leaders_abs[row-2,0]]
        state[0,i+Qlearning.nghbr]                                  = distances[row-2,number_of_particles_swrm+closest_leaders_abs[row-2,0]+row]
        state[0,i+Qlearning.nghbr+Qlearning.nghbr]                  = velocity[number_of_particles_swrm+closest_leaders_abs[row-2,0],0]
        state[0,i+Qlearning.nghbr+Qlearning.nghbr+Qlearning.nghbr]  = velocity[number_of_particles_swrm+closest_leaders_abs[row-2,0],1]
#    state[0,-1] = total_neighbour_distance_avg
    
    print("Game #: %s" % (epch,))
    done            = False
    max_score_done  = False
    max_time_done   = False
    while not done:
        try:
            remainder = iter % 100
            
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == QUIT:
                    keepGoing = False
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    keepGoing = False
                    
            for ii in range(row-2):
                for jj in range(col):
                    apot = 0
                    aslp = 0
                    v1   = 0
                    a1   = 0
                    a6   = 0
                    closestneighbour_number[ii] = 3
                    for ff in range(1,closestneighbour_number[ii]):
                        nb = ff
                        ds = 1
                        dc = ds/2
                        dt = 1
                        if (np.abs(position[closest_particles_abs[ii,nb],jj] - position[ii,jj])) < r0:
                            apot = apot + D * np.minimum(r1,r0 - np.abs(position[closest_particles_abs[ii,nb],jj] - position[ii,jj])) * (position[closest_particles_abs[ii,nb],jj] - position[ii,jj]) / np.abs(position[closest_particles_abs[ii,nb],jj] - position[ii,jj])
                        else:
                            apot = apot + 0
                        aslp = aslp + Cfrict * (velocity[closest_particles_abs[ii,nb],jj] - velocity[ii,jj]) / (np.maximum(np.abs(position[closest_particles_abs[ii,nb],jj] - position[ii,jj])-(r0-r2),1))**2
                        a1   = a1 + (position[closest_particles_abs[ii,nb],jj] - position[ii,jj]) / closestneighbour_number[ii]
                        a6   = a6 + xtrg[jj] - position[ii,jj]
                        v1   = v1 + (velocity[closest_particles_abs[ii,nb],jj] - velocity[ii,jj]) / closestneighbour_number[ii]
                    vspp  = vflock * velocity[ii,jj] / np.abs(velocity[ii,jj])
                    awall = Cshill * smooth_transfer_function(np.abs(xtrg[jj] - position[ii,jj]),R,d) * (vflock * ((xtrg[jj] - position[ii,jj]) / np.abs(xtrg[jj] - position[ii,jj])) - velocity[ii,jj])
                    vtrack = 0
                    vel   = velocity[ii,jj]
                    
                    velocity[ii,jj] = velocity[ii,jj] + wght[5] * (v1+vspp+vtrack-velocity[ii,jj]) * delta_t + (wght[0] * apot + wght[1] * aslp + wght[2] * awall + wght[3] * a1 + wght[4] * a6) * delta_t
                    if math.isinf(velocity[ii,jj]):
                        velocity[ii,jj] = vel
                    if velocity[ii,jj]>vel_max:
                        velocity[ii,jj] = vel_max
                    elif velocity[ii,jj]<vel_min:
                        velocity[ii,jj] = vel_min
                        
                    position_delta[ii,jj] = velocity[ii,jj] * delta_t
    
                particles[ii].dx = position_delta[ii,0]
                particles[ii].dy = position_delta[ii,1]
                particles[ii].update()
                position[ii,0] = particles[ii].positionx
                position[ii,1] = particles[ii].positiony
            
            ################################# Target Position and Velocity ##########################################        
            if counter == 0:
                randx = 2.0 * random.random() * random.randint(-1,1)
                randy = 2.0 * random.random() * random.randint(-1,1)
            particles[row-1].dx = 1 * randx
            particles[row-1].dy = 1 * randy
            particles[row-1].update()
            if particles[row-1].positionx > 1600 or particles[row-1].positionx < 200 or particles[row-1].positiony > 900 or particles[row-1].positiony < 200:
                particles[row-1].positionx = 900
                particles[row-1].positiony = 550
                position[row-1,0] = particles[row-1].positionx
                position[row-1,1] = particles[row-1].positiony
                velocity[row-1,0] = particles[row-1].velx
                velocity[row-1,1] = particles[row-1].vely
            
            ################################### Reinforcement #########################################################
            qval = Qlearning.model1.predict(state.reshape(1,Qlearning.state_number), batch_size=1)
            if random.random() < Qlearning.epsilon:
                action[0,0] = random.randint(0,30)
                action[0,1] = random.randint(0,30)
            else:
                action[0,0] = np.argmax(qval[0][0:31])
                action[0,1] = np.argmax(qval[0][31:62])
            
            delta_velx,delta_vely = act(action)
            
            particles[row-2].velx = particles[row-2].velx + delta_velx
            particles[row-2].vely = particles[row-2].vely + delta_vely
            
            if particles[row-2].velx > vel_max:
                particles[row-2].velx = vel_max
            elif particles[row-2].vely > vel_max:
                particles[row-2].vely = vel_max
            elif particles[row-2].velx < vel_min:
                particles[row-2].velx = vel_min
            elif particles[row-2].vely < vel_min:
                particles[row-2].vely = vel_min
            
            particles[row-2].dx   = particles[row-2].velx * delta_t
            particles[row-2].dy   = particles[row-2].vely * delta_t
            particles[row-2].update()
            velocity[row-2,0]     = particles[row-2].velx
            velocity[row-2,1]     = particles[row-2].vely
            position[row-2,0]     = particles[row-2].positionx
            position[row-2,1]     = particles[row-2].positiony
    
            ###########################################################################################################
            
            particles[row-1].rect.center    = (particles[row-1].positionx, particles[row-1].positiony)
            particles[row-2].rect.center    = (particles[row-2].positionx, particles[row-2].positiony)
            
            distances,distances_abs,closest_distances_abs,closest_particles_abs,closestneighbours,distances_leaders,closest_leaders_abs,closest_leaders_distances_abs = find_distances(row,col,distances,distances_abs,position,closest_distances_abs,closest_particles_abs,closestneighbours,number_of_particles_swrm,number_of_particles_lead,closest_leaders_abs,closest_leaders_distances_abs)
            xtrg                    = np.array((particles[row-1].positionx, particles[row-1].positiony))
            
            ####### bu kısım yeni ekleniyor 22/06/18 #######
            total_neighbour_distance = 0.0
            for ngb in range(Qlearning.nghbr-1):
                total_neighbour_distance = total_neighbour_distance + closest_distances_abs[row-2, 1 + ngb]
            total_neighbour_distance     = total_neighbour_distance + closest_leaders_distances_abs[row-2,0]
            total_neighbour_distance_avg = total_neighbour_distance / Qlearning.nghbr
            ################################################
            
            for i in range(Qlearning.nghbr-1):
                newstate[0,i]                                                   = distances[row-2,closest_particles_abs[row-2,i+1]]
                newstate[0,i+Qlearning.nghbr]                                   = distances[row-2,closest_particles_abs[row-2,i+1]+row]
                newstate[0,i+Qlearning.nghbr+Qlearning.nghbr]                   = velocity[closest_particles_abs[row-2,i+1],0]
                newstate[0,i+Qlearning.nghbr+Qlearning.nghbr+Qlearning.nghbr]   = velocity[closest_particles_abs[row-2,i+1],1]
            
            for i in range(Qlearning.nghbr-1,Qlearning.nghbr):
                newstate[0,i]                                                  = distances[row-2,number_of_particles_swrm+closest_leaders_abs[row-2,0]]
                newstate[0,i+Qlearning.nghbr]                                  = distances[row-2,number_of_particles_swrm+closest_leaders_abs[row-2,0]+row]
                newstate[0,i+Qlearning.nghbr+Qlearning.nghbr]                  = velocity[number_of_particles_swrm+closest_leaders_abs[row-2,0],0]
                newstate[0,i+Qlearning.nghbr+Qlearning.nghbr+Qlearning.nghbr]  = velocity[number_of_particles_swrm+closest_leaders_abs[row-2,0],1]        
            
            ## Reward ##
            if closest_leaders_distances_abs[row-2,0] >= 150.0:
                reward = -10000
            elif closest_leaders_distances_abs[row-2,0] < 150.0 and closest_leaders_distances_abs[row-2,0]>=80:
                reward = - (80 - closest_leaders_distances_abs[row-2,0])**2
            else:
                if closest_distances_abs[row-2,1] < 2.0:
                    reward = -10000
                elif closest_distances_abs[row-2,1] >= 2.0 and closest_distances_abs[row-2,1] < 10.0:
                    reward = 1 * (closest_distances_abs[row-2,1]**3 - total_neighbour_distance)
                else:
                    reward = 1 * (975 - closest_distances_abs[row-2,1]**1.570 - total_neighbour_distance)

            Qlearning.replay_list(state,action,reward,newstate)
    
            state      = newstate
            newstate   = np.zeros((1,Qlearning.state_number),dtype='double')
            
            iter = iter + 1
            timee    = timee + delta_t
            counter = counter + 1
            
            if remainder == 0:
                print('time = ',timee,' s')
                counter = 0            
            
            Qlearning.epsilon = Qlearning.epsilon - (1/Qlearning.buffer)  
            
            if Qlearning.epsilon < 0.05:
                Qlearning.epsilon = 0.05   
                
            score = score + reward    
            if score > max_score:
                max_score = score
                max_score_done = True
            if timee > max_time:
                max_time = timee
            print('epoch = %s , reward = %s , score = %s , time = %s' % (epch,reward,score,timee))
            
            if score <= -500000 or reward == -10000 or timee >= 5000:
                keepGoing = False
                print('time = ',timee,' s')
                print('max_time_ever = ',max_time)
                print('max_score_ever = ',max_score)
                Qlearning.save(timee,score,max_score_done,max_time_done)
                replay_saved = Qlearning.save_replay()            
        
    ###########################################################################################################################################               
            allSprites.clear(screen, background)
            allSprites.draw(screen)         
            pygame.display.flip()
        except:
            keepGoing = False
            print('time = ',timee,' s')
            print('max_time_ever = ',max_time)
            print('max_score_ever = ',max_score)
            Qlearning.save("models_follower_min_distance_model3",timee,score)
            replay_saved = Qlearning.save_replay()            
    
    Qlearning.train_model(epch,choose)
pygame.quit()

###########################################################################################################################################
'''