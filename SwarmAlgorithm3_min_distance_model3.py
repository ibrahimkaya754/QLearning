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
from swarm import *
from QLearningClass import *

# Simulation Parameters
number_of_particles = 51
number_of_axes      = 2
delta_t             = 0.1
t_final             = 1000
#wght_leader         = np.array((0.0,0.0,0.0,0.0,10.0,0.0))

myagent = agent(numberofstate=10,numberofaction=62)

def actions():
    act = np.ndarray(shape=(21,2))
    ctr = 0
    for ii in range(0,21):
        act[ii,0] = int(ctr)
        act[ii,1] = (ii-10)/10
        ctr = ctr + 1
    return act

# main function
def main():
    wght_fllwr               = np.array((paramaters[8],paramaters[9],paramaters[10],
                                         paramaters[11],paramaters[12],paramaters[13]),dtype='double')
#    distances                = np.zeros((number_of_particles,number_of_particles*number_of_axes))
#    distances_abs            = np.zeros((number_of_particles,number_of_particles))
#    closest_distances_abs    = np.zeros((number_of_particles,number_of_particles))
#    closest_particles_abs    = np.zeros((number_of_particles,number_of_particles),dtype='int32')
#    closestneighbours        = np.zeros((number_of_particles),dtype='int16')
#    position                 = np.zeros((number_of_particles,number_of_axes)) 
#    velocity                 = np.zeros((number_of_particles,number_of_axes)) 
#    position_delta           = np.zeros((number_of_particles,number_of_axes))
    dist_twp                 = np.zeros((number_of_particles-1,1))

    screen_size = [3000, 1300]
#    screen = pygame.display.set_mode(tuple(screen_size))
#    pygame.display.set_caption("Swarm")
    
#     xtrg              = [2200.0,650.0] + np.round(np.multiply(np.subtract(screen_size,[2700,1200]),[np.random.random()]))
    xtrg              = [2400,200]
    list_min_distance = []
    list_ave_distance = []
            
#    background = pygame.Surface(screen.get_size())
#    background.fill((255, 255, 255))
#    screen.blit(background, (0, 0))
    
#    particles = np.zeros((number_of_particles),dtype = particle)
    
    particles = swarm(screensize=[1500,800])
#    for ii in range(number_of_particles-2):
#        particles[ii]  = particle(screen, background, color =0)
#        position[ii,0] = particles[ii].positionx
#        position[ii,1] = particles[ii].positiony
#        velocity[ii,0] = particles[ii].velx
#        velocity[ii,1] = particles[ii].vely
#    for ii in range(number_of_particles-2,number_of_particles-1):
#        particles[ii]  = particle(screen, background, color =1)
#        position[ii,0] = particles[ii].positionx
#        position[ii,1] = particles[ii].positiony
#        velocity[ii,0] = particles[ii].velx
#        velocity[ii,1] = particles[ii].vely
#    for ii in range(number_of_particles-1,number_of_particles):
#        particles[ii]  = particle(screen, background, color =2)
#        particles[ii].positionx = xtrg[0]
#        particles[ii].positiony = xtrg[1]
#        position[ii,0] = particles[ii].positionx
#        position[ii,1] = particles[ii].positiony
#        velocity[ii,0] = particles[ii].velx
#        velocity[ii,1] = particles[ii].vely
    
#    dist = distance(population_number=number_of_particles,dimension=number_of_axes,
#                    distances=distances,distances_abs=distances_abs,
#                    position=position,
#                    closest_distances_abs=closest_distances_abs,closest_particles_abs=closest_particles_abs,
#                    closestneighbours=closestneighbours)
    
#    distances,distances_abs,closest_distances_abs,closest_particles_abs,closestneighbours = dist.find_distances()
#    swarm_algo_follower = swarm_algorithm(params=paramaters)
#    swarm_algo_leader   = swarm_algorithm(params=paramaters)
    
#    allSprites = pygame.sprite.Group(particles[:]) # Grouping the objects to use the uniform method
    clock = pygame.time.Clock()
    time1 = time.process_time()
    keepGoing = True
    iter      = 0
    t         = 0
    counter   = 0

    while keepGoing:
        try:
            #xtrg      = np.add(xtrg,np.multiply([4,4],0.01))
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
            
            if dist_to_trg <= 12.0:
                if np.random.rand() >= 0.99:
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
    closest_particles_abs,closest_distances_abs, list_min_distance, list_ave_distance = main(TimeConstant=0.50)