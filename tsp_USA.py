# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:10:03 2017

@author: miguel & ronan

"""
#%% Import libraries & functions

# import libraries
import numpy as np
import matplotlib.pyplot as plt

# import functions
from tspFunctions import *

#%% Load problem

distances_file = 'tsp_matrices/att48_d.csv'
optimal_route_file = 'tsp_matrices/att48_s.csv'

int_R, optimal_route, optimal_route_cost =  loadTSPmatrix(distances_file, optimal_route_file)

#%% Define parameters

epochs = 200 # init epochs count
start = 0 # define start point at row 0

goal_state_reward = 1000

alphas = np.array([0.9]).astype('float32')
gammas = np.array([0.01]).astype('float32')
epsilons = np.array([0.9]).astype('float32')
epsilon_decays = np.array([0.001]).astype('float32')

sampling_sampling_runs = 2

#%% Q-Learning

# init variables for recording Q Learning metrics
seqs = [] # list of lists, where each inner list records the state transitions made in an epoch. State transitions for each epoch in each sampling_run are recorded.
costs_matrix = np.zeros((epochs, sampling_sampling_runs)) # matrix containing epoch cost vectors for each sampling sampling_run
mean_costs_matrix = np.zeros((epochs, np.size(alphas)*np.size(gammas)*np.size(epsilons)*np.size(epsilon_decays))) # contains mean of costs for all sampling_runs of each parameter setting
ps_dic = {} # init; parameter search dictionary
loop_idx = 0 # init


for a in range(0, np.size(alphas)):
    for g in range(0, np.size(gammas)):
        for e in range(0, np.size(epsilons)):
            for d in range(0, np.size(epsilon_decays)):
                
                alpha = alphas[a]; gamma = gammas[g]; epsilon = epsilons[e]; epsilon_decay = epsilon_decays[d]
                print('\nRunning Variant %s [Alpha: %.2f, Gamma: %.2f, Epsilon: %.2f, Decay: %.4f]...' % (loop_idx+1, alpha, gamma, epsilon, epsilon_decay))

                for sampling_run in range (0, sampling_sampling_runs):
                
                    epoch_costs, trans_seqs, _ = qLearn(epochs, int_R, start, alpha, gamma, epsilon, epsilon_decay, goal_state_reward)
                    
                    # record all sequences followed in current sampling_run
                    for epoch in range(0, epochs):
                        seqs.append(trans_seqs[epoch])
                        
                    # add vector of epoch costs to costs_matrix
                    costs_matrix[:, sampling_run] = epoch_costs
                    
                    # calculate mean costs across all 'sampling_runs'
                    mean_costs = np.mean(costs_matrix, 1)
                 
                ps_dic[loop_idx] = ('Epsilon Decay = %.2f' % epsilon_decay) 
                #ps_dic[loop_idx] = ('A [%.2f], G [%.2f], E [%.2f], D [%.4f]' % (alpha, gamma, epsilon, epsilon_decay))   
                mean_costs_matrix[:, loop_idx] = mean_costs; loop_idx +=1


# Calculates average cost of previous epochs (up to 'n' previous epochs)
n = 20
window_ave = np.zeros_like(mean_costs_matrix)
for k in range(0,int(np.size(mean_costs_matrix,1))):
    for i in range(1,int(np.size(mean_costs_matrix[:,k])+1)):
        if i<n-1:
            window_ave[i-1,k] = (np.mean(mean_costs_matrix[:,k][0:i]))
        else:
            window_ave[i-1,k] = (np.mean(mean_costs_matrix[:,k][i-(n-1):i]))

norm_cost = window_ave/optimal_route_cost

#%% Clear Redundant Variables from workspace

# clear input variables
del start, epoch, epochs, sampling_sampling_runs, goal_state_reward

# clear any variables created solely for 'looping' purposes
del distances_file, optimal_route_file, a, alpha, e, epsilon, g, gamma, d,  epsilon_decay , sampling_run, loop_idx

# clear non-aggregate metrics variables
del trans_seqs, epoch_costs, costs_matrix, mean_costs, mean_costs_matrix


#%% Plot performance graphs


# import graph functions
from tsp_PlotData import *

# Plot line graph ------------------------
variable = ['Agent']              # variable to explore
title = 'Tour in the USA'  # title of graph
    
plotLines(norm_cost,variable,title)


# Plot routes ----------------------------
file_xy = 'tsp_matrices/att48_xy.csv'

plotManyRoutes(seqs,file_xy,variable)
    
