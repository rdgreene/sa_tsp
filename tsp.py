# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:10:03 2017

@author: miguel & ronan

"""

#%% Select whether to run full plotting script, or just quick plotting script

plotting = True

#%% Import library dependancies

import numpy as np
import matplotlib.pyplot as plt
from tspFunctions import * # import bespoke TSP functions

#%% Section One: Load problem environment

distances_file = 'tsp_matrices/toy_d.csv'
optimal_route_file = 'tsp_matrices/toy_s.csv'

int_R, optimal_route, optimal_route_cost =  loadTSPmatrix(distances_file, optimal_route_file)

#%% Section Two: Define parameters

epochs = 1000 # init epochs count
start = 0 # define start point at row 0
goal_state_reward = 100 # define reward for completing a tour

# Define Q-Learning Parameters
alphas = np.array([0.1]).astype('float32')
gammas = np.array([1]).astype('float32')
epsilons = np.array([1]).astype('float32')
epsilon_decays = np.array([0.3]).astype('float32')

# define number of sampling runs
sampling_runs = 1

#%% Section Three: Q-Learning

# init variables for recording Q Learning metrics
seqs = [] # list of lists, where each inner list records the state transitions made in an epoch. State transitions for each epoch in each sampling_run are recorded.
costs_matrix = np.zeros((epochs, sampling_runs)) # matrix containing epoch cost vectors for each sampling sampling_run
mean_costs_matrix = np.zeros((epochs, np.size(alphas)*np.size(gammas)*np.size(epsilons)*np.size(epsilon_decays))) # contains mean of costs for all sampling_runs of each parameter setting
ps_dic = {} # init; parameter search dictionary
loop_idx = 0 # records 'tota' loop count for creating test parameter dictionary


# run Q-Learning for specified parameters
for a in range(0, np.size(alphas)):
    for g in range(0, np.size(gammas)):
        for e in range(0, np.size(epsilons)):
            for d in range(0, np.size(epsilon_decays)):
                
                alpha = alphas[a]; gamma = gammas[g]; epsilon = epsilons[e]; epsilon_decay = epsilon_decays[d]
                print('\nRunning Variant %s [Alpha: %.2f, Gamma: %.2f, Epsilon: %.2f, Decay: %.4f]...' % (loop_idx+1, alpha, gamma, epsilon, epsilon_decay))

                for sampling_run in range (0, sampling_runs):
                
                    epoch_costs, trans_seqs, Q_matrix = qLearn(epochs, int_R, start, alpha, gamma, epsilon, epsilon_decay, goal_state_reward)
                    
                    # record all sequences followed in current sampling_run
                    for epoch in range(0, epochs):
                        seqs.append(trans_seqs[epoch])
                        
                    # add vector of epoch costs to costs_matrix
                    costs_matrix[:, sampling_run] = epoch_costs
                    
                    # calculate mean costs across all 'sampling_runs'
                    mean_costs = np.mean(costs_matrix, 1)
                 
                ps_dic[loop_idx] = (r'$\alpha = %.4f$, $\gamma = %.4f$, $\epsilon = %.4f$, $\lambda = %.4f$' % (alpha, gamma, epsilon, epsilon_decay)) 
                #ps_dic[loop_idx] = ('A [%.2f], G [%.2f], E [%.2f], D [%.4f]' % (alpha, gamma, epsilon, epsilon_decay))   
                mean_costs_matrix[:, loop_idx] = mean_costs; loop_idx +=1


#np.save('tspResults', mean_costs_matrix)                

#%% Section Four: Plot Results

# subtract optimal route baseline from cost data
plotData = mean_costs_matrix/optimal_route_cost

# convert plotData to a window average of last 20 epochs (for smoothing plots)
smoothing = 20
plotData = getWindowAverage(plotData, smoothing) 

if plotting == False:
    plt.figure(figsize=(15,10))
    plt.plot(plotData)
    plt.ylim(ymin=1)
    plt.legend(ps_dic.values())
    plt.ylabel('Cost Above Optimal')
    plt.xlabel('Epochs')
    plt.title('Results')
    #plt.savefig('fig_name')

if plotting == True:

    # import graph functions
    from tsp_PlotData import *
    
    file_xy = 'tsp_matrices/toyd_d_xy.csv'  # file with coordenates
    variable = ['Toy environment']                   # variable to explore
    title = 'Toy environment'             # title of graph
        
    # Plot line graph 
    plotLines(plotData,variable,title)

    # Plot routes
    plotFewRoutes(seqs,file_xy,title)

#%% Clear Redundant Variables from workspace

# clear input variables
del start, epoch, epochs, sampling_runs, goal_state_reward

# clear any variables created solely for 'looping' purposes
del a, alpha, e, epsilon, g, gamma, d,  epsilon_decay , sampling_run, loop_idx

# clear non-aggregate metrics variables
del trans_seqs, epoch_costs, costs_matrix, mean_costs

# euler_gamma, pi
