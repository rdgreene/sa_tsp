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
from loadTSPmatrix import loadTSPmatrix
from qLearn import qLearn

#%% Load problem and define parameters

file_name = 'tsp_matrices/toy_d.csv'
int_R = loadTSPmatrix(file_name)

epochs = 100 # init epochs count
start = 0 # define start point at row 0

max_iters = 9999 # redundant? Consider removing
goal_state_reward = 100

alphas = np.array([0.01, 0.03, 0.1, 0.3, 1.0]).astype('float32')
gammas = np.array([0.8]).astype('float32')
epsilons = np.array([0.8]).astype('float32')
epsilon_decays = np.array([0.003]).astype('float32')

sampling_sampling_runs = 10

''' 
#   ***Comment in to prompt selection of learning parameters for Q Learning***

ans = input("Use default values for alpha(%s), gamma(%s), epsilon(%s), epsilon decay factor(%.4f), and goal state reward (%s)? [y/n] > "\
            % (alpha, gamma, epsilon, epsilon_decay, goal_state_reward)) #Define Learning Hyper Parameters
        
if ans != "y":     
    alpha = float(input("alpha value (0-1): ")) 
    gamma = float(input("gamma value (0-1): "))  
    epsilon = float(input("epsilon value (0-1): ")) 
    epsilon_decay = float(input("epsilon decay factor (0-1): ")) 
    #max_iters = int(input("max state transitions per epoch (0-1): ")) # redundant? Consider removing
    goal_state_reward = int(input("reward for reaching goal state: "))    
'''

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
                
                    epoch_costs, trans_seqs, _ = qLearn(epochs, int_R, start, alpha, gamma, epsilon, epsilon_decay, goal_state_reward, max_iters)
                    
                    # record all sequences followed in current sampling_run
                    for epoch in range(0, epochs):
                        seqs.append(trans_seqs[epoch])
                        
                    # add vector of epoch costs to costs_matrix
                    costs_matrix[:, sampling_run] = epoch_costs
                    
                    # calculate mean costs across all 'sampling_runs'
                    mean_costs = np.mean(costs_matrix, 1)
                  
                ps_dic[loop_idx] = ('A [%.2f], G [%.2f], E [%.2f], D [%.4f]' % (alpha, gamma, epsilon, epsilon_decay))   
                mean_costs_matrix[:, loop_idx] = mean_costs; loop_idx +=1

#%% Clear Redundant Variables from workspace

# clear input variables
del start, epoch, epochs, alphas, epsilons, gammas, epsilon_decays, sampling_sampling_runs, max_iters, goal_state_reward

# clear any variables created solely for 'looping' purposes
del file_name, a, alpha, e, epsilon, g, gamma, d,  epsilon_decay , sampling_run, loop_idx

# clear non-aggregate metrics variables
del trans_seqs, epoch_costs, costs_matrix, mean_costs

                
#%% Plot results
    
# plot mean costs
plt.figure(figsize=(15,6)) # set figure size
plt.plot(mean_costs_matrix)
plt.legend(ps_dic.values())
#plt.savefig('alphas.png') # save to drive
plt.show()







