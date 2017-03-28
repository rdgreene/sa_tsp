# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:10:03 2017

@author: miguel & ronan

"""

#%% Import libraries & functions

# import libraries
import numpy as np
from scipy import *
import matplotlib.pyplot as plt

# import functions
from loadTSPmatrix import loadTSPmatrix
from qLearn import qLearn

#%% Load problem and define parameters

file_name = 'tsp_matrices/toy_d.csv'
int_R = loadTSPmatrix(file_name)

epochs = 2000 # init epochs count
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
# del start, epoch, epochs, alphas, epsilons, gammas, epsilon_decays, sampling_sampling_runs, max_iters, goal_state_reward

# clear any variables created solely for 'looping' purposes
del file_name, a, alpha, e, epsilon, g, gamma, d,  epsilon_decay , sampling_run, loop_idx

# clear non-aggregate metrics variables
del trans_seqs, epoch_costs, costs_matrix, mean_costs

                
#%%  
#------------ Plot line graph ------------------------

from plotdata import plotline

baseline = 110                  # minimum posible cost
n = 50                          # calculates average cost of previous epochs (up to 'n' previous epochs)
alphas = np.array([0.01, 0.03, 0.1, 0.3, 1.0]).astype('float32')
variable = alphas               # variable to explore
title = 'Learing Alpha Search'  # title of graph

plotline(mean_costs_matrix,alphas,n,baseline)


# %% 
#------------ Plot routes ----------------------------

#from plotdata import plotroutes



# inside function
# def plotroutes()

import collections

nodes_font = {'family': 'fantasy',
        'color':  'black',
        'weight': 400,
        'size': 14 }

d = collections.OrderedDict()
for a in np.asarray(transition_seqs):
    t = tuple(a)
    if t in d:
        d[t] += 1
    else:
        d[t] = 1

transition_summary = []
for (key, value) in d.items():
    transition_summary.append(list(key) + [value])
    
fig, mapa = plt.subplots()
mapa.patch.set_facecolor('forestgreen')
#mapa.set_ylim()
#mapa.set_xlim()
plt.grid(b=False)
plt.title('Map of Paths', y=1.1,fontdict=title_font)
plt.ylabel('latitud', fontdict=label_font)
plt.xlabel('longitud', fontdict=label_font)
plt.xticks([], [])
plt.yticks([], [])

nodes = np.asarray([coordenates[x,:] for x in temp[0:-1,0]])

cities = mapa.scatter(nodes[:,0],nodes[:,1],    # plot nodes
                      c='white', s=800, 
                      label='white',alpha=1, 
                      edgecolors='black', zorder=3)

temp = np.asarray(transition_summary).T

for i in range(0,size(nodes,0)):    # plot node names
    plt.text(nodes[i,0], nodes[i,1], str(temp[i,0]),
             horizontalalignment='center',
             verticalalignment='center',
             fontdict=nodes_font)
            
for i in range(0,size(temp,1)):     # plot all paths
    path = np.asarray([coordenates[x,:] for x in temp[0:-1,i]])
    width = 5*(temp[end+1,i]/max(temp[end+1,:]))
    plt.plot(path[:,0],path[:,1], 
             linestyle = '-', c='yellow', 
             linewidth=width, alpha=0.8, 
             label='paths', zorder=2)
plt.show()


# %% 
######## gridmap with interpolation ###############


np.random.seed(0)
grid = np.random.rand(4, 4)
grid2 = np.random.rand(4, 4)


fig, ax1 = plt.subplots()
ax1.imshow(grid, interpolation='lanczos', cmap='plasma')
ax1.set_title('Grid search a vs b', y=1.05)
plt.xlabel('a')
plt.ylabel('b')


fig, ax2 = plt.subplots()
ax2.imshow(grid2, interpolation='lanczos', cmap='plasma')
ax2.set_title('Grid search c vs b', y=1.05)
plt.xlabel('c')
plt.ylabel('b')

plt.show()







