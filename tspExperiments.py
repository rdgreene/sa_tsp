# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:10:03 2017

@author: miguel & ronan

"""

#%% Import libraries & functions

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from tspFunctions import *
from plotdata import *
import pickle

#%% Load problem and define parameters

file_name = 'tsp_matrices/toy_d.csv'
int_R = loadTSPmatrix(file_name)

#%% Q-Learning Experiment 1: Default parameters

alphas = [0.7]
gammas = [0.8]
epsilons = [1.0]
epsilon_decays = [0.0005]

start = 0
epochs = 1000 # init epochs count
goal_reward = 100
sampling_runs = 1

title = 'Experiment 1'

# run Q-Learning with specified parameters
mean_costs_matrix, seqs, parameter_records = testParameters(alphas, gammas, epsilons, epsilon_decays, sampling_runs, epochs, int_R, start, goal_reward)

# plot results
window_ave = getWindowAverage(mean_costs_matrix, 50)
#plotLines(window_ave,alphas, 120, title, True)
plt.figure(figsize=(12,8))
plt.plot(window_ave)
plt.legend(parameter_records.values())
plt.title(title)
plt.savefig(title)

# save results
#np.save('exp1_results', mean_costs_matrix)
#np.save('exp1_parameters', parameter_records)        

#%% Q-Learning Experiment 2: Vary Learning Rate (Alpha)

alphas = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
gammas = [0.8]
epsilons = [1.0]
epsilon_decays = [0.0005]

start = 0
epochs = 5000 # init epochs count
goal_reward = 100
sampling_runs = 100

title = 'Experiment 2'

# run Q-Learning with specified parameters
mean_costs_matrix, seqs, parameter_records = testParameters(alphas, gammas, epsilons, epsilon_decays, sampling_runs, epochs, int_R, start, goal_reward)

# plot results
window_ave = getWindowAverage(mean_costs_matrix, 20)
#plotLines(window_ave, alphas, 120, title, True)
plt.figure(figsize=(12,8))
plt.plot(window_ave)
plt.legend(parameter_records.values())
plt.title(title)
plt.savefig(title)

# save results
np.save('exp2_results', mean_costs_matrix)
np.save('exp2_parameters', parameter_records)

#%% Q-Learning Experiment 3: Vary Gamma

alphas = [0.7]
gammas = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
epsilons = [1.0]
epsilon_decays = [0.0005]

start = 0
epochs = 5000 # init epochs count
goal_reward = 100
sampling_runs = 50

title = 'Experiment 3'

# run Q-Learning with specified parameters
mean_costs_matrix, seqs, parameter_records = testParameters(alphas, gammas, epsilons, epsilon_decays, sampling_runs, epochs, int_R, start, goal_reward)

# plot results
window_ave = getWindowAverage(mean_costs_matrix, 20)
#plotLines(window_ave, alphas, 120, title, True)
plt.figure(figsize=(12,8))
plt.plot(window_ave)
plt.legend(parameter_records.values())
plt.title(title)
plt.savefig(title)

# save results
np.save('exp3_results', mean_costs_matrix)
np.save('exp3_parameters', parameter_records)

#%% Q-Learning Experiment 4: Vary Epsilon Decay

alphas = [0.7]
gammas = [0.8]
epsilons = [1.0]
epsilon_decays = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]

start = 0
epochs = 10000 # init epochs count
goal_reward = 100
sampling_runs = 50

title = 'Experiment 4'

# run Q-Learning with specified parameters
mean_costs_matrix, seqs, parameter_records = testParameters(alphas, gammas, epsilons, epsilon_decays, sampling_runs, epochs, int_R, start, goal_reward)

# plot results
window_ave = getWindowAverage(mean_costs_matrix, 50)
#plotLines(window_ave, alphas, 120, title, True)
plt.figure(figsize=(12,8))
plt.plot(window_ave)
plt.legend(parameter_records.values())
plt.title(title)
plt.savefig(title)

# save results
np.save('exp4_results', mean_costs_matrix)
np.save('exp4_parameters', parameter_records)                      

#%% Clear Redundant Variables from workspace

''' KEEP AT END OF SCRIPT'''

# clear input variables
del start, epochs, sampling_runs, goal_reward

# clear any variables created solely for 'looping' purposes
del file_name

# clear non-aggregate metrics variables
#del euler_gamma, pi






