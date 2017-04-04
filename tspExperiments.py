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

distances_file = 'tsp_matrices/toy_d.csv'
optimal_route_file = 'tsp_matrices/toy_s.csv'

int_R, optimal_route, optimal_route_cost =  loadTSPmatrix(distances_file, optimal_route_file)

#%% Set Default Parameters

start = 0
epochs = 5000 # init epochs count
goal_reward = 100
sampling_runs = 100

alphas = [0.7]
gammas = [0.8]
epsilons = [1.0]
epsilon_decays = [0.0005]

baseline = 120

# moving average 'smoothing rate'
smooth = 20

#%% Q-Learning Experiment 1: Default parameters

alphas = [0.7]
gammas = [0.8]
epsilons = [1.0]
epsilon_decays = [0.0005]

#start = 0
#epochs = 5000 # init epochs count
#goal_reward = 100
#sampling_runs = 50

title = ['Experiment 1: Q-Learning with Default Parameters\n', 'expResults1']

# run Q-Learning with specified parameters
mean_costs_matrix, seqs, parameter_records = testParameters(alphas, gammas, epsilons, epsilon_decays, sampling_runs, epochs, int_R, start, goal_reward)

# subtract baseline cost and convert to moving average
plotData = mean_costs_matrix - baseline
plotData = getWindowAverage(plotData, smooth)
legendData = parameter_records

# generate and save plots
diagnosticsPlot(plotData, legendData, title, saveFile = True)
#plotLines(window_ave,alphas, 120, title, True)
#plt.figure(figsize=(12,8))
#plt.plot(window_ave)
#plt.legend(parameter_records.values())
#plt.title(title)
#plt.savefig(title)

# save results
np.save('expResults1', mean_costs_matrix)
pickle.dump(parameter_records, open( "expParameters1.p", "wb" ))   


#%% Q-Learning Experiment 2: Vary Learning Rate (Alpha)

alphas = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
gammas = [0.8]
epsilons = [1.0]
epsilon_decays = [0.0005]

#start = 0
#epochs = 5000
#goal_reward = 100
#sampling_runs = 100

title = ['Experiment 2: Q-Learning with Different Learning Rates\n', 'expResults2']

# run Q-Learning with specified parameters
mean_costs_matrix, seqs, parameter_records = testParameters(alphas, gammas, epsilons, epsilon_decays, sampling_runs, epochs, int_R, start, goal_reward)

# subtract baseline cost and convert to moving average
plotData = mean_costs_matrix - baseline
plotData = getWindowAverage(plotData, smooth)
legendData = parameter_records

# generate and save plots
diagnosticsPlot(plotData, legendData, title, saveFile = True)
#plotLines(window_ave, alphas, 120, title, True)
#plt.figure(figsize=(12,8))
#plt.plot(window_ave)
#plt.legend(parameter_records.values())
#plt.title(title)
#plt.savefig(title)

# save results
np.save('expResults2', mean_costs_matrix)
pickle.dump(parameter_records, open( "expParameters2.p", "wb" ))  


#%% Q-Learning Experiment 3: Vary Gamma

alphas = [0.7]
gammas = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
epsilons = [1.0]
epsilon_decays = [0.0005]

#start = 0
#epochs = 5000
#goal_reward = 100
#sampling_runs = 50

title = ['Experiment 3: Q-Learning with Different Discount Factors\n', 'expResults3']

# run Q-Learning with specified parameters
mean_costs_matrix, seqs, parameter_records = testParameters(alphas, gammas, epsilons, epsilon_decays, sampling_runs, epochs, int_R, start, goal_reward)

# subtract baseline cost and convert to moving average
plotData = mean_costs_matrix - baseline
plotData = getWindowAverage(plotData, smooth)
legendData = parameter_records

# generate and save plots
diagnosticsPlot(plotData, legendData, title, saveFile = True)
#plotLines(window_ave, alphas, 120, title, True)
#plt.figure(figsize=(12,8))
#plt.plot(window_ave)
#plt.legend(parameter_records.values())
#plt.title(title)
#plt.savefig(title)

# save results
np.save('expResults3', mean_costs_matrix)
pickle.dump(parameter_records, open( "expParameters3.p", "wb" ))  


#%% Q-Learning Experiment 4: Vary Epsilon Decay

alphas = [0.7]
gammas = [0.8]
epsilons = [1.0]
epsilon_decays = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]

#start = 0
epochs = 10000
#goal_reward = 100
#sampling_runs = 50

title = ['Experiment 4: Q-Learning with Different Decay Rates\n', 'expResults4']

# run Q-Learning with specified parameters
mean_costs_matrix, seqs, parameter_records = testParameters(alphas, gammas, epsilons, epsilon_decays, sampling_runs, epochs, int_R, start, goal_reward)

# subtract baseline cost and convert to moving average
plotData = mean_costs_matrix - baseline
plotData = getWindowAverage(plotData, smooth)
legendData = parameter_records

# generate and save plots
diagnosticsPlot(plotData, legendData, title, saveFile = True)
#plotLines(window_ave, alphas, 120, title, True)
#plt.figure(figsize=(12,8))
#plt.plot(window_ave)
#plt.legend(parameter_records.values())
#plt.title(title)
#plt.savefig(title)

# save results
np.save('expResults4', mean_costs_matrix)
pickle.dump( parameter_records, open("expParameters4.p", "wb" ))                    


#%% Clear Redundant Variables from workspace

''' KEEP AT END OF SCRIPT'''

# clear input variables
del start, epochs, sampling_runs, goal_reward

# clear any variables created solely for 'looping' purposes
del file_name

# clear non-aggregate metrics variables
#del euler_gamma, pi






