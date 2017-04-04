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

#%% Define 'Experiments' to run

exp1 = False
exp2 = False
exp3 = False
exp4 = False
exp5a = True
exp5b = False


#%% Load problem and define parameters

distances_file = 'tsp_matrices/toy_d.csv'
optimal_route_file = 'tsp_matrices/toy_s.csv'

int_R, optimal_route, optimal_route_cost =  loadTSPmatrix(distances_file, optimal_route_file)

#%% Set Default Parameters

start = 0
goal_reward = 100
sampling_runs = 100

baseline = optimal_route_cost

# moving average 'smoothing rate'
smooth = 50

#%% Q-Learning Experiment 1: Default parameters

if exp1 == True:

    alphas = [0.7]
    gammas = [0.8]
    epsilons = [1.0]
    epsilon_decays = [0.0005]
    
    epochs = 5000 
    
    title = ['Experiment 1: Q-Learning with Default Parameters\n', 'expResults1']
    
    # run Q-Learning with specified parameters
    mean_costs_matrix, seqs, parameter_records = testParameters(alphas, gammas, epsilons, epsilon_decays, sampling_runs, epochs, int_R, start, goal_reward)
    
    # subtract baseline cost and convert to moving average
    plotData = mean_costs_matrix / baseline
    plotData = getWindowAverage(plotData, smooth)
    legendData = parameter_records
    
    # generate and save plots
    diagnosticsPlot(plotData, legendData, title, saveFile = False)

    # save results
    #np.save('results/expResults1', mean_costs_matrix)
    #pickle.dump(parameter_records, open( "results/expParameters1.p", "wb" ))   


#%% Q-Learning Experiment 2: Vary Learning Rate (Alpha)

if exp2 == True:

    alphas = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    gammas = [0.8]
    epsilons = [1.0]
    epsilon_decays = [0.0005]
    
    epochs = 5000
    
    
    title = ['Experiment 2: Q-Learning with Different Learning Rates\n', 'expResults2']
    
    # run Q-Learning with specified parameters
    mean_costs_matrix, seqs, parameter_records = testParameters(alphas, gammas, epsilons, epsilon_decays, sampling_runs, epochs, int_R, start, goal_reward)
    
    # subtract baseline cost and convert to moving average
    plotData = mean_costs_matrix / baseline
    plotData = getWindowAverage(plotData, smooth)
    legendData = parameter_records
    
    # generate and save plots
    diagnosticsPlot(plotData, legendData, title, saveFile = False)

    # save results
    #np.save('results/expResults2', mean_costs_matrix)
    #pickle.dump(parameter_records, open( "results/expParameters2.p", "wb" ))  


#%% Q-Learning Experiment 3: Vary Gamma

if exp3 == True:

    alphas = [0.7]
    gammas = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    epsilons = [1.0]
    epsilon_decays = [0.0005]
    
    epochs = 5000
    
    
    title = ['Experiment 3: Q-Learning with Different Discount Factors\n', 'expResults3']
    
    # run Q-Learning with specified parameters
    mean_costs_matrix, seqs, parameter_records = testParameters(alphas, gammas, epsilons, epsilon_decays, sampling_runs, epochs, int_R, start, goal_reward)
    
    # subtract baseline cost and convert to moving average
    plotData = mean_costs_matrix / baseline
    plotData = getWindowAverage(plotData, smooth)
    legendData = parameter_records
    
    # generate and save plots
    diagnosticsPlot(plotData, legendData, title, saveFile = False)

    # save results
    #np.save('results/expResults3', mean_costs_matrix)
    #pickle.dump(parameter_records, open( "results/expParameters3.p", "wb" ))  


#%% Q-Learning Experiment 4: Vary Epsilon Decay

if exp4 == True:

    alphas = [0.7]
    gammas = [0.8]
    epsilons = [1.0]
    epsilon_decays = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.01, 0.3]
    
    epochs = 10000
    
    
    title = ['Experiment 4: Q-Learning with Different Decay Rates\n', 'expResults4']
    
    # run Q-Learning with specified parameters
    mean_costs_matrix, seqs, parameter_records = testParameters(alphas, gammas, epsilons, epsilon_decays, sampling_runs, epochs, int_R, start, goal_reward)
    
    # subtract baseline cost and convert to moving average
    plotData = mean_costs_matrix / baseline
    plotData = getWindowAverage(plotData, smooth)
    legendData = parameter_records
    
    # generate and save plots
    diagnosticsPlot(plotData, legendData, title, saveFile = False)

    # save results
    np.save('results/expResults4', mean_costs_matrix)
    pickle.dump( parameter_records, open("results/expParameters4.p", "wb" ))                    


#%% Q-Learning Experiment 5a: Optimise Parameters (learning rate)

if exp5a == True:

    alphas = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    gammas = [0.8]
    epsilons = [1.0]
    epsilon_decays = [0.3]
    
    epochs = 5000
    
    
    title = ['Experiment 5a: Optimise Parameters (Learning Rate)\n', 'expResults5']
    
    # run Q-Learning with specified parameters
    mean_costs_matrix, seqs, parameter_records = testParameters(alphas, gammas, epsilons, epsilon_decays, sampling_runs, epochs, int_R, start, goal_reward)
    
    # subtract baseline cost and convert to moving average
    plotData = mean_costs_matrix / baseline
    plotData = getWindowAverage(plotData, smooth)
    legendData = parameter_records
    
    # generate and save plots
    diagnosticsPlot(plotData, legendData, title, saveFile = False)

    # save results
    np.save('results/expResults5a', mean_costs_matrix)
    pickle.dump( parameter_records, open("results/expParameters5a.p", "wb" )) 

    
#%% Q-Learning Experiment 5b: Optimise Parameters (discount factor)

if exp5b == True:

    alphas = [0.1]
    gammas = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    epsilons = [1.0]
    epsilon_decays = [0.3]
    
    epochs = 5000
    
    
    title = ['Experiment 5b: Optimise Parameters (Discount Factor)\n', 'expResults5b']
    
    # run Q-Learning with specified parameters
    mean_costs_matrix, seqs, parameter_records = testParameters(alphas, gammas, epsilons, epsilon_decays, sampling_runs, epochs, int_R, start, goal_reward)
    
    # subtract baseline cost and convert to moving average
    plotData = mean_costs_matrix / baseline
    plotData = getWindowAverage(plotData, smooth)
    legendData = parameter_records
    
    # generate and save plots
    diagnosticsPlot(plotData, legendData, title, saveFile = False)

    # save results
    np.save('results/expResults5b', mean_costs_matrix)
    pickle.dump( parameter_records, open("results/expParameters5b.p", "wb" ))     
    
      
#%% Clear Redundant Variables from workspace

''' KEEP AT END OF SCRIPT'''

# clear input variables
#del start, epochs, sampling_runs, goal_reward

# clear any variables created solely for 'looping' purposes
#del file_name








