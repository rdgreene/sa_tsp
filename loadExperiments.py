# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:57:56 2017

@author: Ronan
"""

#def diagnosticsPlot(plotData, legendData, title): 
#    
#    import matplotlib.pyplot as plt
#    
#    plt.figure(figsize=(12,8))
#    plt.plot(plotData)
#    plt.title(title, fontsize = 20, style = 'normal', fontname = 'cambria')
#    plt.xlabel('Epochs', fontsize = 17, style = 'italic', fontname = 'cambria')
#    plt.ylabel('Cost Difference vs Optimal Tour', fontsize = 17, style = 'italic', fontname = 'cambria')
#    plt.legend(legendData.values())
#    plt.grid()    
    
#%% Load Experiment Data

import numpy as np
import pickle as pk
from plotdata import *

# Load results
exp1_results = np.load('exp1_results.npy')
exp2_results = np.load('exp2_results.npy')
exp2_results = np.load('exp3_results.npy')
exp3_results = np.load('exp3_results.npy')

# Load Parameters
exp1_parameters = pk.load(open( 'exp1_parameters.p', 'rb' ))
exp2_parameters = pk.load(open( 'exp2_parameters.p', 'rb' ))
exp3_parameters = pk.load(open( 'exp3_parameters.p', 'rb' ))
exp4_parameters = pk.load(open( 'exp4_parameters.p', 'rb' ))


#%% Plot Results for Experiment One

from tspFunctions import *
from plotdata import *
import matplotlib.pyplot as plt

#%% Experiment 1 Results



baseline = 120
plotData = exp1_results
plotData = plotData - baseline
plotData = getWindowAverage(plotData, 50)

legendData = exp1_parameters

title = ['Experiment 1: Q-Learning With Default Parameters\n', 'Exp1 Results']

diagnosticsPlot(plotData, legendData, title, saveFile = True)


#%% Save Figure

plt.savefig('delete')

#%% Save dictionary

# Save a dictionary into a pickle file.
import pickle

pickle.dump( favorite_color, open( "save.p", "wd" ) )

pickle.dump( parameter_records, open( "test.p", "wb" ) )

