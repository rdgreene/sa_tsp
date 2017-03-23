# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 13:40:00 2017

@author: ronan
"""

def loadTSPmatrix(file_name):
    
    import numpy as np
    import pandas as pd
    
    matrix = pd.read_csv(file_name, header=None)
    matrix = matrix.as_matrix().astype('float32')
    
    dimensions = matrix.shape[0]
    
    # changes distances to negative values to represent costs
    for i in range(0, dimensions):
        for j in range(0, dimensions):
            matrix[i, j] = 0 - matrix[i, j] 
    
    # set 'diagonol' to nan
    for i in range(0, dimensions):
        matrix[i, i] = np.nan
    
    # add end row of nan values    
    matrix[:,-1] = np.nan
    
    return matrix
