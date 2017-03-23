# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 13:40:00 2017

@author: ronan


Use following to implement in main script:

from loadTSPmatrix import loadTSPmatrix
file_name = 'filename and directory here'
TPSmatrix = loadTSPmatrix(file_name)

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
            
    # add additional colum and row to array for returning to start state
    row = np.array([0]*dimensions)
    col = np.array([0]*(dimensions+1))
    matrix = np.row_stack((matrix,row)) 
    matrix = np.column_stack((matrix,col)) 
    
    # set 'diagonol' to nan
    for i in range(0, dimensions):
        matrix[i, i] = np.nan
    
    # add end row of nan values    
    matrix[:,-1] = np.nan
    
    # set end column to nan
    matrix[:,-1] = np.nan

    # set end row to nan (with exception of first element)
    matrix[-1, 1:-1] = np.nan
    
    return matrix
