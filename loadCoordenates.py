#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:25:00 2017

To implement in main script:

from loadCoordenates import loadCoordenates
file_name_xy = 'tsp_matrices/file_name_xy'
coordenates = loadCoordenates(file_name_xy)

"""

file_name = 'tsp_matrices/p01_xy.csv'


def loadCoordenates(file_name_xy):
    
    import pandas as pd
    
    coordenates = (pd.read_csv(file_name_xy, header=None))*100000
    coordenates = coordenates.as_matrix().astype('float32')
    return coordenates
