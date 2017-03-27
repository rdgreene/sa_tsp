# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:27:32 2017

@author: rdgre
"""

def epsilonGreedy(epsilon, s, Q, A):
    
    import numpy as np
    import random
    
    r = np.random.random_sample()
    if r > epsilon:
        max_Q = max(Q[s,A[:, 0]]) # return max value from Q assocaited with all possible actions (A) in current state (s)
        greedy_actions = [i for i, j in enumerate(Q[s, :]) if j == max_Q and i in A[:, 0].tolist()] # create list of possible actions (A) in current state (s) that return the highest Q value
        a = random.choice(greedy_actions) # select amongst possible greedy actions 
    else:
        a = random.choice(A)[0]
    return a