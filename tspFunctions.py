# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 18:00:36 2017

@author: Ronan
"""

''' e-greedy policy'''

import numpy as np
import random
from copy import deepcopy
import pandas as pd

def epsilonGreedy(epsilon, s, Q, A):
    
    #import numpy as np
    #import random
    
    r = np.random.random_sample()
    if r > epsilon:
        max_Q = max(Q[s,A[:, 0]]) # return max value from Q assocaited with all possible actions (A) in current state (s)
        greedy_actions = [i for i, j in enumerate(Q[s, :]) if j == max_Q and i in A[:, 0].tolist()] # create list of possible actions (A) in current state (s) that return the highest Q value
        a = random.choice(greedy_actions) # select amongst possible greedy actions 
    else:
        a = random.choice(A)[0]
    return a
    
    
'''Q-Learning'''

def qLearn(epochs, int_R, start, alpha, gamma, epsilon, epsilon_decay, goal_state_reward):
    
    # import function dependencies
    #from copy import deepcopy
    #import numpy as np
    #from epsilonGreedy import epsilonGreedy

    # init in loop NEW
    transition_seqs = []
    total_cost = []
    Q = np.zeros_like(int_R)
    end = np.size(int_R,1)-1
    epoch = 0
    
    for i in range(0,epochs): 
    # re-initialise variables
        R = deepcopy(int_R)  # reset the R matrix
        s = start # initial state
        visited = np.ones(np.size(R,1)) # init list to track visited states (1==univisted, 0==visited)
        transitions = 0 # init step transitions
        cost = 0 # init cost transitions
        goal = False #init goal checking variable (when goal==True, goal has been met)
        z = 0 # hack to ensure 'if' inside while loop can only be true once
    
        
        visited[s] = 0 # start point marked as visited
        
        transition_seqs.append([s]) # append new list to record transition sequence for current epoch with starting node recorded as 1st element
        
        while (goal == False):
            R[0:-1,s] = np.nan        # the agent cannot go back to the same node
            A = np.argwhere(~np.isnan(R[s,:]))
            a = epsilonGreedy(epsilon, s, Q, A)
        
            transition_seqs[i].append(a)	 # record transition made in current iteration
            	
            #define next state based on chosen action (s_nxt) and possible actions from this state (A_nxt)
            s_nxt = a     # assign chosen action (a) to be the next state the agent enters
            A_nxt = np.argwhere(~np.isnan(R[s_nxt,:]))   # create list of available actions from next state (s_nxt)
            # update Q matrix
            if (sum(visited) < 3):
                update = Q[s,a] + alpha * ((R[s,a] +  gamma * Q[s_nxt,4]) - Q[s,a]) # calculate update to Q matrix value for current state Q[s,a] given next state (s_nxt)
            else: 
                update = Q[s,a] + alpha * ((R[s,a] +  gamma * max(Q[s_nxt, A_nxt])[0]) - Q[s,a]) # calculate update to Q matrix value for current state Q[s,a] given next state (s_nxt)
            
            Q[s,a] = update # update Q matrix value in current state Q[s,a]
            #cost += R[s,a]
            	
            	
            # move agent to next state (i.e. update s) and keep record previous state (s_lst)
            cost += R[s,a] # count cost of state transition associated with action taken in curretn iteration
            s = s_nxt # the state is updated
            transitions += 1 # count iterations
        
            # update visited and goal variables
            visited[s] = 0 # the current state is marked as visited (RG: move to top of loop (otherwise counting out of sync???))
            goal = (s == end)   # check if the end point has been reached
        
            # ADD: increment operation to count steps
        
            if ((sum(visited) == 1 and z == 0)): # if all nodes have been visited
                R[:,-1] = int_R[:,0] + goal_state_reward # Alternative, reward & cost into R matrix
                R[:,0] = np.nan  # the agent cannot go to start node
                z += 1;
                
        
            #decay epsilon
            epsilon += -epsilon_decay*epsilon
            
        # record metrics for this epoch
        cost += -goal_state_reward # subtract reward for returning to start node from cost
        total_cost.append(np.absolute(cost)) # convert cost value to absolute
        #average_cost.append(sum(total_cost)/len(total_cost)) RG: redundant?
        epoch +=1
        #epsilon_decay.append(epsilon) RG: redundant?
    
    return total_cost, transition_seqs, Q
    


''' Double Q-learning'''

def doubleQLearn(epochs, int_R, start, alpha, gamma, epsilon, epsilon_decay, goal_state_reward):
    
    # import function dependencies
    from copy import deepcopy
    import numpy as np
    from epsilonGreedy import epsilonGreedy

    # init in loop NEW
    transition_seqs = []
    total_cost = []
    Q1 = np.zeros_like(int_R)
    Q2 = np.zeros_like(int_R)
    end = np.size(int_R,1)-1
    epoch = 0
    
    for i in range(0,epochs): 
    # re-initialise variables
        R = deepcopy(int_R)  # reset the R matrix
        s = start # initial state
        visited = np.ones(np.size(R,1)) # init list to track visited states (1==univisted, 0==visited)
        transitions = 0 # init step transitions
        cost = 0 # init cost transitions
        goal = False #init goal checking variable (when goal==True, goal has been met)
        z = 0 # hack to ensure 'if' inside while loop can only be true once
    
        
        visited[s] = 0 # start point marked as visited
        
        transition_seqs.append([s]) # append new list to record transition sequence for current epoch with starting node recorded as 1st element
        
        while (goal == False):
            R[0:-1,s] = np.nan        # the agent cannot go back to the same node
            A = np.argwhere(~np.isnan(R[s,:]))

            doubleQ = Q1+Q2             # using the addition matrix Q1 + Q2 for E-greedy policy
            a = epsilonGreedy(epsilon, s, doubleQ, A)
        
            transition_seqs[i].append(a)     # record transition made in current iteration
                
            #define next state based on chosen action (s_nxt) and possible actions from this state (A_nxt)
            s_nxt = a     # assign chosen action (a) to be the next state the agent enters
            A_nxt = np.argwhere(~np.isnan(R[s_nxt,:]))   # create list of available actions from next state (s_nxt)
            
            if np.random.rand()<=0.5:

                # update Q1 with next step reward estimation from Q2 matrix
                if (sum(visited) < 3):
                    update = Q1[s,a] + alpha * ((R[s,a] +  gamma * Q2[s_nxt,4]) - Q1[s,a])              # calculate update to Q matrix value for current state Q[s,a] given next state (s_nxt)
                else: 
                    update = Q1[s,a] + alpha * ((R[s,a] +  gamma * Q2[s_nxt,np.argmax(Q1[s_nxt, A_nxt])]) - Q1[s,a])     # calculate update to Q matrix value for current state Q[s,a] given next state (s_nxt)
                
                Q1[s,a] = update # update Q matrix value in current state Q[s,a]
                #cost += R[s,a]
                    
                    
                # move agent to next state (i.e. update s) and keep record previous state (s_lst)
                cost += R[s,a] # count cost of state transition associated with action taken in curretn iteration
                s = s_nxt # the state is updated
                transitions += 1 # count iterations
            
                # update visited and goal variables
                visited[s] = 0 # the current state is marked as visited (RG: move to top of loop (otherwise counting out of sync???))
                goal = (s == end)  # check if the end point has been reached (RG: what is the effect of having a max iterations in this part of the loop on how Q gets updated?)    
            
                # ADD: increment operation to count steps
            
                if ((sum(visited) == 1 and z == 0)): # if all nodes have been visited
                    R[:,-1] = int_R[:,0] + goal_state_reward # Alternative, reward & cost into R matrix
                    R[:,0] = np.nan  # the agent cannot go to start node
                    z += 1;                 
           
                #decay epsilon
                epsilon += -epsilon_decay*epsilon

            else:

                # update Q2 with next step reward estimation from Q1 matrix
                if (sum(visited) < 3):
                    update = Q2[s,a] + alpha * ((R[s,a] +  gamma * Q1[s_nxt,4]) - Q2[s,a])              # calculate update to Q matrix value for current state Q[s,a] given next state (s_nxt)
                else: 
                    update = Q2[s,a] + alpha * ((R[s,a] +  gamma * Q1[s_nxt,np.argmax(Q2[s_nxt, A_nxt])]) - Q2[s,a])     # calculate update to Q matrix value for current state Q[s,a] given next state (s_nxt)
                
                Q2[s,a] = update # update Q matrix value in current state Q[s,a]
                #cost += R[s,a]
                    
                    
                # move agent to next state (i.e. update s) and keep record previous state (s_lst)
                cost += R[s,a] # count cost of state transition associated with action taken in curretn iteration
                s = s_nxt # the state is updated
                transitions += 1 # count iterations
            
                # update visited and goal variables
                visited[s] = 0 # the current state is marked as visited (RG: move to top of loop (otherwise counting out of sync???))
                goal = (s == end)  # check if the end point has been reached (RG: what is the effect of having a max iterations in this part of the loop on how Q gets updated?)    
            
                # ADD: increment operation to count steps
            
                if ((sum(visited) == 1 and z == 0)): # if all nodes have been visited
                    R[:,-1] = int_R[:,0] + goal_state_reward # Alternative, reward & cost into R matrix
                    R[:,0] = np.nan  # the agent cannot go to start node
                    z += 1;                    
            
                #decay epsilon
                epsilon += -epsilon_decay*epsilon

        # record metrics for this epoch
        cost += -goal_state_reward # subtract reward for returning to start node from cost
        total_cost.append(np.absolute(cost)) # convert cost value to absolute
        #average_cost.append(sum(total_cost)/len(total_cost)) RG: redundant?
        epoch +=1
        #epsilon_decay.append(epsilon) RG: redundant?
    
    return total_cost, transition_seqs, Q1, Q2




''' Load TSP Problems Matrix'''
    
#def loadTSPmatrix(file_name):
#    
#    #import numpy as np
#    #import pandas as pd
#    
#    matrix = pd.read_csv(file_name, header=None)
#    matrix = matrix.as_matrix().astype('float32')
#    
#    dimensions = matrix.shape[0]
#    
#    # changes distances to negative values to represent costs
#    for i in range(0, dimensions):
#        for j in range(0, dimensions):
#            matrix[i, j] = 0 - matrix[i, j] 
#            
#    # add additional colum and row to array for returning to start state
#    row = np.array([0]*dimensions)
#    col = np.array([0]*(dimensions+1))
#    matrix = np.row_stack((matrix,row)) 
#    matrix = np.column_stack((matrix,col)) 
#    
#    # set 'diagonol' to nan
#    for i in range(0, dimensions):
#        matrix[i, i] = np.nan
#    
#    # add end row of nan values    
#    matrix[:,-1] = np.nan
#    
#    # set end column to nan
#    matrix[:,-1] = np.nan
#
#    # set end row to nan (with exception of first element)
#    matrix[-1, 1:-1] = np.nan
#    
#    return matrix
    
def loadTSPmatrix(distances_file, optimal_route_file):

    # LOAD DISTANCES AND CREATE R MATRIX
    R_matrix = pd.read_csv(distances_file, header=None)
    R_matrix = R_matrix.as_matrix().astype('float32')
    
    dimensions = R_matrix.shape[0]
    
    # changes distances to negative values to represent costs
    for i in range(0, dimensions):
        for j in range(0, dimensions):
            R_matrix[i, j] = 0 - R_matrix[i, j] 
            
    # add additional colum and row to array for returning to start state
    row = np.array([0]*dimensions)
    col = np.array([0]*(dimensions+1))
    R_matrix = np.row_stack((R_matrix,row)) 
    R_matrix = np.column_stack((R_matrix,col)) 
    
    # set 'diagonol' to nan
    for i in range(0, dimensions):
        R_matrix[i, i] = np.nan
    
    # add end row of nan values    
    R_matrix[:,-1] = np.nan
    
    # set end column to nan
    R_matrix[:,-1] = np.nan
    
    # set end row to nan (with exception of first element)
    R_matrix[-1, 1:-1] = np.nan
    
    # LOAD OPTIMAL ROUTE AND CALCULATE COST
    optimal_route_df = pd.read_csv(optimal_route_file, header=None)
    
    # convert to route indexing from zero!
    optimal_route_df = optimal_route_df - 1
    
    optimal_route = []
    tour_cost = 0
    for i in range(0, len(optimal_route_df)):
        optimal_route.append(int(optimal_route_df[0][i]))
    
    for i in range(0, len(optimal_route)-1):
        
        # from
        from_node = optimal_route[i]
        # to
        to_node = optimal_route[i+1]
        
        # cost of single move from ith node to jth node
        trip_cost = R_matrix[from_node, to_node]
        
        # update cummulative cost of tour
        tour_cost += trip_cost
        
        # prints optimal route and costs step by step 
        #print('from %s to %s, trip cost = %d, total tour cost = %d' % (from_node, to_node, trip_cost, tour_cost))
        
    tour_cost = abs(tour_cost)
    
    return R_matrix, optimal_route, tour_cost  




''' Return Window Average'''    

def getWindowAverage(cost_matrix, n):
    window_ave = np.zeros_like(cost_matrix)
    for k in range(0,int(np.size(cost_matrix,1))):
        for i in range(1,int(np.size(cost_matrix[:,k])+1)):
            if i<n-1:
                window_ave[i-1,k] = (np.mean(cost_matrix[:,k][0:i]))
            else:
                window_ave[i-1,k] = (np.mean(cost_matrix[:,k][i-(n-1):i]))
                
    return window_ave
    
    
    
''' Test Parameters'''

def testParameters(alphas, gammas, epsilons, epsilon_decays,  sampling_runs, epochs, int_R, start, goal_state_reward):
    
    alphas = np.array(alphas).astype('float32')
    gammas = np.array(gammas).astype('float32')
    epsilons = np.array(epsilons).astype('float32')
    epsilon_decays = np.array(epsilon_decays).astype('float32') 
    
    seqs = [] # list of lists, where each inner list records the state transitions made in an epoch. State transitions for each epoch in each sampling_run are recorded.
    costs_matrix = np.zeros((epochs, sampling_runs)) # matrix containing epoch cost vectors for each sampling sampling_run
    mean_costs_matrix = np.zeros((epochs, np.size(alphas)*np.size(gammas)*np.size(epsilons)*np.size(epsilon_decays))) # contains mean of costs for all sampling_runs of each parameter setting
    ps_dic = {} # init; parameter search dictionary
    loop_idx = 0 # init
    
    
    for a in range(0, np.size(alphas)):
        for g in range(0, np.size(gammas)):
            for e in range(0, np.size(epsilons)):
                for d in range(0, np.size(epsilon_decays)):
                    
                    alpha = alphas[a]; gamma = gammas[g]; epsilon = epsilons[e]; epsilon_decay = epsilon_decays[d]
                    print('\nRunning Variant %s [Alpha: %.4f, Gamma: %.4f, Epsilon: %.4f, Decay: %.4f]...' % (loop_idx+1, alpha, gamma, epsilon, epsilon_decay))
    
                    for sampling_run in range (0, sampling_runs):
                    
                        epoch_costs, trans_seqs, _ = qLearn(epochs, int_R, start, alpha, gamma, epsilon, epsilon_decay, goal_state_reward)
                        
                        # record all sequences followed in current sampling_run
                        for epoch in range(0, epochs):
                            seqs.append(trans_seqs[epoch])
                            
                        # add vector of epoch costs to costs_matrix
                        costs_matrix[:, sampling_run] = epoch_costs
                        
                        # calculate mean costs across all 'sampling_runs'
                        mean_costs = np.mean(costs_matrix, 1)
                     
                    ps_dic[loop_idx] = (r'$\alpha = %.4f$, $\gamma = %.4f$, $\epsilon = %.4f$, $\lambda = %.4f$' % (alpha, gamma, epsilon, epsilon_decay)) 
                    #ps_dic[loop_idx] = ('A [%.2f], G [%.2f], E [%.2f], D [%.4f]' % (alpha, gamma, epsilon, epsilon_decay))   
                    mean_costs_matrix[:, loop_idx] = mean_costs; loop_idx +=1
                    
    return mean_costs_matrix, seqs, ps_dic