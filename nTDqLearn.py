# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 16:44:28 2017

@author: miguelesteras and rdgre
"""
# (modified from qLearn code by rdgre)
'''
Need to define:
    transition_seqs [x]
     


'''

def nTDqLearn(epochs, int_R, start, alpha, gamma, epsilon, epsilon_decay, goal_state_reward, max_iters, n):
    
    # import function dependencies
    from copy import deepcopy
    import numpy as np
    from epsilonGreedy import epsilonGreedy

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
        goal = False #init goal checking variable (when goal==True, goal has been met and loop can max_iters)
        z = 0 # hack to ensure 'if' inside while loop can only be true once
    
        
        visited[s] = 0 # start point marked as visited
        
        transition_seqs.append([s]) # append new list to record transition sequence for current epoch with starting node recorded as 1st element
        
        while (goal == False):
            R[0:-1,s] = np.nan                  # the agent cannot go back to the same node
            
            for j in range(0,n):

                A[j,:] = np.argwhere(~np.isnan(R[s[-1,0],:]))
                a[j,0] = epsilonGreedy(epsilon, s[-1,0], Q, A[j,:])
            
                transition_seqs[i].append(a[j,0])	    # record transition made in current iteration
                	
                #define next state based on chosen action (s) and possible actions from this state (A)
                s[j+1,0] = a[j,0]                           # assign chosen action (a) to be the next state the agent enters
                A[j+1,:] = np.argwhere(~np.isnan(R[s[j+1,0],:]))  # create list of available actions from next state (s)

            # update Q matrix
            if (sum(visited) < 3):
                update = Q[s,a] + alpha * ((R[s,a] +  gamma * Q[s_nxt,4]) - Q[s,a]) # calculate update to Q matrix value for current state Q[s,a] given next state (s_nxt)
            else: 
                update = Q[s,a] + alpha * ((R[s,a] +  gamma * max(Q[s_nxt, A_nxt])[0]) - Q[s,a]) # calculate update to Q matrix value for current state Q[s,a] given next state (s_nxt)
            
            Q[s,a] = update # update Q matrix value in current state Q[s,a]            	
            cost += R[s,a] # count cost of state transition associated with action taken in curretn iteration
            
            transitions += n # count iterations
        
            # update visited and goal variables
            visited[s] = 0 # visited states marked as visited (RG: move to top of loop (otherwise counting out of sync???))
            goal = (s[-1,0] == end or transitions == max_iters)  # check if the end point has been reached (RG: what is the effect of having a max iterations in this part of the loop on how Q gets updated?)    
        
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
