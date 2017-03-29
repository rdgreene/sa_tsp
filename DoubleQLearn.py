# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 16:44:28 2017

@author: miguelesteras and rdgre
"""
# (modified from qLearn function by rdgre)

'''
Need to define:
    transition_seqs [x]
     


'''

def doubleQLearn(epochs, int_R, start, alpha, gamma, epsilon, epsilon_decay, goal_state_reward, max_iters):
    
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
        goal = False #init goal checking variable (when goal==True, goal has been met and loop can max_iters)
        z = 0 # hack to ensure 'if' inside while loop can only be true once
    
        
        visited[s] = 0 # start point marked as visited
        
        transition_seqs.append([s]) # append new list to record transition sequence for current epoch with starting node recorded as 1st element
        
        while (goal == False):
            R[0:-1,s] = np.nan        # the agent cannot go back to the same node
            A = np.argwhere(~np.isnan(R[s,:]))

            doubleQ = Q1+Q2				# using the addition matrix Q1 + Q2 for E-greedy policy
            a = epsilonGreedy(epsilon, s, dobleQ, A)
        
            transition_seqs[i].append(a)	 # record transition made in current iteration
            	
            #define next state based on chosen action (s_nxt) and possible actions from this state (A_nxt)
            s_nxt = a     # assign chosen action (a) to be the next state the agent enters
            A_nxt = np.argwhere(~np.isnan(R[s_nxt,:]))   # create list of available actions from next state (s_nxt)
            
            if rand()<=0.5:

	            # update Q1 with next step reward estimation from Q2 matrix
	            if (sum(visited) < 3):
	                update = Q1[s,a] + alpha * ((R[s,a] +  gamma * Q2[s_nxt,4]) - Q1[s,a]) 				# calculate update to Q matrix value for current state Q[s,a] given next state (s_nxt)
	            else: 
	                update = Q1[s,a] + alpha * ((R[s,a] +  gamma * Q2[s_nxt,np.argmax(Q1[s_nxt, A_nxt])[0]]) - Q1[s,a]) 	# calculate update to Q matrix value for current state Q[s,a] given next state (s_nxt)
	            
	            Q1[s,a] = update # update Q matrix value in current state Q[s,a]
	            #cost += R[s,a]
	            	
	            	
	            # move agent to next state (i.e. update s) and keep record previous state (s_lst)
	            cost += R[s,a] # count cost of state transition associated with action taken in curretn iteration
	            s = s_nxt # the state is updated
	            transitions += 1 # count iterations
	        
	            # update visited and goal variables
	            visited[s] = 0 # the current state is marked as visited (RG: move to top of loop (otherwise counting out of sync???))
	            goal = (s == end or transitions == max_iters)  # check if the end point has been reached (RG: what is the effect of having a max iterations in this part of the loop on how Q gets updated?)    
	        
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
                    update = Q2[s,a] + alpha * ((R[s,a] +  gamma * Q1[s_nxt,np.argmax(Q2[s_nxt, A_nxt])[0]]) - Q2[s,a])     # calculate update to Q matrix value for current state Q[s,a] given next state (s_nxt)
                
                Q2[s,a] = update # update Q matrix value in current state Q[s,a]
                #cost += R[s,a]
                    
                    
                # move agent to next state (i.e. update s) and keep record previous state (s_lst)
                cost += R[s,a] # count cost of state transition associated with action taken in curretn iteration
                s = s_nxt # the state is updated
                transitions += 1 # count iterations
            
                # update visited and goal variables
                visited[s] = 0 # the current state is marked as visited (RG: move to top of loop (otherwise counting out of sync???))
                goal = (s == end or transitions == max_iters)  # check if the end point has been reached (RG: what is the effect of having a max iterations in this part of the loop on how Q gets updated?)    
            
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
