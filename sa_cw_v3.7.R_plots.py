#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:42:50 2017

@author: miguelesteras
"""

'''Software Agents Code Development'''

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:36:45 2017

@author: Ronan
"""

"""
Ronan and Miguel - Software Agents - Spring 2017
"""

#%% import library dependancies 

import numpy as np
from scipy import *
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import seaborn as sns
from ggplot import *

#%% define functions

# np.random.random_sample()

def epsilonGreedy(epsilon, s, Q, A):
	r = np.random.random_sample()
	if r > epsilon:
		max_Q = max(Q[s,A[:, 0]]) # return max value from Q assocaited with all possible actions (A) in current state (s)
		greedy_actions = [i for i, j in enumerate(Q[s, :]) if j == max_Q and i in A[:, 0].tolist()] # create list of possible actions (A) in current state (s) that return the highest Q value
		a = random.choice(greedy_actions) # select amongst possible greedy actions 
	else:
		a = random.choice(A)[0]
	return a

#%% initialise hyper-parameters            

alpha = 0.8; 
gamma = 0.8; 
epsilon = 0.8; 
max_state_transitions = 100 # max iterations
goal_state_reward = 100
epochs = 5000

# ans = input("Use default values for alpha(%s), gamma(%s), epsilon(%s), max transitions per epoch (%s), and goal state reward (%s)? [y/n]"\
 # % (alpha, gamma, epsilon, max_state_transitions, goal_state_reward)) #Define Learning Hyper Parameters
        
# if ans != "y":     
	# alpha = float(input("alpha value (0-1): ")) 
	# gamma = float(input("gamma value (0-1): "))  
	# epsilon = float(input("epsilon value (0-1): ")) 
	# max_state_transitions = int(input("max state transitions per epoch (0-1): "))
	# goal_state_reward = int(input("reward for reaching goal state: "))	


#%% init Q-Learning objects

"""Q-learning"""

int_R = np.array([[nan, -10, -50, -45, nan], # R matrix [nodes = start, 1, 2, 3, end]
              [-10, nan, -25, -25, nan],
              [-50, -25, nan, -40, nan],
              [-45, -25, -40, nan, nan],
              [0, nan, nan, nan, nan]]).astype("float32") # why float32 ?

Q = np.zeros_like(int_R) # initialize Q matrix
epoch = 0 # init epochs count
start = 0 # Define start point at row 0 (RG: moved outside loop)
end = size(int_R,1)-1 # Define end point (same as start but located in the last column in the R/Q matrix) (RG: moved outside loop)

# initialise tracking variables
total_transitions = [] # records number of steps in each inner loop (i.e. each epoch)
total_cost = [] # records cost of steps in each inner loop (i.e. each epoch)
epsilon_decay = []
average_cost = []

#%% Start Q-Learning

for i in range(0,epochs): 
    # re-initialise variables
    R = deepcopy(int_R)  # reset the R matrix
    s = start # initial state
    visited = np.ones(size(R,1)) # init list to track visited states (1==univisted, 0==visited)
    transitions = 0 # init step transitions
    cost = 0 # init cost transitions
    goal = False #init goal checking variable (when goal==True, goal has been met and loop can max_state_transitions)
    z = 0 # hack to ensure 'if' inside while loop can only be true once
    force_random = 0
    
    visited[s] = 0 # start point marked as visited


    
    while (goal == False):
        R[0:-1,s] = np.nan        # the agent cannot go back to the same node
        A = np.argwhere(~np.isnan(R[s,:]))
        a = epsilonGreedy(epsilon, s, Q, A)
    
        	
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
        goal = (s == end or transitions == max_state_transitions)  # check if the end point has been reached (RG: what is the effect of having a max iterations in this part of the loop on how Q gets updated?)    
    
        # ADD: increment operation to count steps
    
        if ((sum(visited) == 1 and z == 0)): # if all nodes have been visited
            R[:,-1] = int_R[:,0] + goal_state_reward # Alternative, reward & cost into R matrix
            R[:,0] = np.nan  # the agent cannot go to start node
            z += 1;
            
    
        #decay epsilon
        epsilon += -0.0001*epsilon
        	
        # check if agent is doubling back! (old position - saved for record)
        # if s_nxt == s_lst:
        # force_random += 1
        # else:
        # force_random = 0

    # record metrics for this epoch
    total_transitions.append(transitions)
    cost +=- goal_state_reward # subtract reward 
    total_cost.append(np.absolute(cost)) # convert value to absolute to represent cost better
    average_cost.append(sum(total_cost)/len(total_cost))
    epoch +=1
    epsilon_decay.append(epsilon)
         
print('Q matrix:\n %r' % Q)
print('epochs: %r' %epoch )

#        print('current node: %s, current iteration: %d, current distance from goal %d' % (a, count, np.nansum(R)))
#        print("\n")
#        print('R matrix:\n %r' % R)
#        print("\n")
#        print('Q matrix:\n %r' % Q)
#        print("\n\n")

# %%

'''Metrics'''

M_transtions_best = min(total_transitions)
M_transtions_worst = max(total_transitions)
M_transtions_ave = average(total_transitions)

M_score_best = min(total_cost)
M_score_worst = max(total_cost)
M_score_ave = average(total_cost)

#plt.plot(total_transitions)

summary_cost = []
for i in range(0,int(size(total_cost)/10)-1):   # calculates average cost in bins of 10 epochs 
    summary_cost.append(np.mean(total_cost[i:i+10]))
     
summary_cost_sd = []
for i in range(0,int(size(total_cost)/10)-1):   # calculates average cost in bins of 10 epochs 
    summary_cost_sd.append(total_cost[i:i+10])  # with s.d.


n = 100
window_ave = []
for i in range(1,int(size(total_cost))):    # calculates average cost of previous 50 epochs
    if i<n-1:
        window_ave.append(np.mean(total_cost[0:i]))
    else:
        window_ave.append(np.average(total_cost[i-(n-1):i]))
     
n = 100
window_ave_sd = []
for i in range(1,int(size(total_cost))):    # calculates average cost of previous 50 epochs
    if i<n-1:
        window_ave_sd.append(total_cost[0:n-1])
    else:
        window_ave_sd.append(total_cost[i-(n-1):i])
        
    
        
plt.plot(window_ave)
plt.title('average of last 100')
plt.show()
  
#sns.tsplot(np.transpose(summary_cost_sd))
#plt.show()

sns.tsplot(np.transpose(window_ave_sd))
plt.show()

plt.plot(total_cost)
plt.title('epoch cost')
plt.show()

plt.plot(summary_cost)
plt.title('cost binned (size = 10 epochs)')
plt.ylabel('Epoch Cost')
plt.show()

x = np.arange(epochs)
y = average_cost
plt.plot(x,y)
plt.title('running average')
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.xscale('log')
ax = plt.gca()
ax.set_axis_bgcolor((1, 1, 1))
ax.spines['bottom'].set_color('black')
ax.spines['right'].set_color('black')
plt.grid(b=True, which='major', color='0.65',linestyle='-')
plt.show()





