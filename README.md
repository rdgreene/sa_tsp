# sa_tsp

Q-Learning applied to the classic Travelling Salesman Problem

Project resources go here:

TSP problems source: https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html

Guidance on versioning: http://semver.org/

## Description of script files:

### tsp_USA
Python script that runs the TSP in an environment that represent 48 cities in the U.S.A.

The inputs are a file containg the distance matrix, a file containg the optimal tour description, and a file containg the coordenates of the citites (all these placed in the tsp_matrices folder)

The script outputs the learned Q-matrix, a line graph showing learning performance and a map showing the differnet tours taken by the agent during the learning phase (among other parameters).

### tsp_doubleQ
Python script that runs the TSP in an environment that represent 48 cities in the U.S.A. and compares the impleentation of Q-Learning against Double Q-Learning.

The inputs are a file containg the distance matrix, a file containg the optimal tour description, and a file containg the coordenates of the citites (all these placed in the tsp_matrices folder)

The script outputs two learned Q-matrix, a line graph showing learning performance comparing both methods, and a map for each method showing the differnet tours taken by the agent during the learning phase (among other parameters).

### tsp_PlotData
Python script that contains all ploting functions created for this project (using matplotlib).
