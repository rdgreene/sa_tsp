# sa_tsp

Q-Learning applied to the classic Travelling Salesman Problem

TSP problems source: https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html

## Description of script files:

### tsp.py
Pyhton scirpt that runs the intial Toy environment (pre-loaded with optimal parameters identified in coursework write-up)

The inputs are a file containing the distance matrix, a file containing the optimal tour description, and a file containing the coordenates of the cities (all these placed in the tsp_matrices folder)

The script outputs the learned Q-matrix (Q_matrix), a line graph showing learning performance and a map showing the differnet tours taken by the agent during the learning phase (among other parameters).

### tspExperiments.py
Python script that runs all experiments reported using the Toy problem environment in the coursework write up

### tspExperimentsPlotResults.py
Python script that generates all plots included the coursework write up relating to Toy problem environment

### tsp_USA.py
Python script that runs the TSP in an environment that represent 48 cities in the U.S.A.

The inputs are a file containing the distance matrix, a file containing the optimal tour description, and a file containing the coordenates of the cities (all these placed in the tsp_matrices folder)

The script outputs the learned Q-matrix (Q_matrix), a line graph showing learning performance and a map showing the differnet tours taken by the agent during the learning phase (among other parameters).

### tsp_doubleQ.py
Python script that runs the TSP in an environment that represent 48 cities in the U.S.A. and compares the impleentation of Q-Learning against Double Q-Learning.

The inputs are a file containg the distance matrix, a file containg the optimal tour description, and a file containg the coordenates of the citites (all these placed in the tsp_matrices folder)

The script outputs two learned Q-matrix, a line graph showing learning performance comparing both methods, and a map for each method showing the differnet tours taken by the agent during the learning phase (among other parameters).

### tspFunctions.py
Contains all functions created specific to the project. Including functions to implement Q-Learning algortihm (qLearn) and epsilon-greedy policy (epsilonGreedy)

### tsp_PlotData.py
Python script that contains all ploting functions created for this project (using matplotlib).
