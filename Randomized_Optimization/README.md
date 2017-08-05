# Randomized Optimization

# Analysis
  * Located at Randomized Optimization analysis.pdf

# Purpose
  * Explore random search

# Implement four local random search algorithms

  * Randomized hill climbing
  * Simulated annealing
  * A genetic algorithm
  * MIMIC
  * Use the first three algorithms to find good weights for a neural network. Use them instead of backprop for the neural network I used in supervised leardning.
  
# Notes
  * An "optimization problem" is just a fitness function one is trying to maximize (as opposed to a cost function one is trying to minimize).
  *  The first problem highlights advantages of my genetic algorithm, the second of simulated annealing, and the third of MIMIC. 

# System Requirements
  * Java 8
  * ant
  * Abagail (attached)

# Data
  * You will need specify where each data set is located in your system and putting that path in the code where ever it was set to be located on my file directory. The 70/30 split dataset in csv format is located in AbagailProject/30pima.csv and AbagailProject/70pima.csv

# Graphing

  * To plot the graphs I produced you must copy each result, fitness and time into an excel sheet or use a web app such a plotly.

# Part 1: Comparing Randomized Optimizers to Back Propagation Neural Networks

  * Located in the ‘AbagailProject’ file I included run AbagailProject/src/test.java in inside of Eclipse. This will run random hill climbing, simulated annealing, and genetic algorithm. Use the weights,parameters and training iterations specified in my report to replicate the results. 

# Part 2: Optimization Problems

## Traveling Salesman
  * Located in the ‘ABAGAIL’ project file I included run ABAGAIL/src/opt.test/TravelingSalesmanTest.java in inside of Eclipse. This will run random hill climbing, simulated annealing,genetic algorithm and MIMIC. Use the weights, parameters and training iterations specified in my report to replicate the results. 

## Continuous Peaks 
  * Located in the ‘ABAGAIL’ project file I included run ABAGAIL/src/opt.test/ContinuousPeaksTest.java in inside of Eclipse. This will run random hill climbing, simulated annealing,genetic algorithm and MIMIC. Use the weights, parameters and training iterations specified in my report to replicate the results. 

## Max-K Coloring
  * Located in the ‘ABAGAIL’ project file I included run ABAGAIL/src/opt.test/MaxKColoringTest.java in inside of Eclipse. This will run random hill climbing, simulated annealing,genetic algorithm and MIMIC. Use the weights, parameters and training iterations specified in my report to replicate the results. 