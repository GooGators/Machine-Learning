import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Value Iteration

terminal_state_value_iteration = [3455,36,780,18,9,11,11,12,16,12,12,11,12,12,13,13,9,9,15,9,9,11,9,10,11,13,13,13,16,19,19,13,9,15,11,9,12,17,10,9,11,10,10,11,11,12,13,14,11,11,17,11,11,15,11,11,12,10,9,18,11,10,10,11,11,9,20,10,11,9,9,17,17,9,17,11,11,10,13,12,9,14,10,16,15,11,15,12,15,10,13,10,16,10,16,9,9,11,13,11]
terminal_state_policy_iteration = [1304,77,60,79,10,12,12,15,11,10,16,10,9,12,9,10,11,11,9,9,13,16,14,11,15,10,10,14,9,9,9,14,10,10,13,12,11,11,13,18,9,9,12,12,11,9,11,11,13,13,11,11,11,9,9,11,20,9,13,9,16,9,14,10,10,10,9,9,10,14,10,10,12,10,10,13,17,13,9,14,16,15,9,11,15,9,10,12,10,9,9,9,9,9,15,10,13,9,9,15]
terminal_state_value_iteration = pd.DataFrame(terminal_state_value_iteration)
terminal_state_policy_iteration = pd.DataFrame(terminal_state_policy_iteration)




x = xrange(3,7)
fig = plt.figure()
plt.plot(x, terminal_state_value_iteration[3:7], label='Value Iteration')
plt.plot(x, terminal_state_policy_iteration[3:7], label='Policy Iteration')

plt.legend(loc='upper center', shadow=True)
plt.show()



terminal_state_value_iteration = [3455,36,780,18,9,11,11,12,16,12,12,11,12,12,13,13,9,9,15,9,9,11,9,10,11,13,13,13,16,19,19,13,9,15,11,9,12,17,10,9,11,10,10,11,11,12,13,14,11,11,17,11,11,15,11,11,12,10,9,18,11,10,10,11,11,9,20,10,11,9,9,17,17,9,17,11,11,10,13,12,9,14,10,16,15,11,15,12,15,10,13,10,16,10,16,9,9,11,13,11]
terminal_state_value_iteration = pd.DataFrame(terminal_state_value_iteration)

x = xrange(0,7)
fig = plt.figure()
plt.plot(x, terminal_state_value_iteration[0:7], label='Value Iteration')
plt.ylabel("Steps/Actions Required to Reach the Terminal State")
plt.xlabel("Iterations")
plt.legend(loc='upper center', shadow=True)
#plt.show()

terminal_state_value_iteration = [166,2,1,2,2,2,2,2,3,3,3,11,4,4,5,5,5,5,6,6,6,6,6,11,8,8,8,8,8,8,15,10,10,12,10,10,10,11,13,16,18,19,21,18,11,12,13,13,15,14,12,10,16,9,10,10,10,10,10,10,10,10,12,18,21,21,22,20,22,20,19,12,12,16,15,13,13,15,13,14,13,16,17,15,14,14,14,14,15,15,15,15,15,16,16,16,23,29,30,30]
terminal_state_value_iteration = pd.DataFrame(terminal_state_value_iteration)

x = xrange(0,7)
fig = plt.figure()
plt.plot(x, terminal_state_value_iteration[0:7], label='Value Iteration')
plt.ylabel("Milliseconds the Algorithm Required to Generate the Optimal Policy")
plt.xlabel("Iterations")

plt.legend(loc='upper center', shadow=True)
#plt.show()


terminal_state_value_iteration = [-3353.0,66.0,-678.0,84.0,93.0,91.0,91.0,90.0,86.0,90.0,90.0,91.0,90.0,90.0,89.0,89.0,93.0,93.0,87.0,93.0,93.0,91.0,93.0,92.0,91.0,89.0,89.0,89.0,86.0,83.0,83.0,89.0,93.0,87.0,91.0,93.0,90.0,85.0,92.0,93.0,91.0,92.0,92.0,91.0]
terminal_state_value_iteration = pd.DataFrame(terminal_state_value_iteration)

x = xrange(0,7)
fig = plt.figure()
plt.plot(x, terminal_state_value_iteration[0:7], label='Value Iteration')
plt.ylabel("Total Reward Gained for The Optimal Policy ")
plt.xlabel("Iterations")

plt.legend(loc='upper center', shadow=True)
#plt.show()

#Policy Iteration


terminal_state_policy_iteration = [1304,77,60,79,10,12,12,15,11,10,16,10,9,12,9,10,11,11,9,9,13,16,14,11,15,10,10,14,9,9,9,14,10,10,13,12,11,11,13,18,9,9,12,12,11,9,11,11,13,13,11,11,11,9,9,11,20,9,13,9,16,9,14,10,10,10,9,9,10,14,10,10,12,10,10,13,17,13,9,14,16,15,9,11,15,9,10,12,10,9,9,9,9,9,15,10,13,9,9,15]
terminal_state_policy_iteration = pd.DataFrame(terminal_state_policy_iteration)

x = xrange(0,7)
fig1 = plt.figure()
plt.plot(x, terminal_state_policy_iteration[0:7], label='Policy Iteration')
plt.ylabel("Steps/Actions Required to Reach the Terminal State")
plt.xlabel("Iterations")
plt.legend(loc='upper center', shadow=True)
#plt.show()

terminal_state_policy_iteration = [3,1,2,2,3,3,3,3,4,5,7,5,7,7,8,7,9,7,8,9,9,11,13,15,10,11,11,14,13,13,16,13,14,13,15,13,14,14,14,14,15,15,18,24,17,17,16,18,17,17,28,38,39,40,41,40,42,44,44,26,21,21,21,25,23,24,24,24,24,25,24,25,26,26,30,35,35,30,33,28,28,30,29,28,29,30,32,30,31,31,31,32,38,68,70,70,72,72,73,76]
terminal_state_policy_iteration = pd.DataFrame(terminal_state_policy_iteration)

x = xrange(0,7)
fig1 = plt.figure()
plt.plot(x, terminal_state_policy_iteration[0:7], label='Policy Iteration')
plt.ylabel("Milliseconds the Algorithm Required to Generate the Optimal Policy")
plt.xlabel("Iterations")
plt.legend(loc='upper center', shadow=True)
#plt.show()

terminal_state_value_iteration = [-6643.0,-8343.0,69.0,-21.0,89.0,93.0,93.0,90.0,89.0,93.0,93.0,92.0,91.0,92.0,91.0,90.0,85.0,92.0,91.0,88.0,89.0,89.0,93.0,88.0,90.0,92.0,93.0,85.0,90.0,93.0,89.0,87.0,92.0,93.0,92.0,91.0,85.0,93.0,90.0,84.0,92.0,88.0,91.0,91.0,93.0,87.0,83.0,93.0,89.0,93.0,89.0,91.0,93.0,92.0,93.0,93.0,90.0,93.0,90.0,91.0,89.0,92.0,88.0,86.0,88.0,85.0,92.0,90.0,85.0,90.0,92.0,91.0,91.0,90.0,82.0,89.0,90.0,92.0,90.0,92.0,89.0,90.0,88.0,89.0,92.0,91.0,91.0,93.0,87.0,93.0,89.0,92.0,93.0,87.0,88.0,85.0,93.0,93.0,86.0,92.0]
terminal_state_value_iteration = pd.DataFrame(terminal_state_value_iteration)

x = xrange(0,7)
fig = plt.figure()
plt.plot(x, terminal_state_value_iteration[0:7], label='Policy Iteration')
plt.ylabel("Total Reward Gained for The Optimal Policy ")
plt.xlabel("Iterations")

plt.legend(loc='upper center', shadow=True)
plt.show()


