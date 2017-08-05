import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


terminal_state_value_iteration = [180671,4523,8892,7739,27354,17356,2269,455,202,276,167,103,59,35,42,34,31,28,34,39,38,30,29,45,39,31,33,30,37,45,40,40,30,34,32,32,40,36,33,37,33,30,37,36,39,29,30,31,33,31,37,36,31,30,28,34,35,26,35,55,32,35,38,34,39,37,41,37,43,44,27,36,32,37,28,35,36,42,43,39,31,40,30,32,33,27,30,29,42,30,33,48,29,37,40,30,31,36,43,38]
terminal_state_value_iteration = pd.DataFrame(terminal_state_value_iteration)

x = xrange(0,30)
fig = plt.figure()
plt.plot(x, terminal_state_value_iteration[0:30], label='Value Iteration')
plt.ylabel("Steps/Actions Required to Reach the Terminal State")
plt.xlabel("Iterations")
plt.legend(loc='upper center', shadow=True)
plt.show()

terminal_state_value_iteration = [56,5,5,5,5,12,7,8,9,10,13,14,12,15,14,14,17,18,22,20,20,21,22,23,29,24,25,29,29,29,28,29,30,30,31,36,33,42,33,37,37,35,37,38,40,39,40,41,41,43,42,47,50,45,46,47,50,62,48,49,52,53,56,60,55,57,57,56,60,57,60,65,63,60,69,65,67,67,65,66,74,68,70,70,74,73,76,76,72,78,73,80,82,80,85,89,87,85,85,82]
terminal_state_value_iteration = pd.DataFrame(terminal_state_value_iteration)

x = xrange(0,30)
fig = plt.figure()
plt.plot(x, terminal_state_value_iteration[0:30], label='Value Iteration')
plt.ylabel("Milliseconds the Algorithm Required to Generate the Optimal Policy")
plt.xlabel("Iterations")

plt.legend(loc='upper center', shadow=True)
plt.show()


terminal_state_value_iteration = [-180569.0,-4421.0,-8790.0,-7637.0,-27252.0,-17254.0,-2167.0,-353.0,-100.0,-174.0,-65.0,-1.0,43.0,67.0,60.0,68.0,71.0,74.0,68.0,63.0,64.0,72.0,73.0,57.0,63.0,71.0,69.0,72.0,65.0,57.0,62.0,62.0,72.0,68.0,70.0,70.0,62.0,66.0,69.0,65.0,69.0,72.0,65.0,66.0,63.0,73.0,72.0,71.0,69.0,71.0,65.0,66.0,71.0,72.0,74.0,68.0,67.0,76.0,67.0,47.0,70.0,67.0,64.0,68.0,63.0,65.0,61.0,65.0,59.0,58.0,75.0,66.0,70.0,65.0,74.0,67.0,66.0,60.0,59.0,63.0,71.0,62.0,72.0,70.0,69.0,75.0,72.0,73.0,60.0,72.0,69.0,54.0,73.0,65.0,62.0,72.0,71.0,66.0,59.0,64.0]
terminal_state_value_iteration = pd.DataFrame(terminal_state_value_iteration)

x = xrange(0,30)
fig = plt.figure()
plt.plot(x, terminal_state_value_iteration[0:30], label='Value Iteration')
plt.ylabel("Total Reward Gained for The Optimal Policy ")
plt.xlabel("Iterations")

plt.legend(loc='middle bottom', shadow=True)
plt.show()

#Policy Iteration

'''
terminal_state_policy_iteration = [74530,11487,55251,6166,588,691,212,761,345,1228,1876,64,49,558,27,29,41,35,29,40,30,46,45,31,44,29,45,31,30,41,34,51,37,34,29,38,38,32,50,34,36,30,28,36,33,40,26,51,37,36,30,40,38,42,30,38,36,46,34,38,41,43,49,32,31,30,32,29,35,33,31,38,31,29,37,31,36,32,40,32,35,35,53,34,33,31,36,27,33,28,30,36,35,31,37,47,33,46,38,41]
terminal_state_policy_iteration = pd.DataFrame(terminal_state_policy_iteration)

x = xrange(10,20)
fig1 = plt.figure()
plt.plot(x, terminal_state_policy_iteration[10:20], label='Policy Iteration')
plt.ylabel("Steps/Actions Required to Reach the Terminal State")
plt.xlabel("Iterations")
plt.legend(loc='upper center', shadow=True)
plt.show()

terminal_state_policy_iteration = [5,6,8,8,11,12,14,16,18,20,22,23,25,26,31,30,32,32,34,35,38,39,41,42,45,51,47,52,60,52,55,59,58,61,64,64,67,69,72,69,75,77,77,78,77,85,85,86,89,99,98,100,113,97,103,105,110,127,120,121,126,139,117,119,117,119,127,137,146,143,139,144,140,137,138,143,136,143,142,143,140,146,148,153,151,155,158,162,160,166,160,163,173,181,168,180,182,182,181,189]
terminal_state_policy_iteration = pd.DataFrame(terminal_state_policy_iteration)

x = xrange(10,20)
fig1 = plt.figure()
plt.plot(x, terminal_state_policy_iteration[10:20], label='Policy Iteration')
plt.ylabel("Milliseconds the Algorithm Required to Generate the Optimal Policy")
plt.xlabel("Iterations")
plt.legend(loc='upper center', shadow=True)
plt.show()

terminal_state_value_iteration = [-74428.0,-11385.0,-55149.0,-6064.0,-486.0,-589.0,-110.0,-659.0,-243.0,-1126.0,-1774.0,38.0,53.0,-456.0,75.0,73.0,61.0,67.0,73.0,62.0,72.0,56.0,57.0,71.0,58.0,73.0,57.0,71.0,72.0,61.0,68.0,51.0,65.0,68.0,73.0,64.0,64.0,70.0,52.0,68.0,66.0,72.0,74.0,66.0,69.0,62.0,76.0,51.0,65.0,66.0,72.0,62.0,64.0,60.0,72.0,64.0,66.0,56.0,68.0,64.0,61.0,59.0,53.0,70.0,71.0,72.0,70.0,73.0,67.0,69.0,71.0,64.0,71.0,73.0,65.0,71.0,66.0,70.0,62.0,70.0,67.0,67.0,49.0,68.0,69.0,71.0,66.0,75.0,69.0,74.0,72.0,66.0,67.0,71.0,65.0,55.0,69.0,56.0,64.0,61.0]
terminal_state_value_iteration = pd.DataFrame(terminal_state_value_iteration)

x = xrange(10,20)
fig = plt.figure()
plt.plot(x, terminal_state_value_iteration[10:20], label='Policy Iteration')
plt.ylabel("Total Reward Gained for The Optimal Policy ")
plt.xlabel("Iterations")

plt.legend(loc='upper center', shadow=True)
plt.show()
'''