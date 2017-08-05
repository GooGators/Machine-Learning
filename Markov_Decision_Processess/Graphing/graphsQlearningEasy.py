import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#terminal_state_q_value = [35,135,39,35,86,50,18,15,52,19,15,10,15,17,14,97,19,14,12,14,25,12,20,24,37,22,36,14,11,44,85,18,15,10,16,18,14,12,66,13,252,9,16,32,14,20,29,15,9,11,82,12,37,19,13,11,22,16,12,31,20,11,13,16,48,9,18,11,21,21,33,10,14,13,18,12,16,16,58,19,57,63,11,10,13,46,11,51,16,15,36,16,10,15,45,12,21,10,16,9,28,9,9,19,11,17,26,25,20,18,15,16,19,11,17,26,16,33,11,9,15,11,9,22,36,18,12,27,22,9,35,10,25,15,18,71,14,14,9,40,30,11,12,9,28,14,11,20,19,15,12,18,10,22,15,20,41,24,9,13,39,11,28,18,13,18,17,11,25,22,18,9,55,12,17,21,10,14,11,9,17,24,31,27,27,18,11,43,15,13,15,17,37,10,37,20,11,22,15,15,10,11,14,24,24,9,24,11,16,17,15,43,12,14,10,20,10,24,21,25,19,16,14,44,18,14,21,16,17,58,14,55,25,9,33,10,105,13,20,16,17,21,25,9,14,9,43,14,10,19,14,11,11,19,30,33,79,51,27,25,15,21,12,12,11,10,18,11,26,20,15,38,15,14,27,11,44,11,12,25,12,15,9,28,12,21,12,18,10,12,10,12,15,37,27,11,19,10,14,10]
#terminal_state_q_value = pd.DataFrame(terminal_state_q_value)
terminal_state_q_value_2 = [327,161,63,87,41,17,19,15,10,34,39,13,18,12,17,12,17,12,14,13,12,57,16,14,11,10,9,10,29,15,16,14,9,20,54,9,11,13,11,20,14,16,12,15,14,19,15,11,9,11,11,18,12,18,13,14,19,27,10,12,11,12,13,15,10,11,12,15,15,17,12,10,22,15,15,11,11,15,13,11,10,11,19,12,13,12,20,10,15,12,9,13,11,9,19,12,10,19,15,17]
terminal_state_q_value_2 = pd.DataFrame(terminal_state_q_value_2)
terminal_state_q_value_4 = [40,30,147,106,14,24,24,18,54,74,9,27,16,20,11,32,17,9,12,11,9,19,46,9,11,23,18,12,23,17,17,16,30,9,16,16,13,13,22,29,9,13,11,15,13,18,11,20,17,13,11,9,12,14,12,11,14,9,11,12,15,26,19,14,10,13,9,17,9,14,11,14,9,11,13,14,20,10,14,9,12,9,11,9,10,13,14,13,13,15,15,11,13,17,17,16,18,9,24,9]
terminal_state_q_value_4 = pd.DataFrame(terminal_state_q_value_4)
#terminal_state_q_value_6 = [132,28,14,84,27,17,29,13,12,19,9,35,17,11,12,13,10,14,9,13,36,41,10,12,13,9,12,19,10,11,14,9,12,10,22,13,12,9,13,12,49,12,16,17,9,10,14,14,14,9,10,13,15,17,16,10,15,19,9,9,11,16,9,12,18,18,14,9,16,13,19,15,12,13,10,9,11,11,25,17,11,12,14,44,12,21,11,49,16,11,29,11,20,17,11,17,15,39,13,23,13,14,11,9,12,20,10,26,18,13,9,12,12,15,10,12,10,25,21,9,16,12,9,17,16,12,12,11,13,20,12,9,11,13,10,20,12,15,11,17,12,15,12,17,11,14,39,19,9,9,20,12,11,10,17,12,14,9,17,14,15,21,16,15,22,22,11,30,26,14,24,11,13,12,14,13,22,11,14,17,13,12,13,17,14,9,14,20,11,16,17,35,11,9,30,12,11,12,10,18,13,24,11,29,9,15,14,19,9,11,17,21,40,21,16,17,13,11,11,14,17,12,18,36,10,13,12,13,10,14,22,9,10,14,20,19,15,9,12,12,16,16,9,12,13,12,9,13,9,10,20,13,17,10,14,16,13,10,20,13,33,28,14,22,15,22,9,9,14,10,13,14,11,10,11,15,14,19,15,11,13,16,15,11,14,10,32,13,13,11,12,17,24,19,11,13,13,18,12,26]
#terminal_state_q_value_6 = pd.DataFrame(terminal_state_q_value_6)
#terminal_state_q_value_8 = [32,174,40,47,87,33,22,12,16,16,9,68,11,12,21,37,11,15,30,24,21,10,29,47,11,23,59,12,20,13,9,10,14,16,12,14,13,16,21,12,15,14,14,24,32,14,21,19,14,16,12,17,18,12,15,15,14,12,13,38,12,13,13,16,32,17,13,11,15,20,11,16,14,23,10,45,17,13,19,12,10,9,12,12,11,15,18,30,10,15,38,15,11,17,10,11,17,11,11,9,15,19,16,12,13,10,9,23,17,14,21,11,15,26,10,15,10,24,15,12,19,22,11,19,15,28,14,13,19,16,13,10,9,22,51,11,16,61,20,13,24,13,11,16,14,15,13,29,12,12,14,11,22,34,14,12,11,9,61,13,17,14,17,35,10,9,17,18,21,18,11,18,34,13,16,52,13,36,14,14,14,18,21,16,18,12,11,11,17,10,9,20,23,16,20,11,23,12,13,9,11,9,10,12,12,18,9,20,13,11,15,10,13,14,19,16,12,12,13,10,11,13,10,25,11,39,15,37,12,11,15,19,13,18,66,46,9,12,13,9,10,10,10,10,19,74,10,11,13,10,12,9,12,14,19,12,21,21,20,16,24,23,19,25,14,16,17,13,9,30,12,27,18,17,12,18,13,15,67,42,12,11,20,11,9,16,12,12,16,21,20,11,15,20,13,22,13,11,17,20]
#terminal_state_q_value_8 = pd.DataFrame(terminal_state_q_value_8)
x = xrange(0,100)
fig = plt.figure()
plt.plot(x, terminal_state_q_value_4, label='Q-learning Learning Rate .40')
plt.plot(x, terminal_state_q_value_2, label='Q-learning Learning Rate .20')
plt.ylabel("Steps/Actions Required to Reach the Terminal State")
plt.xlabel("Iterations")
plt.legend(loc='upper center', shadow=True)
plt.show()

#terminal_state_q_value = [134,21,14,10,7,11,8,11,15,5,7,13,6,7,26,9,4,4,3,4,3,4,3,3,4,10,5,7,6,7,4,3,3,4,8,4,7,3,7,4,5,10,4,4,4,4,6,4,3,4,9,12,15,14,17,15,9,5,4,10,6,8,5,6,6,5,16,12,6,8,10,9,5,6,8,9,7,16,21,20,16,15,14,16,22,6,7,7,8,10,7,7,8,8,8,9,9,12,9,7,9,9,8,10,8,8,9,12,11,23,13,9,11,11,14,12,13,28,26,47,109,17,21,21,25,25,22,24,22,10,13,9,11,9,13,13,10,9,9,12,10,11,10,11,9,12,14,14,12,11,13,15,13,12,8,10,14,9,11,9,17,11,10,15,11,11,14,10,11,11,12,19,11,15,29,26,30,27,28,36,30,29,29,30,33,10,16,13,13,14,13,12,14,12,15,16,11,14,14,14,11,16,13,14,19,13,14,16,14,13,21,17,14,14,18,15,14,15,14,18,21,15,13,15,16,15,14,16,15,18,13,15,17,12,12,15,17,21,17,41,36,38,33,40,35,37,32,38,41,35,46,35,38,27,16,17,17,21,16,20,15,16,22,21,22,17,23,17,18,15,15,22,14,20,23,26,19,27,22,20,29,27,27,21,22,25,32,20,24,17,17,18,16,21,16,25,17,17,19,20]
#terminal_state_q_value = pd.DataFrame(terminal_state_q_value)
terminal_state_q_value_2 = [64,10,11,11,7,9,8,11,8,14,9,22,7,6,10,3,4,3,3,4,4,5,5,12,4,6,4,2,3,4,3,4,4,12,4,4,4,4,4,6,3,10,4,3,5,5,3,3,5,6,5,8,12,10,11,10,11,14,4,4,4,4,4,4,5,3,4,4,4,4,4,9,4,5,4,4,7,4,4,5,5,5,6,4,4,9,13,13,12,12,12,11,13,13,13,12,6,5,5,6]
terminal_state_q_value_2 = pd.DataFrame(terminal_state_q_value_2)
terminal_state_q_value_4 = [38,25,18,9,9,11,10,10,10,10,8,11,13,16,9,9,4,6,4,4,3,3,7,3,4,10,4,4,5,4,3,5,3,2,4,4,6,6,5,3,3,7,3,3,3,3,11,5,4,3,4,6,3,3,3,4,3,5,13,10,10,10,19,5,5,4,4,6,4,4,5,5,4,4,4,5,3,10,3,4,4,5,4,5,4,4,4,4,5,4,5,7,12,11,13,15,12,13,12,12]
terminal_state_q_value_4 = pd.DataFrame(terminal_state_q_value_4)
#terminal_state_q_value_6 = [63,13,10,8,8,6,12,7,7,7,6,7,9,17,7,5,4,4,5,4,4,3,4,3,3,4,9,3,3,3,3,3,3,3,3,4,7,3,3,4,4,4,4,3,3,10,3,4,4,3,3,4,4,5,4,3,6,9,11,12,11,16,6,5,4,5,3,5,5,4,8,4,4,4,3,10,4,4,6,4,4,4,4,4,4,5,4,4,6,7,13,10,11,15,13,10,12,12,14,4,5,5,7,6,4,6,7,6,6,6,4,5,6,5,4,5,5,6,6,5,6,6,5,6,6,5,7,5,5,9,6,5,6,6,7,8,13,8,14,17,16,20,15,19,18,15,16,18,15,16,17,17,11,6,6,7,7,7,7,10,9,7,9,8,11,8,7,8,8,10,8,7,8,7,7,11,8,8,9,9,9,8,7,8,8,8,12,7,8,8,9,9,9,7,9,9,9,9,8,11,8,8,8,11,21,22,22,28,27,23,23,22,22,24,26,23,23,15,9,12,9,12,11,11,10,14,10,9,12,10,11,12,11,10,11,14,10,11,9,10,10,11,11,10,10,9,9,11,11,11,12,11,14,11,10,12,14,12,12,12,12,12,11,14,11,12,11,12,13,12,11,13,15,13,15,13,11,12,11,27,28,35,27,30,30,30,31,35,34,29,32,35,33,33,33,27,13,12,16,14]
#terminal_state_q_value_6 = pd.DataFrame(terminal_state_q_value_6)
#terminal_state_q_value_8 = [38,29,9,11,9,10,8,8,9,10,12,15,6,7,15,8,5,5,3,5,5,4,3,4,3,3,10,4,4,2,4,4,7,3,4,7,3,2,3,2,3,3,4,5,3,11,3,3,3,5,4,5,3,4,4,7,12,12,11,9,12,3,5,4,5,3,3,4,4,6,8,7,11,6,3,6,4,5,4,7,5,4,6,5,4,9,12,14,14,14,12,14,13,20,5,6,7,8,4,5,5,5,6,9,6,6,5,8,7,6,6,7,6,6,5,5,6,6,6,6,7,8,6,10,9,8,7,15,21,19,21,20,21,19,15,19,23,21,22,21,23,10,9,8,7,7,9,8,7,8,8,7,7,9,9,8,12,11,9,10,8,9,10,14,9,8,11,8,9,14,9,9,8,11,11,9,10,10,11,9,11,9,11,10,13,9,9,9,10,11,22,27,31,28,22,29,29,29,27,27,25,25,18,10,13,9,10,10,11,11,11,12,17,10,11,10,12,15,11,14,11,16,16,12,14,17,11,15,15,11,11,16,13,11,12,12,14,14,15,11,11,12,10,16,12,12,14,12,12,13,13,13,13,13,12,11,13,11,13,13,14,33,30,37,31,42,152,49,33,35,33,34,33,33,41,35,17,19,18,19,16,17,19,21,19,18,15,17,18,19,16,20,18,19,18,17,20,18,19,19]
#terminal_state_q_value_8 = pd.DataFrame(terminal_state_q_value_8)

x = xrange(0,100)
fig = plt.figure()
#plt.plot(x, terminal_state_q_value, label='Q-learning Learning Rate .99')
#plt.plot(x, terminal_state_q_value_8, label='Q-learning Learning Rate .80')
#plt.plot(x, terminal_state_q_value_6, label='Q-learning Learning Rate .60')
plt.plot(x, terminal_state_q_value_4, label='Q-learning Learning Rate .40')
plt.plot(x, terminal_state_q_value_2, label='Q-learning Learning Rate .20')
plt.ylabel("Milliseconds the Algorithm Required to Generate the Optimal Policy")
plt.xlabel("Iterations")

plt.legend(loc='upper center', shadow=True)
plt.show()


#terminal_state_q_value = [67.0,-33.0,63.0,67.0,16.0,52.0,84.0,87.0,50.0,83.0,87.0,92.0,87.0,85.0,88.0,5.0,83.0,88.0,90.0,88.0,77.0,90.0,82.0,78.0,65.0,80.0,66.0,88.0,91.0,58.0,17.0,84.0,87.0,92.0,86.0,84.0,88.0,90.0,36.0,89.0,-150.0,93.0,86.0,70.0,88.0,82.0,73.0,87.0,93.0,91.0,20.0,90.0,65.0,83.0,89.0,91.0,80.0,86.0,90.0,71.0,82.0,91.0,89.0,86.0,54.0,93.0,84.0,91.0,81.0,81.0,69.0,92.0,88.0,89.0,84.0,90.0,86.0,86.0,44.0,83.0,45.0,39.0,91.0,92.0,89.0,56.0,91.0,51.0,86.0,87.0,66.0,86.0,92.0,87.0,57.0,90.0,81.0,92.0,86.0,93.0,74.0,93.0,93.0,83.0,91.0,85.0,76.0,77.0,82.0,84.0,87.0,86.0,83.0,91.0,85.0,76.0,86.0,69.0,91.0,93.0,87.0,91.0,93.0,80.0,66.0,84.0,90.0,75.0,80.0,93.0,67.0,92.0,77.0,87.0,84.0,31.0,88.0,88.0,93.0,62.0,72.0,91.0,90.0,93.0,74.0,88.0,91.0,82.0,83.0,87.0,90.0,84.0,92.0,80.0,87.0,82.0,61.0,78.0,93.0,89.0,63.0,91.0,74.0,84.0,89.0,84.0,85.0,91.0,77.0,80.0,84.0,93.0,47.0,90.0,85.0,81.0,92.0,88.0,91.0,93.0,85.0,78.0,71.0,75.0,75.0,84.0,91.0,59.0,87.0,89.0,87.0,85.0,65.0,92.0,65.0,82.0,91.0,80.0,87.0,87.0,92.0,91.0,88.0,78.0,78.0,93.0,78.0,91.0,86.0,85.0,87.0,59.0,90.0,88.0,92.0,82.0,92.0,78.0,81.0,77.0,83.0,86.0,88.0,58.0,84.0,88.0,81.0,86.0,85.0,44.0,88.0,47.0,77.0,93.0,69.0,92.0,-3.0,89.0,82.0,86.0,85.0,81.0,77.0,93.0,88.0,93.0,59.0,88.0,92.0,83.0,88.0,91.0,91.0,83.0,72.0,69.0,23.0,51.0,75.0,77.0,87.0,81.0,90.0,90.0,91.0,92.0,84.0,91.0,76.0,82.0,87.0,64.0,87.0,88.0,75.0,91.0,58.0,91.0,90.0,77.0,90.0,87.0,93.0,74.0,90.0,81.0,90.0,84.0,92.0,90.0,92.0,90.0,87.0,65.0,75.0,91.0,83.0,92.0,88.0,92.0]
#terminal_state_q_value = pd.DataFrame(terminal_state_q_value)
terminal_state_q_value_2 = [-225.0,-59.0,39.0,15.0,61.0,85.0,83.0,87.0,92.0,68.0,63.0,89.0,84.0,90.0,85.0,90.0,85.0,90.0,88.0,89.0,90.0,45.0,86.0,88.0,91.0,92.0,93.0,92.0,73.0,87.0,86.0,88.0,93.0,82.0,48.0,93.0,91.0,89.0,91.0,82.0,88.0,86.0,90.0,87.0,88.0,83.0,87.0,91.0,93.0,91.0,91.0,84.0,90.0,84.0,89.0,88.0,83.0,75.0,92.0,90.0,91.0,90.0,89.0,87.0,92.0,91.0,90.0,87.0,87.0,85.0,90.0,92.0,80.0,87.0,87.0,91.0,91.0,87.0,89.0,91.0,92.0,91.0,83.0,90.0,89.0,90.0,82.0,92.0,87.0,90.0,93.0,89.0,91.0,93.0,83.0,90.0,92.0,83.0,87.0,85.0]
terminal_state_q_value_2 = pd.DataFrame(terminal_state_q_value_2)
terminal_state_q_value_4 = [62.0,72.0,-45.0,-4.0,88.0,78.0,78.0,84.0,48.0,28.0,93.0,75.0,86.0,82.0,91.0,70.0,85.0,93.0,90.0,91.0,93.0,83.0,56.0,93.0,91.0,79.0,84.0,90.0,79.0,85.0,85.0,86.0,72.0,93.0,86.0,86.0,89.0,89.0,80.0,73.0,93.0,89.0,91.0,87.0,89.0,84.0,91.0,82.0,85.0,89.0,91.0,93.0,90.0,88.0,90.0,91.0,88.0,93.0,91.0,90.0,87.0,76.0,83.0,88.0,92.0,89.0,93.0,85.0,93.0,88.0,91.0,88.0,93.0,91.0,89.0,88.0,82.0,92.0,88.0,93.0,90.0,93.0,91.0,93.0,92.0,89.0,88.0,89.0,89.0,87.0,87.0,91.0,89.0,85.0,85.0,86.0,84.0,93.0,78.0,93.0]
terminal_state_q_value_4 = pd.DataFrame(terminal_state_q_value_4)
#terminal_state_q_value_6 = [-30.0,74.0,88.0,18.0,75.0,85.0,73.0,89.0,90.0,83.0,93.0,67.0,85.0,91.0,90.0,89.0,92.0,88.0,93.0,89.0,66.0,61.0,92.0,90.0,89.0,93.0,90.0,83.0,92.0,91.0,88.0,93.0,90.0,92.0,80.0,89.0,90.0,93.0,89.0,90.0,53.0,90.0,86.0,85.0,93.0,92.0,88.0,88.0,88.0,93.0,92.0,89.0,87.0,85.0,86.0,92.0,87.0,83.0,93.0,93.0,91.0,86.0,93.0,90.0,84.0,84.0,88.0,93.0,86.0,89.0,83.0,87.0,90.0,89.0,92.0,93.0,91.0,91.0,77.0,85.0,91.0,90.0,88.0,58.0,90.0,81.0,91.0,53.0,86.0,91.0,73.0,91.0,82.0,85.0,91.0,85.0,87.0,63.0,89.0,79.0,89.0,88.0,91.0,93.0,90.0,82.0,92.0,76.0,84.0,89.0,93.0,90.0,90.0,87.0,92.0,90.0,92.0,77.0,81.0,93.0,86.0,90.0,93.0,85.0,86.0,90.0,90.0,91.0,89.0,82.0,90.0,93.0,91.0,89.0,92.0,82.0,90.0,87.0,91.0,85.0,90.0,87.0,90.0,85.0,91.0,88.0,63.0,83.0,93.0,93.0,82.0,90.0,91.0,92.0,85.0,90.0,88.0,93.0,85.0,88.0,87.0,81.0,86.0,87.0,80.0,80.0,91.0,72.0,76.0,88.0,78.0,91.0,89.0,90.0,88.0,89.0,80.0,91.0,88.0,85.0,89.0,90.0,89.0,85.0,88.0,93.0,88.0,82.0,91.0,86.0,85.0,67.0,91.0,93.0,72.0,90.0,91.0,90.0,92.0,84.0,89.0,78.0,91.0,73.0,93.0,87.0,88.0,83.0,93.0,91.0,85.0,81.0,62.0,81.0,86.0,85.0,89.0,91.0,91.0,88.0,85.0,90.0,84.0,66.0,92.0,89.0,90.0,89.0,92.0,88.0,80.0,93.0,92.0,88.0,82.0,83.0,87.0,93.0,90.0,90.0,86.0,86.0,93.0,90.0,89.0,90.0,93.0,89.0,93.0,92.0,82.0,89.0,85.0,92.0,88.0,86.0,89.0,92.0,82.0,89.0,69.0,74.0,88.0,80.0,87.0,80.0,93.0,93.0,88.0,92.0,89.0,88.0,91.0,92.0,91.0,87.0,88.0,83.0,87.0,91.0,89.0,86.0,87.0,91.0,88.0,92.0,70.0,89.0,89.0,91.0,90.0,85.0,78.0,83.0,91.0,89.0,89.0,84.0,90.0,76.0]
#terminal_state_q_value_6 = pd.DataFrame(terminal_state_q_value_6)
#terminal_state_q_value_8 = [70.0,-72.0,62.0,55.0,15.0,69.0,80.0,90.0,86.0,86.0,93.0,34.0,91.0,90.0,81.0,65.0,91.0,87.0,72.0,78.0,81.0,92.0,73.0,55.0,91.0,79.0,43.0,90.0,82.0,89.0,93.0,92.0,88.0,86.0,90.0,88.0,89.0,86.0,81.0,90.0,87.0,88.0,88.0,78.0,70.0,88.0,81.0,83.0,88.0,86.0,90.0,85.0,84.0,90.0,87.0,87.0,88.0,90.0,89.0,64.0,90.0,89.0,89.0,86.0,70.0,85.0,89.0,91.0,87.0,82.0,91.0,86.0,88.0,79.0,92.0,57.0,85.0,89.0,83.0,90.0,92.0,93.0,90.0,90.0,91.0,87.0,84.0,72.0,92.0,87.0,64.0,87.0,91.0,85.0,92.0,91.0,85.0,91.0,91.0,93.0,87.0,83.0,86.0,90.0,89.0,92.0,93.0,79.0,85.0,88.0,81.0,91.0,87.0,76.0,92.0,87.0,92.0,78.0,87.0,90.0,83.0,80.0,91.0,83.0,87.0,74.0,88.0,89.0,83.0,86.0,89.0,92.0,93.0,80.0,51.0,91.0,86.0,41.0,82.0,89.0,78.0,89.0,91.0,86.0,88.0,87.0,89.0,73.0,90.0,90.0,88.0,91.0,80.0,68.0,88.0,90.0,91.0,93.0,41.0,89.0,85.0,88.0,85.0,67.0,92.0,93.0,85.0,84.0,81.0,84.0,91.0,84.0,68.0,89.0,86.0,50.0,89.0,66.0,88.0,88.0,88.0,84.0,81.0,86.0,84.0,90.0,91.0,91.0,85.0,92.0,93.0,82.0,79.0,86.0,82.0,91.0,79.0,90.0,89.0,93.0,91.0,93.0,92.0,90.0,90.0,84.0,93.0,82.0,89.0,91.0,87.0,92.0,89.0,88.0,83.0,86.0,90.0,90.0,89.0,92.0,91.0,89.0,92.0,77.0,91.0,63.0,87.0,65.0,90.0,91.0,87.0,83.0,89.0,84.0,36.0,56.0,93.0,90.0,89.0,93.0,92.0,92.0,92.0,92.0,83.0,28.0,92.0,91.0,89.0,92.0,90.0,93.0,90.0,88.0,83.0,90.0,81.0,81.0,82.0,86.0,78.0,79.0,83.0,77.0,88.0,86.0,85.0,89.0,93.0,72.0,90.0,75.0,84.0,85.0,90.0,84.0,89.0,87.0,35.0,60.0,90.0,91.0,82.0,91.0,93.0,86.0,90.0,90.0,86.0,81.0,82.0,91.0,87.0,82.0,89.0,80.0,89.0,91.0,85.0,82.0]
#terminal_state_q_value_8 = pd.DataFrame(terminal_state_q_value_8)

x = xrange(0,100)
fig = plt.figure()
#plt.plot(x, terminal_state_q_value, label='Q-learning Learning Rate .99')
#plt.plot(x, terminal_state_q_value_8, label='Q-learning Learning Rate .80')
#plt.plot(x, terminal_state_q_value_6, label='Q-learning Learning Rate .60')
plt.plot(x, terminal_state_q_value_4, label='Q-learning Learning Rate .40')
plt.plot(x, terminal_state_q_value_2, label='Q-learning Learning Rate .20')
plt.ylabel("Total Reward Gained for The Optimal Policy ")
plt.xlabel("Iterations")

plt.legend(loc='middle bottom', shadow=True)
plt.show()