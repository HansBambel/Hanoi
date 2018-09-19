import numpy as np
import random

GAMMA = 0.9
states = np.arange(12)

           # 0     1     2    3     4      5    6
actions = ["a1", "a2", "a3", "b1", "b2", "b3", "c"]
stateAction = {0:[1, 2],
               1:[0, 2],
               2:[6],
               3:[4, 5],
               4:[4, 5],
               5:[3, 4],
               6:[0, 2, 4, 5],
               7:[1, 2, 3, 5],
               8:[0, 1, 4, 5],
               9:[1, 2, 3, 4],
               10:[0, 2, 3, 4],
               11:[0, 1, 3, 5]
              }

rewards = np.ones((len(states), len(actions)))*-1
# rewards = [[-1]*len(states) * len(actions)]
print(rewards.shape)
rewards[2,6] = 0
rewards[9,2] = 100    # goal state
rewards[10,2] = 100    # goal state
rewards[6,0] = -10    # punishment for big disk on small disk
rewards[7,3] = -10
rewards[8,5] = -10
rewards[9,3] = -10
rewards[10,4] = -10
rewards[11,5] = -10


# transitiontable has size SxAxS
transitionTable = np.zeros((len(states), len(actions), len(states)))
# S0
transitionTable[0,1,6] = 0.9
transitionTable[0,1,8] = 0.1
transitionTable[0,2,8] = 0.9
transitionTable[0,2,6] = 0.1
# S1
transitionTable[1,0,7] = 0.9
transitionTable[1,0,11] = 0.1
transitionTable[1,2,11] = 0.9
transitionTable[1,2,7] = 0.1
# S2
transitionTable[2,6,2] = 1.0
# S3
transitionTable[3,4,7] = 0.9
transitionTable[3,4,9] = 0.1
transitionTable[3,5,9] = 0.9
transitionTable[3,5,7] = 0.1
# S4
transitionTable[4,3,6] = 0.9
transitionTable[4,3,10] = 0.1
transitionTable[4,5,10] = 0.9
transitionTable[4,5,6] = 0.1
# S5
transitionTable[5,3,8] = 0.9
transitionTable[5,3,11] = 0.1
transitionTable[5,4,11] = 0.9
transitionTable[5,4,8] = 0.1
# S6
transitionTable[6,0,0] = 0.9
transitionTable[6,0,8] = 0.1
transitionTable[6,2,8] = 0.9
transitionTable[6,2,0] = 0.1
transitionTable[6,4,4] = 0.9
transitionTable[6,4,10] = 0.1
transitionTable[6,5,10] = 0.9
transitionTable[6,5,4] = 0.1
# S7
transitionTable[7,1,1] = 0.9
transitionTable[7,1,11] = 0.1
transitionTable[7,2,11] = 0.9
transitionTable[7,2,1] = 0.1
transitionTable[7,3,3] = 0.9
transitionTable[7,3,9] = 0.1
transitionTable[7,5,9] = 0.9
transitionTable[7,5,3] = 0.1
# S8
transitionTable[8,0,0] = 0.9
transitionTable[8,0,6] = 0.1
transitionTable[8,1,6] = 0.9
transitionTable[8,1,0] = 0.1
transitionTable[8,4,11] = 0.9
transitionTable[8,4,5] = 0.1
transitionTable[8,5,5] = 0.9
transitionTable[8,5,11] = 0.1
# S9
transitionTable[9,1,10] = 0.9
transitionTable[9,1,2] = 0.1
transitionTable[9,2,2] = 0.9
transitionTable[9,2,10] = 0.1
transitionTable[9,3,3] = 0.9
transitionTable[9,3,7] = 0.1
transitionTable[9,4,7] = 0.9
transitionTable[9,4,3] = 0.1
# S10
transitionTable[10,0,9] = 0.9
transitionTable[10,0,2] = 0.1
transitionTable[10,2,2] = 0.9
transitionTable[10,2,9] = 0.1
transitionTable[10,3,6] = 0.9
transitionTable[10,3,4] = 0.1
transitionTable[10,4,4] = 0.9
transitionTable[10,4,6] = 0.1
# S11
transitionTable[11,0,7] = 0.9
transitionTable[11,0,1] = 0.1
transitionTable[11,1,1] = 0.9
transitionTable[11,1,7] = 0.1
transitionTable[11,3,8] = 0.9
transitionTable[11,3,5] = 0.1
transitionTable[11,5,5] = 0.9
transitionTable[11,5,8] = 0.1

#### VALUE ITERATION
policy = [stateAction[s][0] for s in states]
print("Initial policy: ", policy)
V = [0 for s in states]
epsilon = 0.0001
change = 1
while change > epsilon:
    change = 0
    for s in states:
        newValueOfState = max([rewards[s,a] + GAMMA * sum([transitionTable[s, a, sPrime] * V[sPrime] for sPrime in states]) for a in stateAction[s]])
        if abs(newValueOfState - V[s]) > change:
            change = abs(newValueOfState - V[s])
        V[s] = newValueOfState

for s in states:
    policy[s] = stateAction[s][np.argmax([rewards[s, a] + GAMMA * sum([transitionTable[s,a,sPrime]*V[sPrime] for sPrime in states]) for a in stateAction[s]])]
print("optimal policy: ", policy)
print("V: ", V)
