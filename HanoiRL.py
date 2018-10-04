import numpy as np
import sys
import time
import random

states = np.arange(12)
        #  a1 a2 a3 b1 b2 b3
actions = [0, 1, 2, 3, 4, 5]

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
# absorbing state
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

# rewardsSAS has size SxAxS
# initialize all rewards as -1
rewardsSAS = np.ones((len(states), len(actions), len(states)))*-1
# rewards for ending up in the final state is 100
# rewards for ending up in states 2, 3, 4 and 5 are -10
rewardsSAS[..., 2] = 100
rewardsSAS[..., 3] = -10
rewardsSAS[..., 4] = -10
rewardsSAS[..., 5] = -10

# r(s,a) = SUM(t(s,a,s')*r(s,a,s'))
# rewardsSA has size SxA
rewardsSA = np.sum(transitionTable*rewardsSAS, axis=2)
# print("RewardsSA[9,2]: should be around 89 ", rewardsSA[9,2])

def getNewStateReward(oldState, action):

    newPosStates = transitionTable[oldState, action]
    # print(newPosStates)
    # if action is not allowed:
    if np.sum(newPosStates) == 0.0:
        return s, -1

    # draw one sample from the possible states with the given possibilites
    newState = np.random.choice(len(newPosStates), 1, p=newPosStates)[0]
    reward = rewardsSAS[oldState, action, newState]

    return newState, reward

def getAlpha(state, action):
    visits[state, action] += 1
    return 1.0/visits[state, action]

#### Q-Learning ####
diskMove = ["a1", "a2", "a3", "b1", "b2", "b3"]
GAMMA = 0.9
# learning rate
visits = np.zeros((len(states), len(actions)))
qValues = np.zeros((len(states), len(actions)))

for i in range(10000):
    s = np.random.choice(len(states), 1)[0]
    while s != states[2]:
        # choose action and execute it
        a = np.argmax(qValues[s])
        # identify new state and observe reward
        sPrime, r = getNewStateReward(s, a)
        # assign new q-Values
        qValues[s, a] = qValues[s, a] + getAlpha(s, a)*(r+ GAMMA* np.max([qValues[sPrime, aPrime] for aPrime in actions]) - qValues[s, a])
        # update alpha

        s = sPrime

for i, q in enumerate(qValues):
    print(f"pi(state{i}) = {diskMove[np.argmax(q)]} with q-Value: {np.max(q):.2f}")
print(np.max(qValues, axis=1))
# [73.42236461 60.10292676  0.         84.97138285 86.65556643 50.80545637
#  85.88999301 69.19490563 63.96330884 98.65136193 98.84062416 60.52354838]
# print(qValues)
# getNewStateReward(states[0], actions[1])
