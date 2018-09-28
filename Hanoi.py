import numpy as np
import sys
import time

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
# absorbing state --> all probabilities are zero
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


#### VALUE ITERATION
def valueIteration():
    policy = [0] * len(states)
    value = [0 for s in states]
    change = 1
    loops = 0 # count convergence loops
    # sys.float_info.epsilon is 2.220446049250313e-16
    while change > sys.float_info.epsilon:
        loops += 1
        change = 0
        newV = np.zeros(len(states))
        for s in states:
            newValueOfState = max([rewardsSA[s, a] + GAMMA * sum([transitionTable[s, a, sPrime] * value[sPrime] for sPrime in states]) for a in actions])
            if abs(newValueOfState - value[s]) > change:
                change = abs(newValueOfState - value[s])
            newV[s] = newValueOfState
        value = newV

    for s in states:
        policy[s] = actions[np.argmax([rewardsSA[s, a] + GAMMA * sum([transitionTable[s, a, sPrime] * value[sPrime] for sPrime in states]) for a in actions])]

    return policy, value, loops


#### POLICY ITERATION
def policyIteration():
    policy = [0] * len(states)
    utility = [0 for s in states]
    change = True
    loops = 0 # count convergence loops
    while change:
        loops += 1
        change = False
        # calculate utility given policy
        rewards = [rewardsSA[s, policy[s]] for s in states]
        utilityMatrix = np.eye(len(states))
        for i, s in enumerate(states):
            for j, sPrime in enumerate(states):
                utilityMatrix[i, j] += -(GAMMA * transitionTable[s, policy[s], sPrime])
        utility = np.linalg.solve(utilityMatrix, rewards)

        for s in states:
            newAction = np.argmax([rewardsSA[s, a] + GAMMA * sum([transitionTable[s, a, sPrime] * utility[sPrime] for sPrime in states]) for a in actions])
            if policy[s] != newAction:
                policy[s] = newAction
                change = True

    return policy, utility, loops


GAMMA = 0.9
diskMove = ["a1", "a2", "a3", "b1", "b2", "b3"]

viPolicy, viUtility, viLoops = valueIteration()
print("#####     Value Iteration:      #####")
print("Optimal policy: ")
for s in states:
    print(f"{diskMove[viPolicy[s]]} ", end="")
print()
print("Utility of different states:")
for s in states:
    print(f"{viUtility[s]:.2f} ", end="")
print()
print(f"Converged after {viLoops} loops", )

print()

piPolicy, piUtility, piLoops = policyIteration()
print("#####     Policy Iteration:      #####")
print("Optimal policy: ")
for s in states:
    print(f"{diskMove[piPolicy[s]]} ", end="")
print()
print("Utility of different states:")
for s in states:
    print(f"{piUtility[s]:.2f} ", end="")
print()
print(f"Converged after {piLoops} loops", )

##### COMPARE SPEEDS #####
print()
print("Compare convergence speed:")
n = 100
startTime = time.time()
for i in range(n):
    valueIteration()
valueTime = time.time() - startTime
print(f"{n} iterations of valueIteration took {valueTime:.4f}s. On Average: {valueTime/n:.2}s")

startTime = time.time()
for i in range(n):
    policyIteration()
policyTime = time.time() - startTime
print(f"{n} iterations of policyIteration took {policyTime:.4f}s. On Average: {policyTime/n:.2}s")
