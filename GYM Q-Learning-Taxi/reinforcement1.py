import numpy as np
import gym
import random

# The Taxi Problem

# from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
# by Tom Dietterich

# Description:
# There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.

# Observations: 
# There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations. 

# Passenger locations:
# - 0: R(ed)
# - 1: G(reen)
# - 2: Y(ellow)
# - 3: B(lue)
# - 4: in taxi

# Destinations:
# - 0: R(ed)
# - 1: G(reen)
# - 2: Y(ellow)
# - 3: B(lue)

# Actions:
# There are 6 discrete deterministic actions:
# - 0: move south
# - 1: move north
# - 2: move east 
# - 3: move west 
# - 4: pickup passenger
# - 5: dropoff passenger

# Rewards: 
# There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.

# Rendering:
# - blue: passenger
# - magenta: destination
# - yellow: empty taxi
# - green: full taxi
# - other letters (R, G, Y and B): locations for passengers and destinations

# Get Taxi-v2 environment
env = gym.make("Taxi-v2").env

# Initilize Q Table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparams
lr = 0.1
discount = 0.9
epsilon = 0.1

# Informations
rewards = []
wrongDropouts = []

epoch = 10000

for i in range(1, epoch):
    # Initilize Environment
    # stateID will be 0 (Initial State)
    stateID = env.reset()

    # Total reward for epoch
    totalReward = 0
    wrongDropoutCount = 0

    done = False
    while not done:
        # Exploit vs Explore to find action
        if random.uniform(0,1) < epsilon:
            # Get random action to Explore unknows
            actionID = env.action_space.sample()
        else:
            actionID = np.argmax(q_table[stateID])

        # Perform action and get reward and observations
        nextStateID, reward, done, info = env.step(actionID)

        # Q Learning Function Calc
        stateOldValue = q_table[stateID, actionID]
        # Max value from all actions can be perform in nextState
        nextStateMaxValue = np.max(q_table[nextStateID])

        stateNextValue = (1-lr)*stateOldValue + lr*(reward + discount*nextStateMaxValue)

        # Q Table Update
        q_table[stateID, actionID] = stateNextValue

        # State Update
        stateID = nextStateID

        # Wrong dropouts finder
        if reward == -10:
            wrongDropoutCount += 1

        # Sum total reward
        totalReward += reward

    if i%10 is 0:
        wrongDropouts.append(wrongDropoutCount)
        rewards.append(totalReward)
        print("Epoch: {}, Reward: {}, Wrong Droput {}".format(i,totalReward,wrongDropoutCount))
