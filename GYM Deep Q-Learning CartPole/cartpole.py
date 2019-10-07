import random
import numpy as np
from collections import deque
from time import sleep

import gym

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

class Agent:
    def __init__(self, environment):
        """
        Hyperparameters definition far Agent
        """
        #  State size means state's features number so it's 4 for cartpole
        #  Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip
        #  So it's also dnn input layer node size
        self.stateSize = environment.observation_space.shape[0]
        #  Action space is output size and 2 for cartpole: push left, push right
        self.actionSize = environment.action_space.n

        #  Trust rate to our experiences
        self.gamma = 0.95 #  Discount
        self.alpha = 0.001 #  Learning Rate

        #  After many experinces epsilon will be 0.01
        #  So we will do less Explore more Exploit
        self.epsilon = 1 #  Explore or Exploit
        self.epsilonDecay = 0.995 #  Adaptive Epsilon Decay Rate
        self.epsilonMinimum = 0.01 #  Minimum for Explore

        #  Deque because after it is full 
        #  the old ones start to delete
        self.memory = deque(maxlen = 1000)

        self.dnnModel = self.buildDNN()

    def buildDNN(self, hiddenLayerNodeNum=48):
        """
        DNN Model Definition
        Stanadart DNN model
        """
        model = Sequential()
        model.add(Dense(hiddenLayerNodeNum, input_dim = self.stateSize, activation="relu"))
        model.add(Dense(self.actionSize, activation="linear"))
        model.compile(loss = "mse", optimizer = Adam(lr = self.alpha))
        return model

    def storeResults(self, state, action, reward, nextState, done):
        """
        Store every result to memory
        """
        self.memory.append((state, action, reward, nextState, done))

    def act(self, state):
        """
        Get state and do action
        Exploit or Explore ???
        If explore get random action
        """
        if random.uniform(0,1) <= self.epsilon:
            return environment.action_space.sample()
        else:
            actValues = self.dnnModel.predict(state)
            return np.argmax(actValues[0])

    def train(self, batchSize):
        """
        Training here
        """
        # If memory is not enough for training we pass
        if len(self.memory) < batchSize:
            return
        # Get random samples
        minibatch = random.sample(self.memory, batchSize)

        # For every samples
        for state, action, reward, nextState, done in minibatch:
            if done: # If in that sample done is true we just use reward for y_train
                target = reward
            else: # Else we use Q Learning forumlua r+gamma*max(Q(s')) for y_train
                target = reward + self.gamma*np.amax(self.dnnModel.predict(nextState)[0])
            
            # So x_train = s
            # y_train = r+gamma*max(Q(s')) or if done only r
            # Remeber Q function means predict in here
            # Even we only try to max one action in here
            # We need to get all other actions predict results
            # So we can't just use np.zeros in here
            trainTarget = self.dnnModel.predict(state)
            trainTarget[0][action] = target
            self.dnnModel.fit(state,trainTarget,verbose=0)

    def adaptiveEpsilon(self):
        """
        Adaptive Epsilon means every episode
        we decrease the epsilon so we do less Explore
        """
        if self.epsilon > self.epsilonMinimum:
            self.epsilon *= self.epsilonDecay

def test(trainedAgent, env):
    """
    We perform test here
    """
    state = env.reset() # Reset env
    state = np.reshape(state, [1,4])
    time = 0

    while True:
        env.render() # Show state visually
        action = trainedAgent.act(state) # Do action
        nextState, reward, done, info = env.step(action) # observe
        nextState = np.reshape(nextState, [1,4])
        state = nextState # Update state
        time += 1
        print("Time:{} Reward:{}".format(time, reward))
        sleep(0.2)
        if done:
            print("Test Completed.")
            break

if __name__ == "__main__":
    environment = gym.make("CartPole-v0") # Get env
    agent = Agent(environment) # Create Agent

    # Size of state batch which taken randomly from memory
    # We use 16 random state to fit model in every time step
    batchSize = 16
    # There will be 100 different epsiode
    episodeNum = 100

    for e in range(episodeNum):
        state = environment.reset()

        state = np.reshape(state,[1,4])

        time = 0 # Remeber time limit is 200 after 200 system will be autoreset done will be true

        while True:
            action = agent.act(state) # Act

            nextState, reward, done, info = environment.step(action) # Observe
            nextState = np.reshape(nextState,[1,4])

            agent.storeResults(state, action, reward, nextState, done) # Storage to mem

            state = nextState # Update State

            agent.train(batchSize) # Train with random 16 state taken from mem

            agent.adaptiveEpsilon() # Decrase epsilon

            time += 1 # Increase time

            if done:
                print("Episode:{} Time:{}".format(e,time))
                break

    test(agent, environment)