"""
ContinuousReinforcementAgents.py
This file will define the class ContinuousQLearningAgent as:
An agent that approximates states to maintain a table of (state,action) pairs
in order to estimate Q-values from experience rather than a model
"""

import numpy as np
import gym
import pickle
import random
import math

import ReinforcementAgents

class ContinuousQLearningAgent(ReinforcementAgents.QLearningAgent):

    def __init__(self, alpha, gamma, env, buckets):
        """
        Initializes our basic ContinuousQLearningAgent
        """
        self.alpha = float(alpha) #learning rate
        self.gamma = float(gamma) #discount factor

        self.Q_values = {} # dictionary to hold Q values
        self.env = env #learning environment

        high = env.observation_space.low
        low = env.observation_space.high
        self.buckets = [0 for i in range(len(low))]
        # make the list of buckets to approximate our states
        for i in range(len(low)):
            try:
                self.buckets[i] = np.arange(low[i],high[i],(high[i]-low[i])/buckets)
            # if range is too large to compute buckets
            except:
                self.buckets[i] = np.arange(-10, 10, 20/buckets)

    def setQValues(self, filename):
        """
        if loading a pretrained agent, set the Q-value table to the
        table from that agent's training
        """
        try:
            f = open(filename, 'rb')
            # load the dictionary form the file
            self.Q_values = pkl.load(f)
            return True
        except:
            return False

    def saveQValues(self, filename):
        """
        save Q values to a file for reloading at another time
        """
        try:
            f = open(filename, 'wb')
            # dump the dictionary into the file
            pickle.dump(self.Q_values, f)
            return (True, None)
        except Exception as e:
            return (False, e)

    def getQValues(self):
        """
        returns Q values for debugging purposes
        """
        return self.Q_values


    def greedyPolicy(self, state):
        """
        Chooses the best availble action based on Q-values
        inputs: state
        outputs: action
        """
        actions = range(0, self.env.action_space.n)
        if len(actions) == 0:
            return None
        values = []
        # gets the expected value for each action
        for action in actions:
            values.append((action, self.actionValue(state, action)))
        values = np.array(values)
        np.random.shuffle(values) # shuffle so in case of a tie we choose randomly
        return values[np.argmax(values[:,1]),0]

    def getAction(self, state):
        """
        the method called by the simulation script to get an action
        input: state
        output: action from greedy policy
        """
        # get the closest state approximation for the q-table
        state_approx = state
        for i in range(len(state)):
            state_approx[i] = min(self.buckets[i], key=lambda x:abs(x-state[i]))

        action = self.greedyPolicy(tuple(state_approx))

        return int(action)

    def explore(self, state):
        """
        the method called by the simulation script to get an action during exploration
        input: state
        output: action from exploration policy
        """
        # get the closest state approximation for the q-table
        state_approx = state
        for i in range(len(state)):
            state_approx[i] = min(self.buckets[i], key=lambda x:abs(x-state[i]))

        action = self.explorationPolicy(tuple(state_approx))

        return int(action)

    def update(self, state, action, nextState, reward):
        """
        update the Q table using the reward
        """
        # if we're still learning update the q table
        nextQ = self.stateValue(nextState)
        curQ = self.actionValue(state, action)
        self.Q_values[(state, action)] = (self.actionValue(state, action) +
                self.alpha * (reward + self.gamma * nextQ - curQ))

class GreedyAgent(ContinuousQLearningAgent):

    def __init__(self, alpha, gamma, epsilon, env, buckets):
        """
        Initializes our Epsilon greedy q-learning agent
        """
        self.alpha = float(alpha) #learning rate
        self.gamma = float(gamma) #discount factor
        self.epsilon = float(epsilon) #exploration randomization factor

        self.Q_values = {}
        self.env = env #learning environment

        high = list(env.observation_space.low)
        low = list(env.observation_space.high)
        self.buckets = [0 for i in range(len(low))]
        # make the list of buckets to approximate our states
        for i in range(len(low)):
            try:
                self.buckets[i] = np.arange(low[i],high[i],(high[i]-low[i])/buckets)
            # if range is too large to compute buckets
            except:
                self.buckets[i] = np.arange(-10, 10, 20/buckets)


    def explorationPolicy(self, state):
        """
        implements the epsilon greedy exploration policy
        inputs: state
        outputs: action
        """
        actions = list(range(0, self.env.action_space.n))
        if len(actions) == 0:
            return None
        values = []
        # get the value for each available action from the given state
        for action in actions:
            values.append((action, self.actionValue(state, action)))
        values = np.array(values)
        #keys = list(values.keys())
        np.random.shuffle(values) # shuffle so in case of a tie we choose randomly
        ran = random.random()
        if ran < self.epsilon: # if random value less than epsilon
            return random.choice(values[:,0]) # choose a random action
        else: # otherwise choose the greedy action
            return values[np.argmax(values[:,1]),0]

    def update(self, state, action, nextState, reward):
        """
        update the Q table using the reward
        """
        # get the approximate states for the q-table updates
        state_approx = state
        nextState_approx = nextState
        for i in range(len(state)):
            state_approx[i] = min(self.buckets[i], key=lambda x:abs(x-state[i]))
            nextState_approx[i] = min(self.buckets[i], key=lambda x:abs(x-nextState[i]))
        # convert to tuples for the dictionary
        state_approx = tuple(state_approx)
        nextState_approx = tuple(nextState_approx)

        # if we're still learning update the q table
        nextQ = self.stateValue(nextState_approx)
        curQ = self.actionValue(state_approx, action)
        self.Q_values[(state_approx, action)] = (self.actionValue(state_approx, action) +
                self.alpha * (reward + self.gamma * nextQ - curQ))

class UCBAgent(ContinuousQLearningAgent):

    def __init__(self, alpha, gamma, env, buckets):
        self.alpha = float(alpha) # learning rate
        self.gamma = float(gamma) # discount factor
        self.UCB_const = float(UCB_const) # constant to calculate UCB values in exploration

        self.Q_values = {}
        self.visits = {}
        self.env = env

        high = env.observation_space.low
        low = env.observation_space.high
        self.buckets = [0 for i in range(len(low))]
        # make the list of buckets to approximate our states
        for i in range(len(low)):
            try:
                self.buckets[i] = np.arange(low[i],high[i],(high[i]-low[i])/buckets)
            # if range is too large to compute buckets
            except:
                self.buckets[i] = np.arange(-10, 10, 20/buckets)

    def explorationPolicy(self, state):
        """
        Implements UCB exploration policy
        Computes UCB weights, then normalizes them into a probability
        distribution to sample from
        inputs: state
        outputs: action
        """

        actions = list(range(0,self.env.action_space.n))
        if len(actions) == 0:
            return None
        weights = []

        action_visits = []
        # count how many times each action has been taken from this state
        for action in actions:
            w = self.actionValue(state, action)
            v = self.visits.get((state,action), 0)
            weights.append(w)
            action_visits.append(v)
        # sum of all actions taken from this state equals total visits to the state
        sum_v = sum(action_visits)
        if sum_v != 0:
            # if not the first visit to state, compute UCB weights
            for i in range(len(action_visits)):
                if action_visits[i] != 0:
                    ucb = self.UCB_const * math.sqrt(math.log(sum_v) / action_visits[i])
                else:
                    ucb = self.UCB_const * math.sqrt(math.log(sum_v) / 1)
                weights[i] += ucb

        sum_w = float(sum([abs(w) for w in weights]))
        if sum_w == 0:
            # if the first visit to this state choose action at random
            index = np.random.choice(range(len(weights)))
        else: # create distribution using UCB weights and sample
            # normalize each weight by the sum
            norm_weights = [abs(i)/sum_w for i in weights]
            # randomly select the index of a weight from the normalized weights
            index = np.random.choice(range(len(weights)),1, p=norm_weights)[0]

        return actions[index]

    def update(self, state, action, nextState, reward):
        """
        update the Q table using the reward
        update the visits table
        """
        # get the approximate states for the q-table updates
        state_approx = state
        nextState_approx = nextState
        for i in range(len(state)):
            state_approx[i] = min(self.buckets[i], key=lambda x:abs(x-state[i]))
            nextState_approx[i] = min(self.buckets[i], key=lambda x:abs(x-nextState[i]))
        # convert to tuples for the dictionary
        state_approx = tuple(state_approx)
        nestState_approx = tuple(nextState_approx)

        # update the q table
        nextQ = self.stateValue(nextState_approx)
        curQ = self.actionValue(state_approx, action)
        self.Q_values[(state_approx, action)] = (self.actionValue(state_approx, action) +
                    self.alpha * (reward + self.gamma * nextQ - curQ))

        self.visits[(state_approx,action)] = self.visits.get((state_approx,action),0) + 1
