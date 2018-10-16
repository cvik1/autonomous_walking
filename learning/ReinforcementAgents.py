"""
ReinforcementAgents.py
This file will define the class QLearningAgent as:
An agent that estimates Q-values from experience rather than a model
"""
import numpy as np
import random
import math

class QLearningAgent():

    def __init__(self, alpha, gamma, numTraining, env):
        """
        Initializes our basic Q learning agent
        """
        self.alpha = float(alpha) #learning rate
        self.gamma = float(gamma) #discount factor
        self.steps = 0 # initialize training steps to 0
        self.numTraining = int(numTraining) # number of training steps

        self.Q_values{} # dictionary to hold Q values
        self.env = env #learning environment

    def setQValues(self, Q_values):
        """
        if loading a pretrained agent, set the Q-value table to the
        table from that agent's training
        """
        self.Q_values = Q_values

    def actionValue(self, state, action):
        """
        returns the q value for a given state-action pair
        returns 0 if the pair has not yet been seen
        inputs: state, action
        outputs: Q value of state-action pair
        """
        return self.Q_values.get((state,action), 0.0)

    def stateValue(self, state):
        """
        returns the maximum value of an action for that given state
        if there are no available actions it returns 0
        inputs: state
        outputs: maximum possble value from that state
        """

        actions = range(0,self.env.action_space.n)
        if len(actions) == 0:
            return 0.0
        values = []
        # get the value of each possible action from a given state
        for action in actions:
            values.append(self.actionValue(state, action))
        return max(values) # return the best possible value from the state

    def greedyPolicy(self, state):
        """
        Chooses the best availble action based on Q-values
        inputs: state
        outputs: action
        """
        actions = range(0, self.env.action_space.n)
        if len(actions) == 0:
            return None
        values = {}
        # gets the expected value for each action
        for action in actions:
            values[self.actionValue(state, action)] = action
        keys = list(values.keys())
        random.shuffle(keys) # shuffle so in case of a tie we choose randomly
        return values[max(keys)] # return action with the maximum expected value

    def getAction(self, state):
        """
        the method called by the simulation script to get an action
        input: state
        output: action from exploration or greedy policy
        """
        # if we are still training, use the exploration policy to select actions
        if self.steps < self.numTraining:
            action = self.explorationPolicy(state)
        # if we are done training, use the greedy policy to select actions
        else:
            action = self.greedyPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
        update the Q table using the reward
        """
        if self.steps < self.numTraining:
            nextQ = self.stateValue(nextState)
            curQ = self.actionValue(state, action)
            self.Q_values[(state, action)] = (self.actionValue(state, action) +
                    self.alpha * (reward + self.gamma * nextQ - curQ))


class RandomAgent(QLearningAgent):

    def __init__(self, env):
        """
        Initializes a random agent
        """
        self.steps = 16 #initialize the steps to be larger than training steps
        self.numTraining = 2 # so we skip exploration and simply do random actions
        self.env = env

    # def explorationPolicy(self, state):
    #     """
    #     implements a random agent's exploration policy
    #     """
    #     actions = range(0, self.env.action_space.n)
    #     return random.choice(actions)

    def update(self, state, action, nextState, reward):
        """
        in the random agent we don't keep a q table so this is just a
        placeholder
        """
        # do nothing

class GreedyAgent(QLearningAgent):

    def __init__(self, alpha, gamma, epsilon, numTraining, env):
        """
        Initializes an Epsilon Greedy learning agent
        """
        self.alpha = float(alpha) #learning rate
        self.gamma = float(gamma) #discount factor
        self.epsilon = float(epsilon) #exploration randomization factor
        self.steps = 0
        self.numTraining = int(numTraining) #number of training steps

        self.env = env #learning environment

    def explorationPolicy(self, state):
        """
        implements the epsilon greedy exploration policy
        inputs: state
        outputs: action
        """
        actions = range(0, self.env.action_space.n)
        if len(actions) == 0:
            return None
        values = {}
        # get the value for each available action from the given state
        for action in actions:
            values[self.actionValue(state, action)] = action
        keys = list(values.keys())
        random.shuffle(keys) # shuffle so in case of a tie we choose randomly
        ran = random.random()
        if ran < self.epsilon: # if random value less than epsilon
            return random.choice(list(values.values())) # choose a random action
        else: # otherwise choose the greedy action
            return values[max(keys)]

    # def update(self, state, action, nextState, reward):
    #     """
    #     update the Q table using the reward
    #     """
    #     if self.steps < self.numTraining:
    #         nextQ = self.stateValue(nextState)
    #         curQ = self.actionValue(state, action)
    #         self.Q_values[(state, action)] = (self.actionValue(state, action) +
    #                 self.alpha * (reward + self.gamma * nextQ - curQ))

class UBBAgent(QLearningAgent):

    def __init__(self, alpha, gamma, UCB_const, numTraining, env):
        self.alpha = float(alpha) # learning rate
        self.gamma = float(gamma) # discount factor
        self.UCB_const = UCB_const # constant to calculate UCB values in exploration
        self.steps = 0 # initialize steps to 0
        self.numTraining = int(numTraining) # number of training steps

        self.Q_values = {}
        self.visits = {}
        self.env = env

    def explorationPolicy(self, state):
        """
        Implements UCB exploration policy
        Computes UCB weights, then normalizes them into a probability
        distribution to sample from
        inputs: state
        outputs: action
        """

        actions = range(0,self.env.action_space.n)
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

    def update(self, state):
        """
        update the Q table using the reward
        update the visits table
        """
        if self.steps < self.numTraining:
            nextQ = self.stateValue(nextState)
            curQ = self.actionValue(state, action)
            self.Q_values[(state, action)] = (self.actionValue(state, action) +
                        self.alpha * (reward + self.gamma * nextQ - curQ))

            self.visits[(state,action)] = self.visits.get((state,action),0) + 1
