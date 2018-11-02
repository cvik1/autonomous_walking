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

from ReinforcementAgents import *

class ContinuousQLearningAgent(QlearningAgent):

    def __init__(self, alpha, gamma, numTraining, env, buckets):
        """
        Initializes our basic ContinuousQLearningAgent
        """
        QlearningAgent.__init__(self, alpha, gamma, numTraining, env)
        self.buckets = buckets
