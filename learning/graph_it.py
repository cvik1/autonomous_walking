"""
graph_it.py
This script graphs the data produced by get_data.py
The data shows a traditional epsilon-greedy agent, a traditional upper confidence
bound agent, and a deep q learning agent training on four openai gym environments
with different hyper-parameters.
"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    # create lists environments to itererate over
    environments = ['Taxi-v2', 'MountainCar-v0', 'CartPole-v1', 'LunarLander-v2']
    # create lists of possible learning rates and discount factors to iterate over
    alphas = [.85, .9, .95, .99]
    gammas = [.85, .9, .95, .99]

    # get all of the data to graph
    ucb_train = np.load("../models/ucb_training")
    ucb_test = np.load("../models/ucb_testing")

    for i, env_name in enumerate(environments):
        # loop over the environments to make a separate graph for each one
