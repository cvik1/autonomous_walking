"""
simulate.py
This file will simulate an agents interactions with the openAI gym environment
"""

import numpy as np
import gym
import argparse
from ReinforcementAgent import *


def main():
    parser  = argparse.ArgumentParser()
    parser.add_argument("-s", help="save model after training")
    parser.add_argument("-l", help="load a model to use")
    parser.add_argument("environment", help="environment to train agent on")
    parser.add_argument("-agent", "-a", help="agent type to use")
    parser.add_argument("-n", help="number of episodes to train (default 1000)",
                        type=int)
    parser.add_argument('alpha', help="alpha value to use for learning", type=float)
    parser.add_argument('')
    parser.add_argument("-epsilon", "-e", help="epsilon value to use for greedy exploration")
    parser.paser_args()
    # parse the command line arguments
    env_name =
    agent_name =
    episodes =
    alpha =
    load =
    save =

    # initialize the environment
    env = gym.make(env_name)
    # initialize the agent


    for episode in range(episodes):
        # initialize environment variables
        state = env.reset()
        reward = 0
        done = False
        action = None
        while not done:
            # given a state get an action from the agent
            next_action =
            # apply the action in the environment
            next_state, reward, done, info = env.step(action)
