"""
simulate.py
This file will simulate an agents interactions with the openAI gym environment
"""

import numpy as np
import gym
import argparse
import ReinforcementAgent
from time import sleep


def main():
    parser  = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", help="save model after training")
    parser.add_argument("-l", "--load", help="load a model to use")
    parser.add_argument("environment", help="environment to train agent on")
    parser.add_argument("--agent", "-a", help="agent type to use")
    parser.add_argument("-n", help="number of episodes to train (default 1000)",
                        type=int)
    parser.add_argument('alpha', help="alpha value to use for learning", type=float)
    parser.add_argument('gamma', help="gamma value to use for learning", type=float)
    parser.add_argument("--epsilon", "-e", help="epsilon value to use for greedy exploration")
    args = parser.paser_args()
    # parse the command line arguments
    env_name = args.environment
    agent_name = args.agent
    if args.n != None:
        episodes = args.n
    else:
        episodes = 1000
    alpha = args.alpha
    gamma = args.gamma
    if args.epsilon != None:
        epsilon = args.epsilon
    else:
        epsilon = .3
    if args.load != None:
        load = args.load
    else:
        load = None
    if args.save != None:
        save = args.save
    else:
        save = None


    # initialize the environment
    env = gym.make(env_name)
    # initialize the agent
    if agent_name == "greedy":
        agent = ReinforcementAgent.GreedyAgent(alpha, gamma, epsilon,
                                    numTraining, env)
    else if agent_name == "ucb":
        agent = ReinforcementAgent.UCBAgent(alpha, gamma, UCB_const,
                                    numTraining, env)
    else if agent_name == "qlearn":

    # if we are loading Q values for a RL agent
    if agent_name in ['greedy', 'ucb'] and load != None:
        agent.setQValues(load)

    # if we are loading a model for a Deep RL agent
    # TODO: set up model loading

    # do training episodes
    for episode in range(episodes):
        # initialize environment variables
        state = env.reset()
        reward = 0
        done = False
        action = None

        steps = 0
        sum_reward = 0
        # run until we reach the goal or make one million steps 
        while not done: #or steps < 1000000:
            # given a state get an action from the agent
            next_action = agent.explore(state)

            # apply the action in the environment
            next_state, reward, done, info = env.step(action)

            # update the Q table values
            agent.update(state, action, next_state, reward)

            steps += 1
            sum_reward += reward

        # print the statistics from the training episode
        print("Results from episode {}: Average Reward={} over {} steps".format(
                episode, sum_reward/steps, steps))

    # do a trial episode to show the policy the agent has learned
    # initialize environment variables
    state = env.reset()
    reward = 0
    done = False
    action = None
    # run until we reach the goal
    while not done:
        # given a state get an action from the agent
        next_action = agent.getAction(state)

        # apply the action in the environment
        next_state, reward, done, info = env.step(action)
        env.render()
        sleep(.1)

    # save the Q values if specified from the command line
    if save != None:
        agent.saveQValues(save)
