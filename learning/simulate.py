"""
simulate.py
This file will simulate an agents interactions with the openAI gym environment
"""

import numpy as np
import gym
import argparse
import ReinforcementAgents
from time import sleep


def main():
    parser  = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", help="save model after training")
    parser.add_argument("-l", "--load", help="load a model to use")
    parser.add_argument("environment", help="environment to train agent on")
    parser.add_argument("-a", "--agent", help="agent type to use")
    parser.add_argument("-n", "--numEpisodes", help="number of episodes to train (default 1000)",
                        type=int)
    parser.add_argument('alpha', help="alpha value to use for learning", type=float)
    parser.add_argument('gamma', help="gamma value to use for learning", type=float)
    parser.add_argument("-c", "--constant", help="epsilon for greedy, UCB_const for UCB")
    args = parser.parse_args()
    # parse the command line arguments
    env_name = args.environment
    agent_name = args.agent
    if args.numEpisodes != None:
        episodes = args.numEpisodes
    else:
        episodes = 1000
    alpha = args.alpha
    gamma = args.gamma
    if args.constant != None:
        epsilon = args.constant
        UCB_const = args.constant
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

    numTraining = 1000


    # initialize the environment
    env = gym.make(env_name)
    # initialize the agent
    if agent_name == "greedy":
        agent = ReinforcementAgents.GreedyAgent(alpha, gamma, epsilon,
                                    numTraining, env)
    elif agent_name == "ucb":
        agent = ReinforcementAgents.UCBAgent(alpha, gamma, UCB_const,
                                    numTraining, env)
    elif agent_name == "random":
        agent = ReinforcementAgents.RandomAgent(env)
    elif agent_name == "deepq":
        raise Exception('Deep qlearning not yet implemented ')
    elif agent_name == "deepp":
        raise Exception('Deep policy learning not yet implemented')
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
            action = agent.explore(state)

            # apply the action in the environment
            next_state, reward, done, info = env.step(action)

            # update the Q table values
            agent.update(state, action, next_state, reward)

            steps += 1
            sum_reward += reward

        # print the statistics from the training episode
        print("Results from episode {:4d}: Average Reward={:.2f} over {:3d} steps".format(
                episode, sum_reward/steps, steps))

    # # do a trial episode to show the policy the agent has learned
    # # initialize environment variables
    # state = env.reset()
    # reward = 0
    # done = False
    # action = None
    # # run until we reach the goal
    # while not done:
    #     # given a state get an action from the agent
    #     action = agent.getAction(state)
    #
    #     # apply the action in the environment
    #     next_state, reward, done, info = env.step(action)
    #     env.render()
    #     sleep(.2)
    #
    # # save the Q values if specified from the command line
    # if save != None:
    #     agent.saveQValues(save)


if __name__ == "__main__":
    main()
