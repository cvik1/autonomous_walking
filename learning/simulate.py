"""
simulate.py
This file will simulate an agents interactions with the openAI gym environment
"""

import numpy as np
import gym
import argparse
import ReinforcementAgents
import ContinuousReinforcementAgents
import DeepReinforcementAgents
from time import sleep

from pprint import pprint


def main():

    parser  = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", help="save model after training")
    parser.add_argument("-l", "--load", help="load a model to use")
    parser.add_argument("environment", help="environment to train agent on")
    parser.add_argument("-a", "--agent", help="agent type to use")
    parser.add_argument("-n", "--numEpisodes", help="number of episodes to train (default 1000)",
                        type=int)
    parser.add_argument("-b", "--buckets", help="number of buckets to use when approximating"
                        "a continuous state space in the q-table", type=int)
    parser.add_argument("-m", '--maxsteps', help="max number of steps per episode (default 100)",
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
        episodes = 10000
    alpha = args.alpha
    gamma = args.gamma
    if args.constant != None:
        epsilon = args.constant
        UCB_const = args.constant
    else:
        epsilon = .3
    if args.buckets != None:
        buckets = args.buckets
    else:
        buckets = None
    if args.load != None:
        load = args.load
    else:
        load = None
    if args.save != None:
        save = args.save
    else:
        save = None

    numTraining = 100000000


    # initialize the environment
    env = gym.make(env_name)
    # raise the step limit
    if args.maxsteps != None:
        if env._max_episode_steps < args.maxsteps:
            env._max_episode_steps = args.maxsteps
    else:
        if env._max_episode_steps < 100:
            env._max_episode_steps = 100
    if env_name == "FrozenLake-v0":
        env._is_slippery=False
    # initialize the agent
    if agent_name == "greedy":
        agent = ReinforcementAgents.GreedyAgent(alpha, gamma, epsilon, env)
    elif agent_name == "c-greedy":
        if buckets == None:
            agent = ContinuousReinforcementAgents.GreedyAgent(alpha, gamma,
                                    epsilon, env, 10)
        else:
            agent = ContinuousReinforcementAgents.GreedyAgent(alpha, gamma,
                                    epsilon, env, buckets)
    elif agent_name == "ucb":
        agent = ReinforcementAgents.UCBAgent(alpha, gamma, UCB_const, env)
    elif agent_name =="c-ucb":
        if buckets == None:
            agent = ContinuousReinforcementAgents.UCBAgent(alpha, gamma,
                                    UCB_const, env, 10)
        else:
            agent = ContinuousReinforcementAgents.UCBAgent(alpha, gamma,
                                    UCB_const, env, buckets)
    elif agent_name == "random":
        agent = ReinforcementAgents.RandomAgent(env)
    elif agent_name == "taxi":
        agent = DeepReinforcementAgents.TaxiAgent(alpha, gamma, epsilon, env)
    elif agent_name == "cartpole":
        agent = DeepReinforcementAgents.CartPoleAgent(alpha, gamma, epsilon, env)
    elif agent_name == "mountaincar":
        agent = DeepReinforcementAgents.MountainCarAgent(alpha, gamma, epsilon, env)
    elif agent_name == "lunarlander":
        agent = DeepReinforcementAgents.LunarLanderAgent(alpha, gamma, epsilon, env)


    # if we are loading Q values for a RL agent
    if agent_name in ['greedy', 'ucb'] and load != None:
        agent.setQValues(load)

    # if we are loading a model for a Deep RL agent
    # TODO: set up model loading

    # create array to hold all training data
    training_data = []
    policy_data = []
    # do training episodes
    for episode in range(episodes):
        # initialize environment variables
        state = env.reset()
        reward = 0
        done = False
        action = None

        steps = 0
        sum_reward = 0
        # run a real episode to check the learning progress every 25 episodes
        if (episodes+1)%25 == 0:
            policy_steps = 0
            policy_sum_reward = 0
            while not done:
                # given a state get an action from the agent
                action = agent.getAction(state)
                # apply the action in the environment
                next_state, reward, done, info = env.step(action)
                #update state variable
                state = next_state

                policy_steps +=1
                policy_sum_reward += reward
            policy_data.append([policy_steps, policy_sum_reward])

        # run a training episode every iteration of the loop
        while not done:
            # given a state get an action from the agent
            action = agent.explore(state)
            # apply the action in the environment
            next_state, reward, done, info = env.step(action)

            # update the Q table values
            agent.update(state, action, next_state, reward, done)
            # update the state variable
            state = next_state

            steps += 1
            sum_reward += reward
        # store the training data from this episode
        training_data.append([steps, sum_reward])

        # print the statistics from the training episode
        if (episode+1)%(episodes//10) == 0:
            print("Results from episode {:6d}: Total Reward={:7.2f} over {:3d} steps".format(
                    episode+1, sum_reward, steps))
        # run a batch training on our deep reinforcement agents
        if (episodes%25) == 0:
            try:
                agent.train_model()
            except:
                pass


    # do a trial episode to show the policy the agent has learned
    # initialize environment variables
    state = env.reset()
    reward = 0
    done = False
    action = None

    steps = 0
    sum_reward = 0
    # run until we reach the goal
    while not done:
        # given a state get an action from the agent
        action = agent.getAction(state)
        # apply the action in the environment
        next_state, reward, done, info = env.step(action)
        #update state variable
        state = next_state

        steps +=1
        sum_reward += reward

        env.render()

        sleep(.05)

    print("Results from testing episode: \n {:25s}{:5f}\n{:25s}{:5f}\n{:25s}{:3.2f}".format(
            "Steps:", steps, "Total Reward:", sum_reward, "Average Reward:", sum_reward/steps))

    # save the Q values if specified from the command line
    if save != None:
        status = agent.saveQValues(save)
        if status[0]:
            print("Saved Q table to {}".format(save))
        else:
            print(status[1])


    # print the Q-table
    # pprint(agent.Q_values)

    # return all the data from each training episode and the test trail
    return (policy_data, training_data)


if __name__ == "__main__":
    data = main()
