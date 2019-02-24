"""
get_data.py

This file trains our different kinds of Reinforcement Agents
(traditional epsilon greedy, traditional UCB, deep q-network)
on four open ai problems (taxi, mountain car, cartpole, and lunar lander)
using a variety of different values for our learning rate, discount factor,
and constant (epsilon or our UCB-constant) and graphs their average reward
per episode over ### episodes
"""

import numpy as np
import matplotlib.pyplot as plt
import gym

import ReinforcementAgents
import ContinuousReinforcementAgents
import DeepReinforcementAgents

def run_training_instance(agent, env, episodes, deep):
    """
    runs full training of some number of episodes for a given agent and
    configuration
    returns the training and policy data
    """

    # # return garbage fro debugging
    # return (np.empty([400,2]), np.empty([10000,2]))

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
        if (episode+1)%25 == 0:
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

        # # print the statistics from the training episode
        # if (episode+1)%(episodes//10) == 0:
        #     print("Results from episode {:6d}: Total Reward={:7.2f} over {:3d} steps".format(
        #             episode+1, sum_reward, steps))
        # run a batch training on our deep reinforcement agents
        if deep:
            agent.train_model()

    # return the data for that training run
    return (policy_data, training_data)

def main():
    """
    runs training instances for multiple agents at multiple configurations
    for multiple openai gym problems
    """
    # run every problem for 10,000 episodes
    episodes = 10000

    # create lists environments to itererate over
    environments = ['Taxi-v2', 'MountainCar-v0', 'CartPole-v1', 'LunarLander-v2']
    # create lists of possible learning rates and discount factors to iterate over
    alphas = [.9, .95] #[.85, .9, .95, .99]
    gammas = [.9, .95] #[.85, .9, .95, .99]

    # create lists of possible constants
    constants = [.3] #[.5, .4, .3, .2]

    print('Running UCB agent')

    # create a 4x4x4x4 array to hold the data for each training run
    # we are returning both the training and the policy data in a tuple
    #eg taxi training data for a=.95 and gamma=.99 and constant = .5 would be:
    # data[0,2,3,0][1]
    # ucb_training = np.empty([4,4,4,4, 10000, 2])
    # ucb_testing = np.empty([4,4,4,4, 400, 2])
    # for i, env_name in enumerate(environments):
    #     for j, alpha in enumerate(alphas):
    #         for k, gamma in enumerate(gammas):
    #             for l, c in enumerate(constants):
    #                 if i==0: # run discrete ucb for taxi
    #                     env = gym.make(env_name)
    #                     agent = ReinforcementAgents.UCBAgent(alpha, gamma,
    #                                 c, env)
    #                     test, train = run_training_instance(agent, env,
    #                                 episodes, False)
    #                     ucb_training[i,j,k,l] = train
    #                     ucb_testing[i,j,k,l] = test
    #                 elif i==1: # run continuous for mountain car
    #                     env = gym.make(env_name)
    #                     # use 20 buckets
    #                     agent = ContinuousReinforcementAgents.UCBAgent(alpha, gamma,
    #                                 c, env, 20)
    #                     test, train = run_training_instance(agent, env,
    #                                 episodes, False)
    #                     ucb_training[i,j,k,l] = train
    #                     ucb_testing[i,j,k,l] = test
    #                 elif i ==2: # run continuous for cart pole
    #                     env = gym.make(env_name)
    #                     # use 20 buckets
    #                     agent = ContinuousReinforcementAgents.UCBAgent(alpha, gamma,
    #                                 c, env, 20)
    #                     test, train = run_training_instance(agent, env,
    #                                 episodes, False)
    #                     ucb_training[i,j,k,l] = train
    #                     ucb_testing[i,j,k,l] = test
    #                 elif i==3: # run continuous for lunar lander
    #                     env = gym.make(env_name)
    #                     # use 10 buckets
    #                     agent = ContinuousReinforcementAgents.UCBAgent(alpha, gamma,
    #                                 c, env, 20)
    #                     test, train = run_training_instance(agent, env,
    #                                 episodes, False)
    #                     ucb_training[i,j,k,l] = train
    #                     ucb_testing[i,j,k,l] = test
    #     print('\tFinished running on {} for all constants.'.format(env_name))
    # # save the table for UCB agent
    # np.save('../models/ucb_training', ucb_training)
    # np.save('../models/ucb_testing', ucb_testing)


    # print("\nRunning Greedy Agent")
    #
    # greedy_training = np.empty([4,4,4,4, 10000, 2])
    # greedy_testing = np.empty([4,4,4,4, 400, 2])
    # for i, env_name in enumerate(environments):
    #     for j, alpha in enumerate(alphas):
    #         for k, gamma in enumerate(gammas):
    #             for l, c in enumerate(constants):
    #                 if i==0: # run discrete ucb for taxi
    #                     env = gym.make(env_name)
    #                     agent = ReinforcementAgents.GreedyAgent(alpha, gamma,
    #                                 c, env)
    #                     test, train = run_training_instance(agent, env,
    #                                 episodes, False)
    #                     greedy_training[i,j,k,l] = train
    #                     greedy_testing[i,j,k,l] = test
    #                 elif i==1: # run continuous for mountain car
    #                     env = gym.make(env_name)
    #                     # use 20 buckets
    #                     agent = ContinuousReinforcementAgents.GreedyAgent(alpha, gamma,
    #                                 c, env, 20)
    #                     test, train = run_training_instance(agent, env,
    #                                 episodes, False)
    #                     greedy_training[i,j,k,l] = train
    #                     greedy_testing[i,j,k,l] = test
    #                 elif i ==2: # run continuous for cart pole
    #                     env = gym.make(env_name)
    #                     # use 20 buckets
    #                     agent = ContinuousReinforcementAgents.GreedyAgent(alpha, gamma,
    #                                 c, env, 20)
    #                     test, train = run_training_instance(agent, env,
    #                                 episodes, False)
    #                     greedy_training[i,j,k,l] = train
    #                     greedy_testing[i,j,k,l] = test
    #                 elif i==3: # run continuous for lunar lander
    #                     env = gym.make(env_name)
    #                     # use 20 buckets
    #                     agent = ContinuousReinforcementAgents.GreedyAgent(alpha, gamma,
    #                                 c, env, 20)
    #                     test, train = run_training_instance(agent, env,
    #                                 episodes, False)
    #                     greedy_training[i,j,k,l] = train
    #                     greedy_testing[i,j,k,l] = test
    #     print('\tFinished running on {} for all constants.'.format(env_name))

    # # save the epsilon greedy data table
    # np.save('../models/greedy_training', greedy_training)
    # np.save('../models/greedy_testing', greedy_testing)

    print("\nRunning on Deep Learning Agents")

    deep_discount = [.05, .01] # [.1, .05, .01, .005]
    deep_epsilon = [1., .8] #[1., .8, .6, .4]
    deep_training = np.empty([4,4,4,4, 10000, 2])
    deep_testing = np.empty([4,4,4,4, 400, 2])
    for i, env_name in enumerate(environments):
        for j, alpha in enumerate(alphas):
            for k, gamma in enumerate(deep_discount):
                for l, c in enumerate(deep_epsilon):
                    if i==0: # run agent for taxi
                        env = gym.make(env_name)
                        agent = DeepReinforcementAgents.TaxiAgent(alpha, gamma,
                                    c, env)
                        test, train = run_training_instance(agent, env,
                                    episodes, False)
                        deep_training[i,j,k,l] = train
                        deep_testing[i,j,k,l] = test
                    elif i==1: # run agent for mountain car
                        env = gym.make(env_name)
                        agent = DeepReinforcementAgents.MountainCarAgent(alpha, gamma,
                                    c, env)
                        test, train = run_training_instance(agent, env,
                                    episodes, False)
                        deep_training[i,j,k,l] = train
                        deep_testing[i,j,k,l] = test
                    elif i ==2: # run continuous for cart pole
                        env = gym.make(env_name)
                        # use 5 buckets
                        agent = DeepReinforcementAgents.CartPoleAgent(alpha, gamma,
                                    c, env)
                        test, train = run_training_instance(agent, env,
                                    episodes, False)
                        deep_training[i,j,k,l] = train
                        deep_testing[i,j,k,l] = test
                    elif i==3: # run continuous for lunar lander
                        env = gym.make(env_name)
                        # use 10 buckets
                        agent = DeepReinforcementAgents.LunarLanderAgent(alpha, gamma,
                                    c, env)
                        test, train = run_training_instance(agent, env,
                                    episodes, False)
                        deep_training[i,j,k,l] = train
                        deep_testing[i,j,k,l] = test
        print('\tFinished running on {} for all constants.'.format(env_name))

    # save the deep q network data to file
    np.save('../models/deep_testing', deep_testing)
    np.save('../models/deep_training', deep_training)





if __name__ == "__main__":
    main()
