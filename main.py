"""
Example running the trading environment
"""

import env
import sqlgen
# import random
import pg
import ddqn
import numpy as np
from colorama import Fore, Back, init
import tensorflow as tf
from keras import backend as K
from networks import Networks

# Start the generator for training
init(autoreset=True)
generatorTrain = sqlgen.SQLStreamer(configFile='config.json',
                                    numDays=5,
                                    YearBegin=2014,
                                    YearEnd=2016,
                                    scrip='RELIANCE')

# Avoid Tensorflow eats up GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


# setup the environment
trading_fee = 0.0005
time_fee = 0.000004
# history_length number of historical states in the observation vector.
history_length = 30
episodes = 15
episode_length = 300
environment = env.TradingEnv(data_generator=generatorTrain,
                             episode_length=episode_length,
                             trading_fee=trading_fee,
                             time_fee=time_fee,
                             history_length=history_length,
                             pos_size=500,
                             )

state = environment.reset()
state_size = len(state)
action_size = len(environment._actions)


agent = ddqn.DQNAgent(state_size, action_size)  # , history_length)
batch_size = 100
pos = 0
scoreTot = 0
scoreNeg = 0
for e in range(episodes):
    state = environment.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    while not done:
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = environment.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            # loss = agent.train_model/()
            if environment._total_reward > 0:
                pos += 1
                scoreTot += environment._total_reward
                print(Fore.GREEN + "episode: {}/{}, score: {:.3f}"
                      .format(e, episodes, np.float(environment._total_reward)))
            elif environment._total_reward < 0:
                scoreNeg += environment._total_reward
                print(Fore.RED + "episode: {}/{}, score: {:.3f}"
                      .format(e, episodes, np.float(environment._total_reward)))
            else:
                print("episode: {}/{}, score: {:.3f}"
                      .format(e, episodes, np.float(environment._total_reward)))
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
print(pos/episodes)
print(scoreTot + scoreNeg)

generatorTest = sqlgen.SQLStreamer(configFile='config.json',
                                   numDays=5,
                                   YearBegin=2017,
                                   YearEnd=2017,
                                   scrip='RELIANCE')
# episode_length = 300
environment = env.TradingEnv(data_generator=generatorTrain,
                             episode_length=episode_length,
                             trading_fee=trading_fee,
                             time_fee=time_fee,
                             history_length=history_length,
                             pos_size=500,
                             )

# Running the agent
done = False
state = environment.reset()
score = 0
while not done:  # not done:
    # print(state)
    state = np.reshape(state, [1, state_size])
    try:
        # print(state)
        # print(action)
        action = agent.act(state)
        # print("We're fine")
    except Exception as e:
        print(e)
        print(state.shape)
    # a_t = np.zeros(action_size)
    # action = np.argmax(action)
    # print(action)
#
    # a_t[action] = 1
    state, _, done, info = environment.step(action)
    # if 'status' in info and info['status'] == 'Closed plot':
    #     done = True
    if done:
        score += environment._total_reward
        print(Fore.CYAN + "Total so far: %f" % (score))
        # loss = agent.train_model/()
        if environment._total_reward > 0:
            print(Fore.GREEN + "score: {:.3f}"
                  .format(np.float(environment._total_reward)))
        elif environment._total_reward < 0:
            print(Fore.RED + "score: {:.3f}"
                  .format(np.float(environment._total_reward)))
        else:
            print("score: {:.3f}"
                  .format(np.float(environment._total_reward)))
        print(score)
        state = environment.reset()
        done = False
    # environment.render()
