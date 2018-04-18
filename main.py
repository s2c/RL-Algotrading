"""
Example running the trading environment
"""

import env
import sqlgen
# import random
import pg
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
                                    YearEnd=2015,
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
history_length = 60
episodes = 30
episode_length = 300
environment = env.TradingEnv(data_generator=generatorTrain,
                             episode_length=episode_length,
                             trading_fee=trading_fee,
                             time_fee=time_fee,
                             history_length=history_length,
                             stop_loss=-5,
                             pos_size=500,
                             profit_taken=0)

state = environment.reset()
state_size = len(state)
action_size = len(environment._actions)


agent = pg.A2CAgent(state_size, action_size, history_length)
agent.actor = Networks.actor_network(state_size, action_size, agent.actor_lr)
agent.critic = Networks.critic_network(state_size, agent.value_size, agent.critic_lr)

for e in range(episodes):
    x_t = environment.reset()
    x_t = np.reshape(x_t, [1, state_size])
    done = False
    while not done:
        loss = 0
        r_t = 0
        a_t = np.zeros([action_size])
        # Get action
        action_idx, policy = agent.act(x_t)
        a_t[action_idx] = 1
        x_t, r_t, done, _ = environment.step(a_t)
        if done:
            loss_actor, loss_critic = agent.train_model()
            if environment._total_reward > 0:
                print(Fore.GREEN + "episode: {}/{}, score: {:.3f}, loss_actor: {:.2f}, loss_critic: {:.2f}"
                      .format(e, episodes, np.float(environment._total_reward), loss_actor[0], loss_critic[0]))
            elif environment._total_reward < 0:
                print(Fore.RED + "episode: {}/{}, score: {:.3f}, loss_actor: {:.2f}, loss_critic: {:.2f}"
                      .format(e, episodes, np.float(environment._total_reward), loss_actor[0], loss_critic[0]))
            else:
                print("episode: {}/{}, score: {:.3f}, loss_actor: {:.2.}, loss_critic: {:.2.}"
                      .format(e, episodes, np.float(environment._total_reward), loss_actor[0], loss_critic[0]))
            break
        agent.append_sample(x_t, action_idx, r_t)
        if e % 50 == 0:
            loss = agent.save_model("models/a2c")


generatorTest = sqlgen.SQLStreamer(configFile='config.json',
                                   numDays=5,
                                   YearBegin=2016,
                                   YearEnd=2017,
                                   scrip='RELIANCE')
episode_length = 300
environment = env.TradingEnv(data_generator=generatorTest,
                             episode_length=episode_length,
                             trading_fee=trading_fee,
                             time_fee=time_fee,
                             history_length=history_length,
                             stop_loss=-5,
                             pos_size=500,
                             profit_taken=0)

# Running the agent
done = False
state = environment.reset()

while not done:  # not done:
    # print(state)
    try:
        action, policy = agent.act(state)
        # print("We're fine")
    except Exception as e:
        print(e)
        print(state.shape)
    a_t = np.zeros(action_size)
    print(action)
    a_t[action] = 1
    state, _, done, info = environment.step(a_t)
    # if 'status' in info and info['status'] == 'Closed plot':
    #     done = True
    if done:
        state = environment.reset()
        done = False
    environment.render()
