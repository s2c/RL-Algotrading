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
import matplotlib.pyplot as plt
# Start the generator for training
init(autoreset=True)
generatorTrain = sqlgen.SQLStreamer(configFile='config.json',
                                    numDays=2,
                                    YearBegin=2014,
                                    YearEnd=2017,
                                    scrip='CIPLA')

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
episodes = 300
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
xdata = []
ydata = []
plt.ion()
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.set_xlabel('Time')
ax.set_ylabel('Score')
ax.plot(xdata, ydata, markersize=100)  # add an empty line to the plot
fig.show()  # show the window (figure will be in foreground, but the user may move it to background)

for e in range(episodes):
    state = environment.reset()
    state = np.reshape(state, [1, state_size, 1])
    done = False
    while not done:
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = environment.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size, 1])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(Fore.CYAN + "Total so far: %f" % (scoreTot + scoreNeg))
            xdata.append(e)
            ydata.append(scoreTot + scoreNeg)
            ax.lines[0].set_data( xdata,ydata )
            ax.relim()                  # recompute the data limits
            ax.autoscale_view()         # automatic axis scaling
            fig.canvas.flush_events()   # update the plot and take care of window events (like resizing etc.)
            # plt.pause(0.01)               # wait for next loop iteration
            # time.sleep(0.1)
            # loss = agent.train_model/()
            if environment._total_pnl > 0:
                pos += 1
                scoreTot += environment._total_pnl
                print(Fore.GREEN + "episode: {}/{}, score: {:.3f}, exploration:{:.3f}"
                      .format(e, episodes, np.float(environment._total_pnl), np.float(agent.epsilon)))
                agent.save("DDQNAgent")
            elif environment._total_pnl < 0:
                scoreNeg += environment._total_pnl
                print(Fore.RED + "episode: {}/{}, score: {:.3f}, exploration:{:.3f}"
                      .format(e, episodes, np.float(environment._total_pnl), np.float(agent.epsilon)))
            else:
                print("episode: {}/{}, score: {:.3f}, exploration:{:.3f}"
                      .format(e, episodes, np.float(environment._total_pnl), np.float(agent.epsilon)))
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
print(pos / episodes)
print(scoreTot + scoreNeg)

generatorTest = sqlgen.SQLStreamer(configFile='config.json',
                                   numDays=5,
                                   YearBegin=2018,
                                   YearEnd=2018,
                                   scrip='CIPLA')
# episode_length = 300
environment = env.TradingEnv(data_generator=generatorTest,
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
    state = np.reshape(state, [1, state_size, 1])
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
        score += environment._total_pnl
        print(Fore.CYAN + "Total so far: %f" % (score))
        # loss = agent.train_model/()
        if environment._total_pnl > 0:
            print(Fore.GREEN + "score: {:.3f}"
                  .format(np.float(environment._total_pnl)))
        elif environment._total_pnl < 0:
            print(Fore.RED + "score: {:.3f}"
                  .format(np.float(environment._total_pnl)))
        else:
            print("score: {:.3f}"
                  .format(np.float(environment._total_pnl)))
        state = environment.reset()
        done = False
    # environment.render()
