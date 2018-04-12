"""
Example running the trading environment
"""

import env
import sqlgen
# import random
import ddqn
import numpy as np
from colorama import Fore, Back,init

init(autoreset=True)
generatorTrain = sqlgen.SQLStreamer(configFile='config.json',
                                    numDays=30,
                                    YearBegin=2014,
                                    YearEnd=2015,
                                    scrip='RELIANCE')
trading_fee = 0.0005
time_fee = 0.000004
# history_length number of historical states in the observation vector.
history_length = 30
episodes = 10000
episode_length = 300
environment = env.TradingEnv(data_generator=generatorTrain,
                             episode_length=episode_length,
                             trading_fee=trading_fee,
                             time_fee=time_fee,
                             history_length=history_length,
                             stop_loss=-5,
                             pos_size=5000,
                             profit_taken=0)
state = environment.reset()
state_size = len(state)
gamma = 0.9
epsilon_min = 0.01
batch_size = 64
action_size = len(environment._actions)
memory_size = 3000
train_interval = 10
learning_rate = 0.001
agent = ddqn.DDQNAgent(state_size=state_size,
                       action_size=action_size,
                       # memory_size=memory_size,
                       # episodes=episodes,
                       # episode_length=episode_length,
                       # train_interval=train_interval,
                       # gamma=gamma,
                       # learning_rate=learning_rate,
                       # batch_size=batch_size,
                       # epsilon_min=epsilon_min
                       )

for e in range(episodes):
    state = environment.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(episode_length):
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = environment.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        # print(type(reward))
        if done:
            agent.update_target_model()
            if environment._total_reward > 0:
                print(Fore.GREEN + "episode: {}/{}, score: {:.3f}, e: {:.2}"
                      .format(e, episodes, np.float(environment._total_reward), agent.epsilon))
            elif environment._total_reward < 0:
                print(Fore.RED + "episode: {}/{}, score: {:.3f}, e: {:.2}"
                      .format(e, episodes, np.float(environment._total_reward), agent.epsilon))
            else:
                print("episode: {}/{}, score: {:.3f}, e: {:.2}"
                      .format(e, episodes, np.float(environment._total_reward), agent.epsilon))
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

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
                             pos_size=5000,
                             profit_taken=0)

# Running the agent
done = False
state = environment.reset()
# print(state)
while True:  # not done:
    # print(state)
    try:
        action = agent.act(state)
        # print("We're fine")
    except Exception as e:
        print(e)
        print(state.shape)
    state, _, done, info = environment.step(action)
    if done or 'status' in info and info['status'] == 'Closed plot':
        done = True
    environment.render()
