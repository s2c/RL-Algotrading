"""
Example running the trading environment
"""

import env
import sqlgen
import random


generator = sqlgen.SQLStreamer(configFile='config.json',
                               numDays=3,
                               YearBegin=2014,
                               YearEnd=2017,
                               scrip='RELIANCE')

episode_length = 1000
trading_fee = 0.0005
time_fee = 0.00004
# history_length number of historical states in the observation vector.
history_length = 30

# environment = SpreadTrading(spread_coefficients=[1],
#                             data_generator=generator,
#                             trading_fee=trading_fee,
#                             time_fee=time_fee,
#                             history_length=history_length,
#                             episode_length=episode_length)
environment = env.TradingEnv(data_generator=generator,
                             episode_length=episode_length,
                             trading_fee=trading_fee,
                             time_fee=time_fee,
                             history_length=history_length,
                             stop_loss=-5,
                             pos_size=500,
                             profit_taken=0)

environment.render()
while True:
    actions = [[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]]
    environment.step(random.choice(actions))
    environment.render()
