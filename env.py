"""Equity trading environment that learns between buy/hold/sell

Environment for trading on the equity market. 
States: Last t minute prices, Cash in hand, Holdings
Actions: 0 = Buy
         1 = Hold
         2 = Sell
Rewards: Total Return at end of episode
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import gym
# import tgym
# from autoencoder import autoencoder

plt.style.use('dark_background')
mpl.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 15,
        "lines.linewidth": 1,
        "lines.markersize": 8
    }
)



class TradingEnv(gym.Env):
    """
    Sublcass of openAI's gym.env. Used to create the right environment
    """
    _actions = {
        'hold': np.array([1, 0, 0]),
        'buy': np.array([0, 1, 0]),
        'sell': np.array([0, 0, 1])
    }

    _positions = {
        'flat': np.array([1, 0, 0]),
        'long': np.array([0, 1, 0]),
        'short': np.array([0, 0, 1])
    }

    def __init__(self, data_generator, 
                 episode_length=1000, 
                 trading_fee=0.0005, 
                 time_fee=0.00004, 
                 history_length=2,
                 stop_loss=-5,
                 pos_size = 500):
        """
        Init Function
        
        data_generator: self explanatory. Generates data for the environment of type DataGenerator
        holdings: All current holdings
        episode_length: number of steps to use. TODO: Figure out optimal number of steps
        trading_fee: % fee for trading
        time_fee: Fee for holding for too long
        history_length: Last t prices where t is in minutes
        pos_size: Size of position we take

        """
        assert history_length > 0
        self._data_generator = data_generator
        self._first_render = True
        self._trading_fee = trading_fee
        self._time_fee = time_fee
        self._episode_length = episode_length
        self.n_actions = 3
        self._prices_history = []
        self._history_length = history_length
        self._stoploss = stop_loss
        self._profit_taken = profit_taken 
        self._holding_position = []  # keeps track of position entry details
        self._pos_size = pos_size

        self.reset()

        # Actions spaces
        
     def reset(self):
        """
        Reset the trading environment. Reset rewards, data generator...
        Returns:
            observation (numpy.array): observation of the state
        """
        self._iteration = 0
        self._data_generator.rewind()
        self._total_reward = 0
        self._total_pnl = 0
        self._position = self._positions['flat']
        self._entry_price = 0
        self._exit_price = 0
        self._closed_plot = False
        self._holding_position = []

        for i in range(self._history_length):
            self._prices_history.append(next(self._data_generator))

        observation = self._get_observation()
        self.state_shape = observation.shape
        self._action = self._actions['hold']


        return observation

     def step(self,action):
        """Take an action (buy/sell/hold) and computes the immediate reward.
        Args:
            action (numpy.array): Action to be taken, one-hot encoded.
        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        assert action.lower() in _actions # make sure it is a valid action
        action = action.lower()
        self._action = action
        self._iteration += 1
        done = False
        instant_pnl = 0 # pnl in this round
        reward = 0 # reward this round
        info = {}


        if all(self._actions[action]==self._actions['buy']): # if we are to go long and ...

            if all(self._position == self._positions['flat']): # if we currently have nothing in hand
                self._position = self._positions['long'] #then we go long
                self._entry_price = self._prices_history[-1]  # We enter at the last closing price

            elif all(self._position == self._positions['short']): # if we were short
                self._exit_price = self._prices_history[-1] # We exit at the last closing price
                instant_pnl = self._trading_fee*self._pos_size*(self._entry_price - self._exit_price)  # pnl is just (entry-exit)*tradingfee* pos_size
                self._position = self._positions['flat']  # our position is now flat because we no longer hold anything
                self._entry_price = 0 # and our entry price resets to 0 

        elif all(self._actions[action]==self._actions['sell']): # if we are going short and ...

            if all(self._position == self._positions['flat']): # if we currently have nothing in hand
                self._position = self._positions['short'] # then we go short
                self._entry_price = self._prices_history[-1]  # We enter at the last closing price

            elif all(self._position == self._positions['long']): # if we were long
                self._exit_price = self._prices_history[-1] # We exit at the last closing price
                instant_pnl = self._trading_fee*self._pos_size*(self._entry_price - self._exit_price)  # pnl is just (entry-exit)*tradingfee* pos_size
                self._position = self._positions['flat']  # our position is now flat because we no longer hold anything
                self._entry_price = 0 # and our entry price resets to 0 


        else: # no decision by the brain

            if all(self._position == self._positions['long']) or all(self._position == self._positions['long']): # If we are currently short or long
                reward -= self._time_fee*pos_size*self._entry_price        # Punish for holding too long
  

        reward += instant_pnl  # reward is just profit added. Might change to percent profit
        self._total_pnl += instant_pnl # Total pnl update
        self._total_reward += reward # Total reward update

        # Game over logic
        try:
            self._prices_history.append(next(self._data_generator))
        except StopIteration:
            done = True
            info['status'] = 'No more data.'
        if self._iteration >= self._episode_length:
            done = True
            info['status'] = 'Time out.'
        if self._closed_plot:
            info['status'] = 'Closed plot'

        observation = self._get_observation()
        return observation, reward, done, info    
