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
# import gym
import tgym
import sklearn.preprocessing as skp
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


class TradingEnv(tgym.Env):
    """
    Sublcass of openAI's gym.env. Used to create the right environment
    """
    _actions = {
        'sell': 1,
        'hold': 0,
        'buy': 2
    }

    _positions = {
        'short': 1,
        'flat': 0,
        'long': 2
    }

    def __init__(self, data_generator,
                 episode_length=1000,
                 trading_fee=0.0005,
                 time_fee=0.00004,
                 history_length=2,
                 max_hold=200,
                 pos_size=500):
        """
        Init Function
        data_generator: self explanatory. Generates data for the
        environment of type DataGenerator
        holdings: All current holdings
        episode_length: number of steps to use, in minutes
                TODO: Figure out optimal number of steps
        trading_fee: % fee for trading
        time_fee: Fee for holding for too long
        history_length: Last t prices where t is in minutes
        pos_size: Size of position we take
        perfect scores are benchmark scores, change their definition in step. Currently buy and hold

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
        self._pos_size = pos_size
        self._epiCounter = 0
        self._max_hold = max_hold
        self._spread_coefficients = [pos_size]

        self.reset()

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
        self._trade_duration = 0
        self._exit_price = 0
        self._closed_plot = False
        self._first_render = True
        self._epiCounter += 1

        try:
            for i in range(self._history_length):
                self._prices_history.append(next(self._data_generator))
            observation = self._get_observation()
            self.state_shape = observation.shape
            self._action = self._actions['hold']
        except Exception as e:  # Tries another reset if this one fails
            print(e)
            observation = self.reset()

        return observation

    def calc_holding(self, reward):

        if self._position == self._positions['long']:
            self._trade_duration += 1
            reward = (self._prices_history[-1] - self._entry_price) * (self._trade_duration / self._max_hold)
            reward = reward - (reward * self._trading_fee)
        elif self._position == self._positions['short']:
            self._trade_duration += 1
            reward = -1 * (self._prices_history[-1] - self._entry_price) * (self._trade_duration / self._max_hold)
            reward = reward - (reward * self._trading_fee)

        return reward * self._pos_size

    def calc_close(self, reward):

        if self._position == self._positions['long']:
            reward = (self._prices_history[-1] - self._entry_price)
            reward = reward - (reward * self._trading_fee)
        elif self._position == self._positions['short']:
            reward = -1 * (self._prices_history[-1] - self._entry_price)
            reward = reward - (reward * self._trading_fee)

        self._trade_duration = 0
        self._entry_price = 0
        self._position = self._positions['flat']

        return reward * self._pos_size

    def step(self, action):
        """Take an action (buy/sell/hold) and computes the immediate reward.
        Args:
            action (numpy.array): Action to be taken
        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        # assert action.lower() in self._actions  # make sure it is a valid action
        # action = action.lower()
        self._action = action
        self._iteration += 1
        done = False
        instant_pnl = 0  # pnl in this round
        reward = 0  # reward this round
        info = {}
        # print(action)
        if action == self._actions['hold']:
            reward = self.calc_holding(reward)
        if action == self._actions['buy']:
            if self._position == self._positions['long']:
                reward = self.calc_holding(reward)
            elif self._position == self._positions['short']:
                reward = self.calc_close(reward)
            elif self._position == self._positions['flat']:
                self._position = self._positions['long']
                self._entry_price = self._prices_history[-1]
        if action == self._actions['sell']:
            # print("here")
            if self._position == self._positions['short']:
                reward = self.calc_holding(reward)
            elif self._position == self._positions['long']:
                reward = self.calc_close(reward)
            elif self._position == self._positions['flat']:
                self._position = self._positions['short']
                self._entry_price = self._prices_history[-1]

        instant_pnl = reward
        if self._trade_duration == 0:
            reward -= 0  # increase to encourage longer trades
        self._total_pnl += instant_pnl  # Total pnl update
        self._total_reward += reward  # Total reward update

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

    def _handle_close(self, evt):
        self._closed_plot = True

    def render(self, savefig=True, filename='myfig'):
        """Matlplotlib rendering of each step.
        Args:
            savefig (bool): Whether to save the figure as an image or not.
            filename (str): Name of the image file.
        """
        if self._first_render:
            try:
                plt.close()
            except:
                pass
            self._f, self._ax = plt.subplots(
                len(self._spread_coefficients) + int(len(self._spread_coefficients) > 1),
                sharex=True
            )
            if len(self._spread_coefficients) == 1:
                self._ax = [self._ax]
            self._f.set_size_inches(12, 6)
            self._first_render = False
            self._f.canvas.mpl_connect('close_event', self._handle_close)

        # if len(self._spread_coefficients) > 1:
        #     # TODO: To be checked
        #     for prod_i in range(len(self._spread_coefficients)):
        #         bid = self._prices_history[-1][2 * prod_i]
        #         ask = self._prices_history[-1][2 * prod_i + 1]
        #         self._ax[prod_i].plot([self._iteration, self._iteration + 1],
        #                               [bid, bid], color='white')
        #         self._ax[prod_i].plot([self._iteration, self._iteration + 1],
        #                               [ask, ask], color='white')
        #         self._ax[prod_i].set_title('Product {} (spread coef {})'.format(
        #             prod_i, str(self._spread_coefficients[prod_i])))

        # Spread price
        # prices = self._prices_history[-1]
        bid, ask = self._prices_history[-1], self._prices_history[-1]
        self._ax[-1].plot([self._iteration, self._iteration + 1],
                          [bid, bid], color='white')
        self._ax[-1].plot([self._iteration, self._iteration + 1],
                          [ask, ask], color='white')
        ymin, ymax = self._ax[-1].get_ylim()
        yrange = ymax - ymin
        # print("ACTION:")
        # print(self._action)
        # print("POSITION:")
        # print(self._position)
        if (self._action == self._actions['sell']):  # and all(self._position != self._positions['short']):
            self._ax[-1].scatter(self._iteration + 0.5, bid + 0.03 *
                                 yrange, color='orangered', marker='v')
        elif (self._action == self._actions['buy']):  # and all(self._position != self._positions['long']):
            self._ax[-1].scatter(self._iteration + 0.5, ask - 0.03 *
                                 yrange, color='lawngreen', marker='^')
        elif (self._action == self._actions['hold']):
            pass
        plt.suptitle('Cumulated Reward: ' + "%.2f" % self._total_reward + ' ~ ' +
                     'Cumulated PnL: ' + "%.2f" % self._total_pnl + ' ~ ' +
                     'Position: ' + ['short', 'flat', 'long'][self._position] + ' ~ ' +
                     'Entry Price: ' + "%.2f" % self._entry_price  # + ' ~ \n'
                     # 'Perfect Score:' + "%.2f" % self._perfect_score
                     )
        self._f.tight_layout()
        plt.xticks(range(self._iteration)[::5])
        plt.xlim([max(0, self._iteration - 80.5), self._iteration + 0.5])
        plt.subplots_adjust(top=0.85)
        plt.pause(0.01)
        if savefig:
            plt.savefig(filename + str(self._epiCounter))

    def _get_observation(self):
        """Concatenate all necessary elements to create the observation.
        Returns:
            numpy.array: observation array.
        """
        prices = self._prices_history[-self._history_length:]
        # print(prices)
        # print(self._position)
        # print(self._entry_price)
        prices = skp.minmax_scale(prices)
        # print(prices)

        return np.concatenate((np.array(prices),
                               [self._entry_price],
                               [self._position],
                               [self._trade_duration]))
