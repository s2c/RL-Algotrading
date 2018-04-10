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
                 pos_size=500,
                 profit_taken=0):
        """
        Init Function
        data_generator: self explanatory. Generates data for the
        environment of type DataGenerator
        holdings: All current holdings
        episode_length: number of steps to use.
                TODO: Figure out optimal number of steps
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
        self._exit_price = 0
        self._closed_plot = False
        self._holding_position = []

        for i in range(self._history_length):
            self._prices_history.append(next(self._data_generator))

        observation = self._get_observation()
        self.state_shape = observation.shape
        self._action = self._actions['hold']

        return observation

    def step(self, action):
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
        assert action.lower() in self._actions  # make sure it is a valid action
        action = action.lower()
        self._action = action
        self._iteration += 1
        done = False
        instant_pnl = 0  # pnl in this round
        reward = 0  # reward this round
        info = {}

        # if we are to go long and ...
        if all(self._actions[action] == self._actions['buy']):

            # if we currently have nothing in hand
            if all(self._position == self._positions['flat']):
                self._position = self._positions['long']  # then we go long
                # We enter at the last closing price
                self._entry_price = self._prices_history[-1]

            # if we were short
            elif all(self._position == self._positions['short']):
                # We exit at the last closing price
                self._exit_price = self._prices_history[-1]
                instant_pnl = (self._entry_price - self._exit_price) - \
                    (self._trading_fee * self._pos_size * (self._entry_price - self._exit_price))  # pnl is just entry-exit-transaction costs
                self._position = self._positions['flat']  # our position is now flat because we no longer hold anything
                self._entry_price = 0  # and our entry price resets to 0

        # if we are going short and ...
        elif all(self._actions[action] == self._actions['sell']):

            # if we currently have nothing in hand
            if all(self._position == self._positions['flat']):
                self._position = self._positions['short']  # then we go short
                # We enter at the last closing price
                self._entry_price = self._prices_history[-1]

            # if we were long
            elif all(self._position == self._positions['long']):
                # We exit at the last closing price
                self._exit_price = self._prices_history[-1]
                instant_pnl = (self._entry_price - self._exit_price) - \
                    (self._trading_fee * self._pos_size * (self._entry_price - self._exit_price))  # pnl is just entry-exit-transaction costs
                # our position is now flat because we no longer hold anything
                self._position = self._positions['flat']
                self._entry_price = 0  # and our entry price resets to 0

        else:  # no decision by the brain

            # If we are currently short or long
            if all(self._position == self._positions['long']) or all(self._position == self._positions['short']):
                reward -= self._time_fee * self._pos_size * self._entry_price        # Punish for holding too long

        reward += instant_pnl  # reward is just profit added. Might change to percent profit
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

    def render(self, savefig=False, filename='myfig'):
        """Matlplotlib rendering of each step.
        Args:
            savefig (bool): Whether to save the figure as an image or not.
            filename (str): Name of the image file.
        """
        if self._first_render:
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
        if (self._action == self._actions['sell']).all():
            self._ax[-1].scatter(self._iteration + 0.5, bid + 0.03 *
                                 yrange, color='lawngreen', marker='v')
        elif (self._action == self._actions['buy']).all():
            self._ax[-1].scatter(self._iteration + 0.5, ask - 0.03 *
                                 yrange, color='orangered', marker='^')
        plt.suptitle('Cumulated Reward: ' + "%.2f" % self._total_reward + ' ~ ' +
                     'Cumulated PnL: ' + "%.2f" % self._total_pnl + ' ~ ' +
                     'Position: ' + ['flat', 'long', 'short'][list(self._position).index(1)] + ' ~ ' +
                     'Entry Price: ' + "%.2f" % self._entry_price)
        self._f.tight_layout()
        plt.xticks(range(self._iteration)[::5])
        plt.xlim([max(0, self._iteration - 80.5), self._iteration + 0.5])
        plt.subplots_adjust(top=0.85)
        plt.pause(0.01)
        if savefig:
            plt.savefig(filename)

        def _get_observation(self):
            """Concatenate all necessary elements to create the observation.
            Returns:
                numpy.array: observation array.
            """
            return np.concatenate(
                [prices for prices in self._prices_history[-self._history_length:]] +
                [
                    np.array([self._entry_price]),
                    np.array(self._position)
                ]
            )
