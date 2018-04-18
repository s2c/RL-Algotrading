import numpy as np


class A2CAgent:
    """
    The A2C agent
    """

    def __init__(self, state_size, action_size, history_length):
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1  # Reward is scalar
        self.observe = 0
        self.history_length = history_length  # Replaces frame per action # not used atm

        # Hyper-parameters for Policy Gradient
        self.gamma = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001

        # Model for policy and critic network
        self.actor = None
        self.critic = None

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

    def act(self, state):
        """
        # using the output of policy network, pick action stochastically (Stochastic Policy)
        """
        state = state.flatten()
        state = state.reshape(1, -1, 1)
        policy = self.actor.predict(state).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0], policy

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

        # save <s, a ,r> of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

# update policy network every episode
    def train_model(self):
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)
        # Standardized discounted rewards
        discounted_rewards -= np.mean(discounted_rewards)
        if np.std(discounted_rewards):
            discounted_rewards /= np.std(discounted_rewards)
        else:
            self.states, self.actions, self.rewards = [], [], []
            print ('std = 0!')
            return 0

        update_inputs = np.zeros(((episode_length,) + (self.state_size,)))  # Episode_lengthx64x64x4

        # Episode length is like the minibatch size in DQN
        for i in range(episode_length):
            update_inputs[i, :] = self.states[i]

        # Prediction of state values for each state appears in the episode
        print(update_inputs)
        values = self.critic.predict(update_inputs.reshape(episode_length, -1, 1))

        # Similar to one-hot target but the "1" is replaced by Advantage Function i.e. discounted_rewards R_t - Value
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            advantages[i][self.actions[i]] = discounted_rewards[i] - values[i]

        actor_loss = self.actor.fit(update_inputs.reshape(len(update_inputs), -1, 1), advantages, nb_epoch=1, verbose=0)
        critic_loss = self.critic.fit(update_inputs.reshape(len(update_inputs), -1, 1), discounted_rewards, nb_epoch=1, verbose=0)

        self.states, self.actions, self.rewards = [], [], []

        return actor_loss.history['loss'], critic_loss.history['loss']

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5", overwrite=True)
        self.critic.save_weights(name + "_critic.h5", overwrite=True)

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")
