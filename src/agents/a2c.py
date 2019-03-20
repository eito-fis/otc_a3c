
import logging
import os

import numpy as np

import tensorflow as tf
from tensorflow import keras

class ProbabilityDistribution(keras.Model):
    def call(self, logits):
        # Sample a random action from logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
    def __init__(self,
                 num_actions=None,
                 actor_fc=None,
                 critic_fc=None):
        #super().__init__('mlp_policy')
        super().__init__()

        self.actor_fc = [keras.layers.Dense(neurons, activation="relu") for neurons in actor_fc]
        self.critic_fc = [keras.layers.Dense(neurons, activation="relu") for neurons in critic_fc]

        self.value = keras.layers.Dense(1, name='value')
        self.actor_logits = keras.layers.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)

        actor_logits = x
        for layer in self.actor_fc:
            actor_logits = layer(actor_logits)
        actor_logits = self.actor_logits(actor_logits)

        critic_logits = x
        for layer in self.critic_fc:
            critic_logits = layer(critic_logits)
        value = self.value(critic_logits)

        return actor_logits, value

    def get_action_value(self, obs):
        logits, value = self.predict(obs)

        action = self.dist.predict(logits)
        action = action[0]
        value = value[0][0]

        return action, value

class A2CAgent:
    def __init__(self,
                 model,
                 env,
                 gamma=0.99,
                 value=0.5,
                 entropy=0.0001,
                 summary_writer=None,
                 log_period=10,
                 save_period=100,
                 save_path="/tmp/model"):

        self.gamma = gamma
        self.value = value
        self.entropy = entropy

        self.log_period = log_period
        self.save_period = save_period
        self.summary_writer = summary_writer
        self.save_path = save_path

        self.env = env
        self.model = model
        self.model.compile(
            optimizer=keras.optimizers.RMSprop(lr=0.0007),
            loss=[self._logits_loss, self._value_loss]
        )
    
    def train(self, n_epochs=10000):
        episode_rewards = []
        for epoch in range(n_epochs):
            obs = self.env.reset()
            actions = []
            rewards = []
            dones = []
            values = []
            observations = []
            
            logging.info("Episode generation started...")
            while len(dones) == 0 or dones[-1] != 1:
                observations.append(obs)
                _action, _value = self.model.get_action_value(obs[None, :])
                obs, _reward, _done, _ = self.env.step(_action)

                actions.append(_action)
                values.append(_value)
                rewards.append(_reward)
                dones.append(_done)

            episode_rewards.append(sum(rewards))
            logging.info("Episode: %03d, Reward: %03d" % (len(episode_rewards), episode_rewards[-1]))

            returns, advs = self._returns_advantages(rewards, dones, values)
            acts_and_advs = np.concatenate([np.array(actions)[:, None], advs[:, None]], axis=-1)

            logging.info("Training started...")
            #logging.info("{}".format(self.model.metrics_names))
            losses = self.model.train_on_batch(np.array(observations), [acts_and_advs, returns])
            logging.info("[%d/%d] Losses: %s" % (epoch + 1, epoch, losses))

            if epoch % self.log_period == 0:
                with self.summary_writer.as_default():
                    tf.summary.scalar("Episode Reward", episode_rewards[-1], epoch)
                    tf.summary.scalar("Episode Loss", np.sum(losses), epoch)
            if epoch % self.save_period == 0:
                _save_path = os.path.join(self.save_path, "model_{}".format(epoch))
                self.model.save_weights(_save_path)

        return episode_rewards

    def _returns_advantages(self, rewards, dones, values):
        returns = np.zeros(len(rewards) + 1)

        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]

        # advantages are returns - baseline, value estimates in our case
        advantages = returns - np.array(values)
        return returns, advantages
    
    def _value_loss(self, returns, value):
        return self.value * keras.losses.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits):
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        weighted_sparse_crossentropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)

        policy_loss = weighted_sparse_crossentropy(actions, logits, sample_weight=advantages)
        entropy_loss = keras.losses.categorical_crossentropy(logits, logits, from_logits=True)

        return policy_loss - (self.entropy * entropy_loss)
