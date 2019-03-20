import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import threading
import multiprocessing

import gym
import numpy as np

from queue import Queue

import tensorflow as tf
from tensorflow import keras

def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
    """Helper function to store score and print statistics.
    Arguments:
    episode: Current episode
    episode_reward: Reward accumulated over the current episode
    worker_idx: Which thread (worker)
    global_ep_reward: The moving average of the global reward
    result_queue: Queue storing the moving average of the scores
    total_loss: The total loss accumualted over the current episode
    num_steps: The number of steps the episode took to complete
    """
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    print(
        f"Episode: {episode} | "
        f"Moving Average Reward: {int(global_ep_reward)} | "
        f"Episode Reward: {int(episode_reward)} | "
        f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
        f"Steps: {num_steps} | "
        f"Worker: {worker_idx}"
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

class ProbabilityDistribution(keras.Model):
    def call(self, logits):
        # Sample a random action from logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class ActorCriticModel(keras.Model):
    def __init__(self,
             num_actions=None,
             state_size=None,
             actor_fc=None,
             critic_fc=None):
        #super().__init__('mlp_policy')
        super().__init__()

        self.actor_fc = [keras.layers.Dense(neurons, activation="relu") for neurons in actor_fc]
        self.critic_fc = [keras.layers.Dense(neurons, activation="relu") for neurons in critic_fc]

        self.actor_logits = keras.layers.Dense(num_actions, name='policy_logits')
        self.value = keras.layers.Dense(1, name='value')

        self.actor_model = tf.keras.Sequential(self.actor_fc + [self.actor_logits])
        self.actor_model.build((None, state_size))
        self.critic_model = tf.keras.Sequential(self.critic_fc + [self.value])
        self.critic_model.build((None, state_size))

        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        actor_logits = self.actor_model(inputs)
        value = self.critic_model(inputs)

        return actor_logits, value

    def get_action_value(self, obs):
        logits, value = self.predict(obs)

        action = self.dist.predict(logits)
        action = action[0]
        value = value[0][0]

        return action, value

class MasterAgent():
    def __init__(self,
                 env_func=None,
                 num_actions=2,
                 state_size=4,
                 learning_rate=0.0001,
                 actor_fc=None,
                 critic_fc=None,
                 save_path="/tmp/a3c"):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.num_actions = num_actions
        self.state_size = state_size
        self.actor_fc = actor_fc
        self.critic_fc = critic_fc

        self.opt = keras.optimizers.Adam(learning_rate)
        self.env_func = env_func

        self.global_model = ActorCriticModel(num_actions=self.num_actions,
                                             state_size=self.state_size,
                                             actor_fc=self.actor_fc,
                                             critic_fc=self.critic_fc)

    def train(self):
        res_queue = Queue()

        workers = [Worker(idx=i,
                   num_actions=self.num_actions,
                   state_size=self.state_size,
                   actor_fc=self.actor_fc,
                   critic_fc=self.critic_fc,
                   global_model=self.global_model,
                   opt=self.opt,
                   result_queue=res_queue,
                   env_func=self.env_func,
                   save_path=self.save_path) for i in range(multiprocessing.cpu_count())]
                   #save_path=self.save_path) for i in range(1)]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

        moving_average_rewards = []  # record episode reward to plot
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        [w.join() for w in workers]
        print("Done!")
        self.play()
        return moving_average_rewards

    def play(self):
        env = gym.make('CartPole-v0').unwrapped
        state = env.reset()
        model = self.global_model
        model_path = os.path.join(self.save_path, 'model.h5')
        print('Loading model from: {}'.format(model_path))
        model.load_weights(model_path)
        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                env.render(mode='rgb_array')
                action, value = model.get_action_value(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()

class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self,
                 idx=0,
                 num_actions=2,
                 state_size=4,
                 actor_fc=None,
                 critic_fc=None,
                 global_model=None,
                 opt=None,
                 result_queue=None,
                 env_func=None,
                 max_episodes=500,
                 save_path='/tmp/a3c/workers',
                 gamma=0.99):
        super(Worker, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size

        self.opt = opt
        self.global_model = global_model
        self.local_model = ActorCriticModel(num_actions=self.num_actions,
                                            state_size=self.state_size,
                                            actor_fc=actor_fc,
                                            critic_fc=critic_fc)
        self.local_model.set_weights(self.global_model.get_weights())
        # Run model to build graph during init because it can't be done async
        _ = self.local_model.get_action_value(
                            tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

        if env_func == None:
            self.env = gym.make('CartPole-v0').unwrapped
        else:
            print("Building environment")
            self.env = env_func(idx)
        print("Environment built!")

        self.result_queue = result_queue
        self.save_path = save_path
        self.worker_idx = idx
        self.ep_loss = 0.0
        self.gamma = gamma
        self.max_episodes = max_episodes

    def run(self):
        total_step = 1
        mem = Memory()
        while Worker.global_episode < self.max_episodes:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0

            time_count = 0
            done = False
            while not done:
                action, _ = self.local_model.get_action_value(
                                    tf.convert_to_tensor(current_state[None, :],
                                    dtype=tf.float32))
                new_state, reward, done, _ = self.env.step(action)
                ep_reward += reward
                mem.store(current_state, action, reward)

                if done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape

                    # Update model
                    with tf.GradientTape() as tape:
                        total_loss  = self.compute_loss(mem, self.gamma)
                    self.ep_loss += tf.reduce_sum(total_loss)

                    # Calculate and apply policy gradients
                    total_grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    self.opt.apply_gradients(zip(total_grads,
                                           self.global_model.trainable_weights))

                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())

                    mem.clear()
                    time_count = 0

                    Worker.global_moving_average_reward = \
                    record(Worker.global_episode, ep_reward, self.worker_idx,
                         Worker.global_moving_average_reward, self.result_queue,
                         self.ep_loss, ep_steps)
                    # We must use a lock to save our model and to print to prevent data races.
                    if ep_reward > Worker.best_score:
                        with Worker.save_lock:
                            print("Saving best model to {}, "
                                "episode score: {}".format(self.save_path, ep_reward))
                            self.global_model.save_weights(
                                os.path.join(self.save_path, 'model.h5')
                            )
                            Worker.best_score = ep_reward
                    Worker.global_episode += 1
                else:
                    ep_steps += 1
                    time_count += 1
                    current_state = new_state
                    total_step += 1
        self.result_queue.put(None)

    def compute_loss(self,
                     memory,
                     gamma):
        # Get discounted rewards
        reward_sum = 0.
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        # Get logits and values
        logits, values = self.local_model(
                                tf.convert_to_tensor(np.vstack(memory.states),
                                dtype=tf.float32))

        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32) - values

        # Calculate our policy loss
        entropy_loss = keras.losses.categorical_crossentropy(logits, logits, from_logits=True)
        weighted_sparse_crossentropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = weighted_sparse_crossentropy(np.array(memory.actions), logits, sample_weight=advantage)
        policy_loss = policy_loss - (0.01 * entropy_loss)

        # Value loss
        value_loss = keras.losses.mean_squared_error(np.array(discounted_rewards), values)

        return policy_loss + (value_loss * 0.5)


if __name__ == '__main__':
  print(args)
  master = MasterAgent()
  if args.train:
    master.train()
  else:
    master.play()

