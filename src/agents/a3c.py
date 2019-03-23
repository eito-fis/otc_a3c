
import os
import pickle
import argparse

import threading
import multiprocessing

import gym
import numpy as np

from queue import Queue

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from collections import Counter

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
    global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    print("Episode: {} | Moving Average Reward: {} | Episode Reward: {} | Loss: {} | Steps: {} | Worker: {}".format(episode, global_ep_reward, episode_reward, int(total_loss / float(num_steps) * 1000) / 1000, num_steps, worker_idx))
    result_queue.put(global_ep_reward)
    return global_ep_reward

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.obs = []
        self.probs = []
        self.values = []
        self.novelty = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.obs=[]
        self.probs = []
        self.novelty = []

class ProbabilityDistribution(keras.Model):
    def call(self, logits):
        # Sample a random action from logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class ActorCriticModel(keras.Model):
    def __init__(self,
                 num_actions=None,
                 state_size=None,
                 conv_size=None,
                 actor_fc=None,
                 critic_fc=None):
        super().__init__()
    
        #self.conv_layers = [hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2",
        #                                   output_shape=conv_size,
        #                                   trainable=False)]
        
        self.actor_fc = [keras.layers.Dense(neurons, activation="relu") for neurons in actor_fc]
        self.critic_fc = [keras.layers.Dense(neurons, activation="relu") for neurons in critic_fc]

        self.actor_logits = keras.layers.Dense(num_actions, name='policy_logits')
        self.value = keras.layers.Dense(1, name='value', activation='relu')

        #self.conv_model = tf.keras.Sequential(self.conv_layers)
        #self.conv_model.build([None] + state_size)

        self.actor_model = tf.keras.Sequential(self.actor_fc + [self.actor_logits])
        self.actor_model.build([None] + state_size)
        self.critic_model = tf.keras.Sequential(self.critic_fc + [self.value])
        self.critic_model.build([None] + state_size)
        #self.critic_model.build((None, self.conv_model.layers[-1].output_shape))

        self.dist = ProbabilityDistribution()
        self.get_action_value(tf.convert_to_tensor(np.random.random((1,) + tuple(state_size)), dtype=tf.float32))

    def call(self, inputs):
        #inputs = self.conv_model(inputs)
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
                 num_episodes=1000,
                 env_func=None,
                 num_actions=2,
                 state_size=[4],
                 conv_size=None,
                 learning_rate=0.0000042,
                 gamma=0.99,
                 entropy_discount=0.05,
                 value_discount=0.1,
                 boredom_thresh=10,
                 update_freq=601,
                 actor_fc=None,
                 critic_fc=None,
                 summary_writer=None,
                 log_period=10,
                 checkpoint_period=10,
                 visual_period=None,
                 visual_path="/tmp/a3c/visuals",
                 save_path="/tmp/a3c",
                 load_path=None):

        self.visual_path = visual_path
        self.save_path = save_path
        self.summary_writer = summary_writer
        self.log_period = log_period
        self.checkpoint_period = checkpoint_period
        self.visual_period = visual_period
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.num_episodes = num_episodes
        self.num_actions = num_actions
        self.state_size = state_size
        self.conv_size = conv_size
        self.actor_fc = actor_fc
        self.critic_fc = critic_fc

        self.gamma = gamma
        self.entropy_discount = entropy_discount
        self.value_discount = value_discount
        self.boredom_thresh = boredom_thresh
        self.update_freq = update_freq

        self.opt = keras.optimizers.Adam(learning_rate)
        self.env_func = env_func

        self.global_model = ActorCriticModel(num_actions=self.num_actions,
                                             state_size=self.state_size,
                                             conv_size=self.conv_size,
                                             actor_fc=self.actor_fc,
                                             critic_fc=self.critic_fc)
        if load_path != None:
            self.global_model.load_weights(load_path)
            print("Loaded model from {}".format(load_path))

    def distributed_train(self):
        res_queue = Queue()

        workers = [Worker(idx=i,
                   num_actions=self.num_actions,
                   state_size=self.state_size,
                   conv_size=self.conv_size,
                   actor_fc=self.actor_fc,
                   critic_fc=self.critic_fc,
                   gamma=self.gamma,
                   entropy_discount=self.entropy_discount,
                   value_discount=self.value_discount,
                   boredom_thresh=self.boredom_thresh,
                   update_freq=self.update_freq,
                   global_model=self.global_model,
                   opt=self.opt,
                   result_queue=res_queue,
                   env_func=self.env_func,
                   max_episodes=self.num_episodes,
                   summary_writer=self.summary_writer,
                   log_period=self.log_period,
                   checkpoint_period=self.checkpoint_period,
                   visual_period=self.visual_period,
                   visual_path=self.visual_path,
                   save_path=self.save_path) for i in range(multiprocessing.cpu_count())]
                   #save_path=self.save_path) for i in range(1)]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

        moving_average_rewards = []
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

    def human_train(self, data_path, train_steps):
        # Load human input from pickle file
        data_file = open(data_path, 'rb')
        memory_list = pickle.load(data_file)
        data_file.close()
        print("Loaded files")

        all_actions = [frame for memory in memory_list for frame in memory.actions]
        all_states = [frame.numpy() for memory in memory_list for frame in memory.states]

        counts = [(len(all_actions) - c) / len(all_actions) for c in list(Counter(all_actions).values())]
        all_weights = [counts[action] for action in all_actions]
        print("Counts: {}".format(Counter(all_actions)))

        print("Starting steps...")
        for train_step in range(train_steps):
            with tf.GradientTape() as tape:
                total_loss = self.compute_loss(all_actions,
                                               all_states,
                                               all_weights,
                                               self.gamma)
        
            # Calculate and apply policy gradients
            total_grads = tape.gradient(total_loss, self.global_model.actor_model.trainable_weights)
            self.opt.apply_gradients(zip(total_grads,
                                             self.global_model.actor_model.trainable_weights))
            print("Step: {} | Loss: {}".format(train_step, total_loss))
            
    def compute_loss(self,
                     all_actions,
                     all_states,
                     all_weights,
                     gamma):

        # Get logits and values
        logits, values = self.global_model(
                                tf.convert_to_tensor(np.vstack(all_states),
                                dtype=tf.float32))

        weighted_sparse_crossentropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = weighted_sparse_crossentropy(np.array(all_actions)[:, None], logits, sample_weight=all_weights)

        return policy_loss

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
    global_moving_average_reward = 0.0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self,
                 idx=0,
                 num_actions=2,
                 state_size=[4],
                 conv_size=None,
                 actor_fc=None,
                 critic_fc=None,
                 gamma=0.99,
                 entropy_discount=None,
                 value_discount=None,
                 boredom_thresh=None,
                 update_freq=None,
                 global_model=None,
                 opt=None,
                 result_queue=None,
                 env_func=None,
                 max_episodes=500,
                 summary_writer=None,
                 log_period=10,
                 checkpoint_period=10,
                 visual_period=25,
                 visual_path='/tmp/a3c/visuals/',
                 save_path='/tmp/a3c/workers'):
        super(Worker, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        self.conv_size = conv_size

        self.opt = opt
        self.global_model = global_model
        self.local_model = ActorCriticModel(num_actions=self.num_actions,
                                            state_size=self.state_size,
                                            conv_size=self.conv_size,
                                            actor_fc=actor_fc,
                                            critic_fc=critic_fc)
        self.local_model.set_weights(self.global_model.get_weights())

        if env_func == None:
            self.env = gym.make('CartPole-v0').unwrapped
        else:
            print("Building environment")
            self.env = env_func(idx)
        print("Environment built!")

        self.gamma = gamma
        self.entropy_discount = entropy_discount
        self.value_discount = value_discount
        self.boredom_thresh = boredom_thresh
        self.update_freq = update_freq

        self.save_path = save_path
        self.visual_path = visual_path
        self.summary_writer = summary_writer
        self.log_period = log_period
        self.visual_period = visual_period
        self.checkpoint_period = checkpoint_period

        self.result_queue = result_queue

        self.ep_loss = 0.0
        self.worker_idx = idx
        self.max_episodes = max_episodes

    def run(self):
        total_step = 1
        mem = Memory()

        while Worker.global_episode < self.max_episodes:
            current_state, obs = self.env.reset()
            rolling_average_state = current_state
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0
            current_episode = Worker.global_episode
            save_visual = (self.visual_path != None and current_episode % self.visual_period == 0)

            action = 0
            time_count = 0
            done = False
            while not done:
                _deviation = tf.reduce_sum(tf.math.squared_difference(rolling_average_state, current_state))
                if time_count > 10 and _deviation < self.boredom_thresh:
                    possible_actions = np.delete(np.array([0, 1, 2]), action)
                    action = np.random.choice(possible_actions)
                else:
                    action, _ = self.local_model.get_action_value(
                                        tf.convert_to_tensor(current_state[None, :],
                                        dtype=tf.float32))
                (new_state, reward, done, _), new_obs = self.env.step(action)
                ep_reward += reward
                mem.store(current_state, action, reward)
                if save_visual:
                    mem.obs.append(obs)
                # _deviation = tf.reduce_sum(tf.math.squared_difference(rolling_average_state, new_state))
                
                if time_step == self.update_freq or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape

                    # Update model
                    with tf.GradientTape() as tape:
                        total_loss  = self.compute_loss(mem, self.gamma, save_visual)
                    self.ep_loss += tf.reduce_sum(total_loss)

                    # Calculate and apply policy gradients
                    total_grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    self.opt.apply_gradients(zip(total_grads,
                                           self.global_model.trainable_weights))

                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())

                    if done:
                        self.log_episode(save_visual, current_episode, ep_steps, ep_reward, mem, total_loss)
                        Worker.global_episode += 1
                    mem.clear()
                    time_count = 0
                else:
                    ep_steps += 1
                    time_count += 1
                    current_state = new_state
                    rolling_average_state = rolling_average_state * 0.8 + new_state * 0.2
                    obs = new_obs
                    total_step += 1
        self.result_queue.put(None)

    def compute_loss(self,
                     memory,
                     gamma,
                     save_visual):

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
        if save_visual:
            memory.probs.extend(np.squeeze(tf.nn.softmax(logits).numpy()).tolist())
            memory.values.extend(np.squeeze(values.numpy()).tolist())

        # Calculate our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32) - values

        # Calculate our policy loss
        entropy_loss = keras.losses.categorical_crossentropy(tf.stop_gradient(logits),
                                                             tf.nn.softmax(logits),
                                                             from_logits=True)
        weighted_sparse_crossentropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = weighted_sparse_crossentropy(np.array(memory.actions)[:, None],
                                                   logits,
                                                   sample_weight=tf.stop_gradient(advantage))
        policy_loss = policy_loss - (self.entropy_discount * entropy_loss)

        # Calculate our value loss
        value_loss = keras.losses.mean_squared_error(np.array(discounted_rewards)[:, None], values)

        return policy_loss + (value_loss * self.value_discount)

    def log_episode(self, save_visual, current_episode, ep_steps, ep_reward, mem, total_loss):
        # Save the memory of our episode
        if save_visual:
            pickle_path = os.path.join(self.visual_path, "memory_{}_{}".format(current_episode, self.worker_idx))
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
            pickle_file = open(pickle_path, 'wb+')
            pickle.dump(mem, pickle_file)
            pickle_file.close()
            print("Memory saved to {}".format(pickle_path))
        # Metrics logging and saving
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

        if current_episode % self.log_period == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar("Episode Reward", ep_reward, current_episode)
                tf.summary.scalar("Episode Loss", tf.reduce_sum(total_loss), current_episode)
                tf.summary.scalar("Moving Global Average", Worker.global_moving_average_reward, current_episode)
        if current_episode % self.checkpoint_period == 0:
            _save_path = os.path.join(self.save_path, "worker_{}_model_{}.h5".format(self.worker_idx, current_episode))
            self.local_model.save_weights(_save_path)
            print("Checkpoint saved to {}".format(_save_path))

