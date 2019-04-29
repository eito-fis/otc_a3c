
import os
import gc
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
    print("Episode: {} | Moving Average Reward: {} | Episode Reward: {} | Loss: {} | Steps: {} | Worker: {}".format(episode, global_ep_reward, episode_reward, total_loss, num_steps, worker_idx))
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
        self.key = []
        self.time = []

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
        self.key = []
        self.time = []

class ProbabilityDistribution(keras.Model):
    def call(self, logits):
        # Sample a random action from logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class ActorCriticModel(keras.Model):
    def __init__(self,
                 num_actions=None,
                 state_size=None,
                 stack_size=None,
                 sparse_stack_size=0,
                 action_stack_size=0,
                 conv_size=None,
                 actor_fc=None,
                 critic_fc=None,
                 opt=None):
        super().__init__()
        state_size = state_size[:-1] + [state_size[-1] * (stack_size + sparse_stack_size) + (num_actions * action_stack_size)]

        self.actor_fc = [keras.layers.Dense(neurons, activation="relu") for neurons in actor_fc]
        self.critic_fc = [keras.layers.Dense(neurons, activation="relu") for neurons in critic_fc]

        self.actor_logits = keras.layers.Dense(num_actions, name='policy_logits')
        self.value = keras.layers.Dense(1, name='value', activation='relu')

        self.actor_model = tf.keras.Sequential(self.actor_fc + [self.actor_logits])
        self.actor_model.build([None] + state_size)
        self.critic_model = tf.keras.Sequential(self.critic_fc + [self.value])
        self.critic_model.build([None] + state_size)

        self.dist = ProbabilityDistribution()
        self.get_action_value(np.random.random((1,) + tuple(state_size)))

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
                 num_episodes=1000,
                 env_func=None,
                 num_actions=3,
                 state_size=[4],
                 stack_size=4,
                 sparse_stack_size=0,
                 sparse_update=0,
                 action_stack_size=4,
                 conv_size=None,
                 learning_rate=0.00042,
                 gamma=0.99,
                 entropy_discount=0.05,
                 value_discount=0.1,
                 boredom_thresh=10,
                 update_freq=650,
                 actor_fc=None,
                 critic_fc=None,
                 summary_writer=None,
                 log_period=10,
                 checkpoint_period=10,
                 visual_period=None,
                 memory_path="/tmp/a3c/visuals",
                 save_path="/tmp/a3c",
                 load_path=None):

        self.memory_path = memory_path
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
        self.stack_size = stack_size
        self.sparse_stack_size = sparse_stack_size
        self.action_stack_size = action_stack_size
        self.update_freq = update_freq
        self.sparse_update = sparse_update
        self.conv_size = conv_size
        self.actor_fc = actor_fc
        self.critic_fc = critic_fc

        self.gamma = gamma
        self.entropy_discount = entropy_discount
        self.value_discount = value_discount
        self.boredom_thresh = boredom_thresh
        self.update_freq = update_freq

        self.opt = tf.keras.optimizers.Adam(learning_rate)
        self.env_func = env_func

        self.global_model = ActorCriticModel(num_actions=self.num_actions,
                                             state_size=self.state_size,
                                             stack_size=self.stack_size,
                                             sparse_stack_size=self.sparse_stack_size,
                                             action_stack_size=self.action_stack_size,
                                             conv_size=self.conv_size,
                                             actor_fc=self.actor_fc,
                                             critic_fc=self.critic_fc,
                                             opt=self.opt)
        if load_path != None:
            self.global_model.load_weights(load_path)
            print("Loaded model from {}".format(load_path))

    def distributed_train(self):
        res_queue = Queue()

        workers = [Worker(idx=i,
                   num_actions=self.num_actions,
                   state_size=self.state_size,
                   stack_size=self.stack_size,
                   sparse_stack_size=self.sparse_stack_size,
                   sparse_update=self.sparse_update,
                   action_stack_size=self.action_stack_size,
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
                   memory_path=self.memory_path,
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
        return moving_average_rewards

    def human_train(self, data_path, train_steps, batch_size):
        # Load human input from pickle file
        data_file = open(data_path, 'rb')
        memory_list = pickle.load(data_file)
        data_file.close()
        print("Loaded files")

        counts = np.zeros(self.num_actions)
        for memory in memory_list:
            for action in memory.actions:
                if action < self.num_actions: counts[action] += 1
        counts = [(sum(counts) - c) / sum(counts) for c in counts]

        def gen():
            while True:
                for memory in memory_list:
                    prev_states = [np.random.random(tuple(self.state_size)) for _ in range(self.stack_size)]
                    sparse_states = [np.random.random(tuple(self.state_size)) for _ in range(self.sparse_stack_size)]
                    prev_actions = [np.zeros(self.num_actions) for _ in range(self.action_stack_size)]
                    for index, (action, state) in enumerate(zip(memory.actions, memory.states)):
                        if action >= self.num_actions: continue
                        prev_states = prev_states[1:] + [state]
                        if self.action_stack_size > 0:
                            one_hot_action = np.zeros(self.num_actions)
                            one_hot_action[action] = 1
                            prev_actions = prev_actions[1:] + [one_hot_action]
                        if self.sparse_stack_size > 0 and index % self.sparse_update == 0:
                            sparse_states = sparse_states[1:] + [state]
                        stacked_state = np.concatenate(prev_states + sparse_states + prev_actions)
                        yield (action, stacked_state)

        dataset = tf.data.Dataset.from_generator(generator=gen,
                                                 output_types=(tf.float32, tf.float32))
        dataset = dataset.shuffle(10000).batch(batch_size)
        generator = iter(dataset)

        print("Starting steps...")
        for train_step in range(train_steps):
            actions, states  = next(generator)
            weights = [counts[action] for action in actions]
            with tf.GradientTape() as tape:
                total_loss = self.compute_loss(actions,
                                               states,
                                               weights,
                                               self.gamma)

            # Calculate and apply policy gradients
            total_grads = tape.gradient(total_loss, self.global_model.actor_model.trainable_weights)
            self.opt.apply_gradients(zip(total_grads, self.global_model.actor_model.trainable_weights))
            print("Step: {} | Loss: {}".format(train_step, total_loss))
            if train_step % 100 == 0:
                target_actions, states  = next(generator)
                predict_actions = self.global_model.actor_model(states)
                predict_actions = [np.argmax(distribution) for distribution in predict_actions]
                target_actions = list(target_actions.numpy())
                correct = [1 if t == p else 0 for t, p in zip(target_actions, predict_actions)]
                print("Accuracy: {}".format(sum(correct) / len(correct)))


        critic_batch_size = 100
        critic_steps = 1000
        self.initialize_critic_model(critic_batch_size, critic_steps)

        _save_path = os.path.join(self.save_path, "human_trained_model.h5")
        os.makedirs(os.path.dirname(_save_path), exist_ok=True)
        self.global_model.save_weights(_save_path)
        print("Checkpoint saved to {}".format(_save_path))

        gc.collect()

    def compute_loss(self,
                     actions,
                     states,
                     weights,
                     gamma):

        # Get logits and values
        logits, values = self.global_model(states)

        weighted_sparse_crossentropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = weighted_sparse_crossentropy(actions[:, None], logits, sample_weight=weights)

        return policy_loss

    def initialize_critic_model(self, batch_size, critic_steps):
        random_states = np.random.random((batch_size,) + tuple(self.state_size[:-1] +
                                                            [self.state_size[-1] * self.stack_size + (self.num_actions * self.action_stack_size)]))
        zero_rewards = np.zeros(batch_size)
        for critic_step in range(critic_steps):
            with tf.GradientTape() as tape:
                values = self.global_model.critic_model(random_states)
                value_loss = keras.losses.mean_squared_error(zero_rewards[:, None], values)
            value_grads = tape.gradient(value_loss, self.global_model.critic_model.trainable_weights)

            self.opt.apply_gradients(zip(value_grads, self.global_model.critic_model.trainable_weights))

    def play(self, env):
        state, observation = env.reset()
        done = False
        step_counter = 0
        reward_sum = 0
        rolling_average_state = np.zeros(state.shape) + (0.2 * state)
        memory = Memory()
        floor = 0

        try:
            prev_states = [np.random.random(state.shape) for _ in range(self.stack_size)]
            sparse_states = [np.random.random(state.shape) for _ in range(self.sparse_stack_size)]
            prev_actions = [np.zeros(self.num_actions) for _ in range(self.action_stack_size)]
            while not done:
                prev_states = prev_states[1:] + [state]
                if self.action_stack_size > 0 and step_counter > 0:
                    one_hot_action = np.zeros(self.num_actions)
                    one_hot_action[action] = 1
                    prev_actions = prev_actions[1:] + [one_hot_action]
                # print("prev_actions: {}".format(prev_actions))
                # input()
                if self.sparse_stack_size > 0 and step_counter % self.sparse_update == 0:
                    sparse_states = sparse_states[1:] + [state]
                _deviation = tf.reduce_sum(tf.math.squared_difference(rolling_average_state, state))
                if step_counter > 10 and _deviation < self.boredom_thresh:
                    possible_actions = np.delete(np.array(range(self.num_actions)), action)
                    action = np.random.choice(possible_actions)
                    # distribution = np.zeros(self.num_actions)
                else:
                    stacked_state = np.concatenate(prev_states + sparse_states + prev_actions)
                    logits = self.global_model.actor_model(stacked_state[None, :])
                    distribution = tf.squeeze(tf.nn.softmax(logits)).numpy()
                    action = np.argmax(logits)
                    value = self.global_model.critic_model(stacked_state[None, :])
                (new_state, reward, done, _), new_observation = env.step(action)
                memory.store(state, action, reward)
                if reward >= 1: floor += 1
                if self.memory_path:
                    memory.obs.append(observation)
                    memory.probs.append(distribution)
                    memory.values.append(value)
                state = new_state
                observation = new_observation
                reward_sum += reward
                rolling_average_state = rolling_average_state * 0.8 + new_state * 0.2
                print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            if self.memory_path:
                _mem_path = os.path.join(self.memory_path, "evaluation_memory")
                os.makedirs(os.path.dirname(_mem_path), exist_ok=True)
                output_file = open(_mem_path, 'wb+')
                pickle.dump(memory, output_file)
        return floor


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
                 stack_size=None,
                 sparse_stack_size=None,
                 sparse_update=None,
                 action_stack_size=None,
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
                 memory_path='/tmp/a3c/visuals/',
                 save_path='/tmp/a3c/workers'):
        super(Worker, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        self.stack_size = stack_size
        self.sparse_stack_size = sparse_stack_size
        self.action_stack_size = action_stack_size
        self.sparse_update = sparse_update
        self.conv_size = conv_size

        self.opt = opt
        self.global_model = global_model
        self.local_model = ActorCriticModel(num_actions=self.num_actions,
                                            state_size=self.state_size,
                                            stack_size=stack_size,
                                            sparse_stack_size=self.sparse_stack_size,
                                            action_stack_size=self.action_stack_size,
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
        self.memory_path = memory_path
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
            state, obs = self.env.reset()
            rolling_average_state = state
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0
            current_episode = Worker.global_episode
            save_visual = (self.memory_path != None and current_episode % self.visual_period == 0)

            action = 0
            time_count = 0
            done = False

            prev_states = [np.random.random(state.shape) for _ in range(self.stack_size)]
            sparse_states = [np.random.random(state.shape) for _ in range(self.sparse_stack_size)]
            prev_actions = [np.zeros(self.num_actions) for _ in range(self.action_stack_size)]
            while not done:
                prev_states = prev_states[1:] + [state]
                if self.action_stack_size > 0 and time_count > 0:
                    one_hot_action = np.zeros(self.num_actions)
                    one_hot_action[action] = 1
                    prev_actions = prev_actions[1:] + [one_hot_action]
                if self.sparse_stack_size > 0 and time_count % self.sparse_update == 0:
                    sparse_states = sparse_states[1:] + [state]
                _deviation = tf.reduce_sum(tf.math.squared_difference(rolling_average_state, state))
                if time_count > 10 and _deviation < self.boredom_thresh:
                    possible_actions = np.delete(np.array(range(self.num_actions)), action)
                    action = np.random.choice(possible_actions)
                    # distribution = np.zeros(self.num_actions)
                else:
                    stacked_state = np.concatenate(prev_states + sparse_states + prev_actions)
                    action, _ = self.local_model.get_action_value(stacked_state[None, :])

                (new_state, reward, done, _), new_obs = self.env.step(action)
                ep_reward += reward
                mem.store(state, action, reward)
                if save_visual:
                    mem.obs.append(obs)

                if time_count == self.update_freq or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape

                    # Update model
                    with tf.GradientTape() as tape:
                        total_loss  = self.compute_loss(mem, done, self.gamma, save_visual, stacked_state)
                    self.ep_loss += total_loss

                    # Calculate and apply policy gradients
                    total_grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    self.opt.apply_gradients(zip(total_grads,
                                           self.global_model.trainable_weights))

                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())

                    if done:
                        self.log_episode(save_visual, current_episode, ep_steps, ep_reward, mem, total_loss)
                        Worker.global_episode += 1
                    time_count = 0

                    mem.clear()
                else:
                    ep_steps += 1
                    time_count += 1
                    state = new_state
                    rolling_average_state = rolling_average_state * 0.8 + new_state * 0.2
                    obs = new_obs
                    total_step += 1
        self.result_queue.put(None)

    def compute_loss(self,
                     memory,
                     done,
                     gamma,
                     save_visual,
                     stacked_state):
        # If not done, estimate the future discount reward of being in the final state
        # using the critic model
        if done:
            reward_sum = 0.
        else:
            reward_sum = self.local_model.critic_model(stacked_state[None, :])
            reward_sum = np.squeeze(reward_sum.numpy())

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        stacked_states = []
        prev_states = [np.random.random(tuple(self.state_size)) for _ in range(self.stack_size)]
        sparse_states = [np.random.random(tuple(self.state_size)) for _ in range(self.sparse_stack_size)]
        prev_actions = [np.zeros(self.num_actions) for _ in range(self.action_stack_size)]
        for index, (state, action) in enumerate(zip(memory.states, memory.actions)):
            prev_states = prev_states[1:] + [state]
            if self.action_stack_size > 0:
                one_hot_action = np.zeros(self.num_actions)
                one_hot_action[action] = 1
                prev_actions = prev_actions[1:] + [one_hot_action]
            if self.sparse_stack_size > 0 and index % self.sparse_update == 0:
                sparse_states = sparse_states[1:] + [state]
            stacked_states.append(np.concatenate(prev_states + sparse_states + prev_actions))

        # Get logits and values
        logits, values = self.local_model(np.array(stacked_states))

        # Calculate our advantages
        advantage = np.array(discounted_rewards)[:, None] - values

        # Calculate our policy loss
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        entropy_loss = cce(tf.stop_gradient(tf.nn.softmax(logits)), logits)
        wce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = wce(memory.actions,
                          logits,
                          sample_weight=tf.stop_gradient(advantage))
        policy_loss = policy_loss - (self.entropy_discount * entropy_loss)

        # Calculate our value loss
        mse = tf.keras.losses.MeanSquaredError()
        value_loss = mse(tf.stop_gradient(np.array(discounted_rewards)[:, None]), values)

        if save_visual:
            memory.probs.extend(np.squeeze(tf.nn.softmax(logits).numpy()).tolist())
            memory.values.extend(np.squeeze(values.numpy()).tolist())

        return policy_loss + (value_loss * self.value_discount)

    def log_episode(self, save_visual, current_episode, ep_steps, ep_reward, mem, total_loss):
        # Save the memory of our episode
        if save_visual:
            pickle_path = os.path.join(self.memory_path, "memory_{}_{}".format(current_episode, self.worker_idx))
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

        # Save model and logs for tensorboard
        if current_episode % self.log_period == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar("Episode Reward", ep_reward, current_episode)
                tf.summary.scalar("Episode Loss", tf.reduce_sum(total_loss), current_episode)
                tf.summary.scalar("Moving Global Average", Worker.global_moving_average_reward, current_episode)
        if current_episode % self.checkpoint_period == 0:
            _save_path = os.path.join(self.save_path, "worker_{}_model_{}.h5".format(self.worker_idx, current_episode))
            self.local_model.save_weights(_save_path)
            print("Checkpoint saved to {}".format(_save_path))
