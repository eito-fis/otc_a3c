
import os
import pickle
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
        self.values = []
        self.novelty = []

class ProbabilityDistribution(keras.Model):
    def call(self, logits):
        # Sample a random action from logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class ActorCriticModel(keras.Model):
    def __init__(self,
                 num_actions=None,
                 state_size=None,
                 stack_size=None,
                 conv_size=None,
                 actor_fc=None,
                 critic_fc=None,
                 curiosity_fc=None):
        super().__init__()
        state_size = state_size[:-1] + [state_size[-1] * stack_size]
        
        # Build fully connected layers for our models
        self.actor_fc = [keras.layers.Dense(neurons, activation="relu") for neurons in actor_fc]
        self.critic_fc = [keras.layers.Dense(neurons, activation="relu") for neurons in critic_fc]
        self.curiosity_fc = [keras.layers.Dense(neurons, activation="relu") for neurons in curiosity_fc]
        self.target_fc = [keras.layers.Dense(neurons, activation="relu") for neurons in curiosity_fc]

        # Build endpoints for our models
        self.actor_logits = keras.layers.Dense(num_actions, name='policy_logits')
        self.value = keras.layers.Dense(1, name='value', activation='relu')

        # Build A2C models
        self.actor_model = tf.keras.Sequential(self.actor_fc + [self.actor_logits])
        self.actor_model.build([None] + state_size)
        self.critic_model = tf.keras.Sequential(self.critic_fc + [self.value])
        self.critic_model.build([None] + state_size)

        # Build Cuiriosity models
        self.curiosity_model = tf.keras.Sequential(self.curiosity_fc)
        self.curiosity_model.build([None] + state_size)
        self.target_model = tf.keras.Sequential(self.target_fc)
        self.target_model.build([None] + state_size)

        # Build sample chooser TODO: Replace with tf.distribution
        self.dist = ProbabilityDistribution()

        # Run the entire pipeline to build the graph before async workers start
        self.get_action_value(np.random.random((1,) + tuple(state_size)))
        self.curiosity_model(np.random.random((1,) + tuple(state_size)))
        self.target_model(np.random.random((1,) + tuple(state_size)))

    def call(self, inputs):
        # Call our models on the input and return
        actor_logits = self.actor_model(inputs)
        value = self.critic_model(inputs)
        prediction = self.curiosity_model(inputs)
        target = self.target_model(inputs)

        return actor_logits, value, prediction, target

    def get_action_value(self, obs):
        logits = self.actor_model(obs)
        value = self.critic_model(obs)

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
                 stack_size=4,
                 conv_size=None,
                 learning_rate=0.00001,
                 gamma=0.99,
                 entropy_discount=0.01,
                 value_discount=0.25,
                 novelty_discount=0.01,
                 intrinsic_reward_discount=0.1,
                 boredom_thresh=0,
                 update_freq=650,
                 actor_fc=None,
                 critic_fc=None,
                 curiosity_fc=None,
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
        self.stack_size = stack_size
        self.conv_size = conv_size
        self.actor_fc = actor_fc
        self.critic_fc = critic_fc
        self.curiosity_fc = curiosity_fc

        self.gamma = gamma
        self.entropy_discount = entropy_discount
        self.value_discount = value_discount
        self.novelty_discount = novelty_discount
        self.intrinsic_reward_discount = intrinsic_reward_discount
        self.boredom_thresh = boredom_thresh
        self.update_freq = update_freq

        self.opt = keras.optimizers.Adam(learning_rate)
        self.env_func = env_func
        self.global_model = ActorCriticModel(num_actions=self.num_actions,
                                             state_size=self.state_size,
                                             stack_size=self.stack_size,
                                             conv_size=self.conv_size,
                                             actor_fc=self.actor_fc,
                                             critic_fc=self.critic_fc,
                                             curiosity_fc=self.curiosity_fc)
        if load_path != None:
            self.global_model.load_weights(load_path)

    def distributed_train(self):
        res_queue = Queue()

        workers = [Worker(idx=i,
                   num_actions=self.num_actions,
                   state_size=self.state_size,
                   stack_size=self.stack_size,
                   conv_size=self.conv_size,
                   actor_fc=self.actor_fc,
                   critic_fc=self.critic_fc,
                   curiosity_fc=self.curiosity_fc,
                   gamma=self.gamma,
                   entropy_discount=self.entropy_discount,
                   value_discount=self.value_discount,
                   novelty_discount=self.novelty_discount,
                   intrinsic_reward_discount=self.intrinsic_reward_discount,
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

    def human_train(self, data_path, train_steps, batch_size):
        # Load memories from pickle file
        data_file = open(data_path, 'rb')
        memory_list = pickle.load(data_file)
        data_file.close()
        print("Loaded files")

        counts = [frame for memory in memory_list for frame in memory.actions]
        counts = [(len(counts) - c) / len(counts) for c in list(Counter(counts).values())]
        print("Counts: {}".format(Counter(all_actions)))

        def gen():
            while True:
                for memory in memory_list:
                    for index, (action, state) in enumerate(zip(memory.actions, memory.states)):
                        if len(actions) == batch_size:
                            weights = [counts[action] for action in actions]
                            yield actions, states, weights
                            actions = []
                            states = [] 
                        actions.append(action)
                        stacked_state = [np.zeros_like(state) if index - i < 0 else memory.states[index - i].numpy()
                                      for i in reversed(range(self.stack_size))]
                        stacked_state = np.concatenate(stacked_state)
                        states.append(stacked_state)

        print("Starting steps...")
        generator = gen()
        for train_step in range(train_steps):
            actions, states, weights = next(generator)
            with tf.GradientTape() as tape:
                total_loss = self.compute_loss(actions,
                                               states,
                                               weights,
                                               self.gamma)
        
            # Calculate and apply policy gradients
            total_grads = tape.gradient(total_loss, self.global_model.actor_model.trainable_weights)
            self.opt.apply_gradients(zip(total_grads,
                                             self.global_model.actor_model.trainable_weights))
            print("Step: {} | Loss: {}".format(train_step, total_loss))
            
    def compute_loss(self,
                     actions,
                     states,
                     weights,
                     gamma):
        # Get logits and values
        logits, values, predict, train = self.global_model(states)

        weighted_sparse_crossentropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = weighted_sparse_crossentropy(np.array(actions)[:, None], logits, sample_weight=weights)

        return policy_loss

    def play(self):
        env = self.env_func(idx=0)
        state, observation = env.reset()
        done = False
        step_counter = 0
        reward_sum = 0
        rolling_average_state = np.zeros(state.shape) + (0.2 * state)
        if self.memory_path:
            memory = Memory()

        try:
            while not done:
                _deviation = tf.reduce_sum(tf.math.squared_difference(rolling_average_state, state))
                if step_counter > 10 and _deviation < self.boredom_thresh:
                    possible_actions = np.delete(np.array([0, 1, 2]), action)
                    action = np.random.choice(possible_actions)
                    distribution = np.zeros(self.num_actions)
                    value = 100
                else:
                    stacked_state = [np.zeros_like(state) if step_counter - i < 0
                                                          else memory.states[step_counter - i].numpy()
                                                          for i in reversed(range(self.stack_size))]
                    logits = self.global_model.actor_model(stacked_state[None, :])
                    distribution = tf.squeeze(tf.nn.softmax(logits)).numpy()
                    action = np.argmax(logits)
                    value = self.global_model.critic_model(stacked_state[None, :])
                (new_state, reward, done, _), new_observation = env.step(action)
                memory.store(state, action, reward)
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
                 stack_size=None,
                 conv_size=None,
                 actor_fc=None,
                 critic_fc=None,
                 curiosity_fc=None,
                 gamma=None,
                 entropy_discount=None,
                 value_discount=None,
                 novelty_discount=None,
                 intrinsic_reward_discount=None,
                 boredom_thresh=None,
                 update_freq=None,
                 global_model=None,
                 opt=None,
                 result_queue=None,
                 env_func=None,
                 max_episodes=None,
                 summary_writer=None,
                 log_period=10,
                 checkpoint_period=10,
                 visual_period=10,
                 visual_path='/tmp/a3c/visuals',
                 save_path='/tmp/a3c/workers'):
        super(Worker, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        self.stack_size = stack_size
        self.conv_size = conv_size

        self.opt = opt
        self.global_model = global_model
        self.local_model = ActorCriticModel(num_actions=self.num_actions,
                                            state_size=self.state_size,
                                            stack_size=self.stack_size,
                                            conv_size=self.conv_size,
                                            actor_fc=actor_fc,
                                            critic_fc=critic_fc,
                                            curiosity_fc=curiosity_fc)
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
        self.novelty_discount = novelty_discount
        self.intrinsic_reward_discount = intrinsic_reward_discount
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
            save_visual = self.visual_path != None and current_episode % self.visual_period == 0

            time_count = 0
            done = False
            while not done:
                _deviation = tf.reduce_sum(tf.math.squared_difference(rolling_average_state, current_state))
                if time_count > 10 and _deviation < self.boredom_thresh:
                    possible_actions = np.delete(np.array([0, 1, 2]), action)
                    action = np.random.choice(possible_actions)
                else:
                    stacked_state = [np.zeros_like(current_state) if time_count - i < 0
                                                                  else mem.states[time_count - i].numpy()
                                                                  for i in reversed(range(1, self.stack_size))]
                    stacked_state.append(current_state)
                    stacked_state = np.concatenate(stacked_state)
                    action, _ = self.local_model.get_action_value(stacked_state[None, :])

                (new_state, reward, done, _), new_obs = self.env.step(action)
                ep_reward += reward
                mem.store(current_state, action, reward)
                if save_visual:
                    mem.obs.append(obs)
                               
                if time_count == self.update_freq or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape

                    # Update model
                    with tf.GradientTape() as tape:
                        total_loss  = self.compute_loss(mem, done, self.gamma, save_visual)
                    self.ep_loss += tf.reduce_sum(total_loss)

                    # Calculate and apply policy gradients
                    total_grads = tape.gradient(total_loss,
                                           self.local_model.actor_model.trainable_weights +
                                           self.local_model.critic_model.trainable_weights +
                                           self.local_model.curiosity_model.trainable_weights)
                    self.opt.apply_gradients(zip(total_grads,
                                           self.global_model.actor_model.trainable_weights +
                                           self.global_model.critic_model.trainable_weights +
                                           self.global_model.curiosity_model.trainable_weights))

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
                     done,
                     gamma,
                     save_visual):
        # If not done, estimate the future discount reward of being in the final state
        # using the critic model
        if done:
            reward_sum = 0.
        else:
            reward_sum = self.local_model.critic_model(np.concatenate(memory.states[-self.stack_size:])[None, :])
            reward_sum = np.squeeze(reward_sum.numpy())
 
        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        stacked_states = []
        for index, state in enumerate(memory.states):
            stacked_state = [np.zeros_like(state) if index - i < 0 else memory.states[index - i].numpy()
                                      for i in reversed(range(self.stack_size))]
            stacked_states.append(np.concatenate(stacked_state))

        # Get logits, values, predictions and targets
        logits, values, predictions, targets = self.local_model(stacked_states)

        # Calculate our novelty
        novelty = keras.losses.mean_squared_error(predictions, tf.stop_gradient(targets))
        novelty = tf.math.divide_no_nan(novelty - tf.math.reduce_mean(novelty), tf.math.reduce_std(novelty))
        print(novelty)

        # Calculate our advantages
        advantage = np.array(discounted_rewards)[:, None] - values

        # Calculate our total reward
        total_reward = advantage + (novelty[:, None] * self.intrinsic_reward_discount)

        # Calculate our entropy loss
        entropy_loss = keras.losses.categorical_crossentropy(tf.stop_gradient(tf.nn.softmax(logits)),
                                                             logits,
                                                             from_logits=True)

        # Calculate our policy loss
        weighted_sparse_crossentropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = weighted_sparse_crossentropy(np.array(memory.actions)[:, None],
                                                   logits,
                                                   sample_weight=tf.stop_gradient(total_reward))
        policy_loss = policy_loss - (entropy_loss * self.entropy_discount)

        # Calculate our value loss
        value_loss = keras.losses.mean_squared_error(np.array(discounted_rewards)[:, None], values)

        if save_visual:
            memory.probs.extend(np.squeeze(tf.nn.softmax(logits).numpy()).tolist())
            memory.values.extend(np.squeeze(values.numpy()).tolist())
            memory.novelty.extend(np.squeeze(novelty.numpy()).tolist())

        return policy_loss + (value_loss * self.value_discount) + (novelty * self.novelty_discount)

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

