
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

class ActorModel(keras.Model):
    def __init__(self,
                 num_actions=None,
                 state_size=None,
                 stack_size=None,
                 sparse_stack_size=None,
                 conv_size=None,
                 actor_fc=None,
                 critic_fc=None,
                 curiosity_fc=None):
        super().__init__()
        state_size = state_size[:-1] + [state_size[-1] * (stack_size + sparse_stack_size)]
        
        # Build fully connected layers for our models
        self.actor_fc = [keras.layers.Dense(neurons, activation="relu") for neurons in actor_fc]

        # Build endpoints for our models
        self.actor_logits = keras.layers.Dense(num_actions, name='policy_logits')
        # Build A2C models
        self.actor_model = tf.keras.Sequential(self.actor_fc + [self.actor_logits])
        self.actor_model.build([None] + state_size)

        # Build sample chooser TODO: Replace with tf.distribution
        self.dist = ProbabilityDistribution()

        # Run the entire pipeline to build the graph before async workers start
        self.actor_model(np.random.random((1,) + tuple(state_size)))
        self.dist(np.random.random((1, num_actions)))

    def call(self, inputs):
        # Call our models on the input and return
        actor_logits = self.actor_model(inputs)

        return actor_logits

    def get_action_value(self, obs):
        logits = self.actor_model(obs)

        action = self.dist.predict(logits)
        action = action[0]

        return action

class MasterAgent():
    def __init__(self,
                 train_steps=1000,
                 env_func=None,
                 num_actions=2,
                 state_size=[4],
                 stack_size=6,
                 sparse_stack_size=4,
                 boredom_thresh=10,
                 actor_fc=None,
                 memory_path="/tmp/a3c/visuals",
                 save_path="/tmp/a3c",
                 load_path=None):

        self.memory_path = memory_path
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.train_steps = train_steps
        self.num_actions = num_actions
        self.state_size = state_size
        self.stack_size = stack_size
        self.sparse_stack_size = sparse_stack_size
        self.actor_fc = actor_fc

        self.boredom_thresh = boredom_thresh

        self.env_func = env_func
        self.global_model = ActorModel(num_actions=self.num_actions,
                                       state_size=self.state_size,
                                       stack_size=self.stack_size,
                                       sparse_stack_size=sparse_stack_size,
                                       actor_fc=self.actor_fc)
        if load_path != None:
            self.global_model.load_weights(load_path, by_name=True)

    def distributed_eval(self):
        res_queue = Queue()

        workers = [Worker(idx=i,
                   num_actions=self.num_actions,
                   state_size=self.state_size,
                   stack_size=self.stack_size,
                   sparse_stack_size=self.sparse_stack_size,
                   actor_fc=self.actor_fc,
                   boredom_thresh=self.boredom_thresh,
                   global_model=self.global_model,
                   result_queue=res_queue,
                   env_func=self.env_func,
                   max_episodes=self.train_steps,
                   memory_path=self.memory_path,
                   save_path=self.save_path) for i in range(multiprocessing.cpu_count())]
                   #save_path=self.save_path) for i in range(1)]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

        all_floors = []
        all_rewards = []
        while True:
            data = res_queue.get()
            if data is not None:
                floor = data[0]
                total_reward = data[1]
                all_floors.append(floor)
                all_rewards.append(total_reward)
                floors_hist = np.histogram(all_floors, 10, (0,10))
                print("Floor histogram: {}".format(floors_hist[0]))
                print("Average reward: {}".format(sum(all_rewards) / len(all_rewards)))
            else:
                break
        [w.join() for w in workers]
        print("Done!")
        floors_hist = np.histogram(all_floors, 10, (0,10))
        print("Final Floor histogram: {}".format(floors_hist[0]))
        print("Final Average reward: {}".format(sum(all_rewards) / len(all_rewards)))
        return all_floors, all_rewards

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
                 actor_fc=None,
                 boredom_thresh=None,
                 global_model=None,
                 result_queue=None,
                 env_func=None,
                 max_episodes=None,
                 memory_path='/tmp/a3c/visuals',
                 save_path='/tmp/a3c/workers'):
        super(Worker, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        self.stack_size = stack_size
        self.sparse_stack_size = sparse_stack_size

        self.global_model = global_model
        self.local_model = ActorModel(num_actions=self.num_actions,
                                            state_size=self.state_size,
                                            stack_size=self.stack_size,
                                            sparse_stack_size=self.sparse_stack_size,
                                            actor_fc=actor_fc)
        self.local_model.set_weights(self.global_model.get_weights())

        print("Building environment")
        self.env = env_func(idx)
        print("Environment built!")

        self.boredom_thresh = boredom_thresh

        self.save_path = save_path
        self.memory_path = memory_path

        self.result_queue = result_queue

        self.worker_idx = idx
        self.max_episodes = max_episodes

    def run(self):
        mem = Memory()

        while Worker.global_episode < self.max_episodes:
            current_state, obs = self.env.reset()
            rolling_average_state = current_state
            mem.clear()
            current_episode = Worker.global_episode

            time_count = 0
            floor = 0
            total_reward = 0
            done = False
            prev_states = [np.random.random(state.shape) for _ in range(self.stack_size)]
            sparse_states = [np.random.random(state.shape) for _ in range(self.sparse_stack_size)]
            boredom_actions = []
            while not done:
                prev_states = prev_states[1:] + [state]
                if self.sparse_stack_size > 0 and time_count % self.sparse_update == 0:
                    sparse_states = sparse_states[1:] + [state]
                _deviation = tf.reduce_sum(tf.math.squared_difference(rolling_average_state, state))
                if time_count > 10 and _deviation < self.boredom_thresh:
                    possible_actions = np.delete(np.array([0, 1, 2]), action)
                    action = np.random.choice(possible_actions)
                    boredom_actions = [action] + [1] * 20
                if len(boredom_actions):
                    action = boredom_actions.pop()
                else:
                    stacked_state = np.concatenate(prev_states + sparse_states)
                    action = self.local_model.get_action_value(stacked_state[None, :])

                (new_state, reward, done, _), new_obs = self.env.step(action)

                if reward == 1: floor += 1
                total_reward += reward
                mem.store(current_state, action, reward)

                time_count += 1
                current_state = new_state
                rolling_average_state = rolling_average_state * 0.8 + new_state * 0.2
                obs = new_obs
            print("Episode {} | Floor {} | Reward {}".format(current_episode, floor, total_reward))
            self.result_queue.put((floor, total_reward))
        self.result_queue.put(None)
