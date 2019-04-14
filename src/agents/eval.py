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
from functools import reduce

from src.agents.a3c import ActorCriticModel as A3CModel
from src.agents.a3c import Memory, ProbabilityDistribution
from src.agents.curiosity import ActorCriticModel as CuriosityModel

def average_level(histogram):
        inverse_histogram = list(map(lambda x: 1 - x, histogram))
        max_level = len(histogram)
        avg_level = 0
        for i, (ratio, inv_ratio) in enumerate(zip(histogram, inverse_histogram)):
            level = i
            fail_ratio = inv_ratio
            if i > 0:
                prev_ratios = histogram[:i]
                pass_ratio = reduce((lambda x, y: x * y), prev_ratios)
                fail_ratio = fail_ratio * pass_ratio
            avg_level += level * fail_ratio
        final_avg_level = avg_level + reduce((lambda x, y: x * y), histogram) * max_level
        print("Average level: {}".format(final_avg_level))

class MasterAgent():
    def __init__(self,
                 train_steps=1000,
                 env_func=None,
                 curiosity=False,
                 num_actions=4,
                 state_size=[4],
                 stack_size=6,
                 sparse_stack_size=4,
                 action_stack_size=4,
                 max_floor=5,
                 boredom_thresh=10,
                 actor_fc=None,
                 conv_size=None,
                 memory_path="/tmp/a3c/visuals",
                 save_path="/tmp/a3c",
                 load_path=None):

        self.memory_path = memory_path
        self.save_path = save_path
        self.max_floor = max_floor
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.train_steps = train_steps
        self.num_actions = num_actions
        self.state_size = state_size
        self.stack_size = stack_size
        self.sparse_stack_size = sparse_stack_size
        self.action_stack_size = action_stack_size
        self.actor_fc = actor_fc
        self.conv_size = conv_size

        self.boredom_thresh = boredom_thresh

        self.env_func = env_func
        self.curiosity = curiosity
        if curiosity:
            self.global_model = CuriosityModel(num_actions=self.num_actions,
                                               state_size=self.state_size,
                                               stack_size=self.stack_size,
                                               actor_fc=self.actor_fc,
                                               critic_fc=(1024,512),
                                               curiosity_fc=(1024,512))
        else:
            self.global_model = A3CModel(num_actions=self.num_actions,
                                         state_size=self.state_size,
                                         stack_size=self.stack_size,
                                         actor_fc=self.actor_fc,
                                         critic_fc=(1024,512),
                                         conv_size=self.conv_size)

        if load_path != None:
            self.global_model.load_weights(load_path)
            print("LOADED!")

    def distributed_eval(self):
        res_queue = Queue()

        workers = [Worker(idx=i,
                   num_actions=self.num_actions,
                   max_floor=self.max_floor,
                   state_size=self.state_size,
                   stack_size=self.stack_size,
                   sparse_stack_size=self.sparse_stack_size,
                   action_stack_size=self.action_stack_size,
                   actor_fc=self.actor_fc,
                   conv_size=self.conv_size,
                   boredom_thresh=self.boredom_thresh,
                   global_model=self.global_model,
                   result_queue=res_queue,
                   env_func=self.env_func,
                   curiosity=self.curiosity,
                   max_episodes=self.train_steps,
                   memory_path=self.memory_path,
                   save_path=self.save_path) for i in range(multiprocessing.cpu_count())]
                #    save_path=self.save_path) for i in range(1)]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

        all_floors = np.array([[0,0] for _ in range(self.max_floor)])
        while True:
            data = res_queue.get()
            if data is not None:
                floor, passed = data

                all_floors[floor,passed] += 1
                floor_hist = [p / (p + not_p) if p + not_p > 0 else -1 for p, not_p in all_floors]
                print("Floor histogram: {}".format(floor_hist))
                print("Floors overall: {}".format([p + not_p for p, not_p in all_floors]))
                average_level(floor_hist)
            else:
                break
        [w.join() for w in workers]
        print("Done!")
        # floors_hist = np.histogram(all_floors, 10, (0,10))
        floor_hist = [p / (p + not_p) if p + not_p > 0 else -1 for p, not_p in all_floors]
        print("Final Floor histogram: {}".format(floor_hist))
        print("Final floors overall: {}".format([p + not_p for p, not_p in all_floors]))
        average_level(floor_hist)
        return all_floors

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
                 max_floor=None,
                 state_size=[4],
                 stack_size=None,
                 sparse_stack_size=None,
                 action_stack_size=None,
                 actor_fc=None,
                 conv_size=None,
                 boredom_thresh=None,
                 global_model=None,
                 result_queue=None,
                 env_func=None,
                 curiosity=False,
                 max_episodes=None,
                 memory_path='/tmp/a3c/visuals',
                 save_path='/tmp/a3c/workers'):
        super(Worker, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        self.stack_size = stack_size
        self.sparse_stack_size = sparse_stack_size
        self.action_stack_size = action_stack_size
        self.max_floor = max_floor

        self.global_model = global_model
        if curiosity:
            self.local_model = CuriosityModel(num_actions=self.num_actions,
                                              state_size=self.state_size,
                                              stack_size=self.stack_size,
                                              actor_fc=actor_fc,
                                              critic_fc=(1024,512),
                                              curiosity_fc=(1024,512))
        else:
            self.local_model = A3CModel(num_actions=self.num_actions,
                                        state_size=self.state_size,
                                        stack_size=self.stack_size,
                                        actor_fc=actor_fc,
                                        critic_fc=(1024,512),
                                        conv_size=conv_size)
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
            floor = np.random.randint(0, self.max_floor)
            self.env.floor(floor)
            state, obs = self.env.reset()
            rolling_average_state = state
            mem.clear()
            current_episode = Worker.global_episode

            time_count = 0
            total_reward = 0
            done = False
            prev_states = [np.random.random(state.shape) for _ in range(self.stack_size)]
            passed = False
            while not done:
                prev_states = prev_states[1:] + [state]
                _deviation = tf.reduce_sum(tf.math.squared_difference(rolling_average_state, state))
                if time_count > 10 and _deviation < self.boredom_thresh:
                    possible_actions = np.delete(np.array([range(self.num_actions)]), action)
                    action = np.random.choice(possible_actions)
                else:
                    stacked_state = np.concatenate(prev_states, axis=-1).astype(np.float32)
                    action = self.local_model.actor_model(stacked_state[None, :])
                    action = np.squeeze(tf.nn.softmax(action).numpy())
                    action = np.argmax(action)

                (new_state, reward, done, _), new_obs = self.env.step(action)

                total_reward += reward
                mem.store(state, action, reward, floor)
                if reward >= 1:
                    passed = True
                    break

                time_count += 1
                state = new_state
                rolling_average_state = rolling_average_state * 0.8 + new_state * 0.2
                obs = new_obs
            print("Episode {} | Floor {} | Reward {}".format(current_episode, floor, total_reward))
            if passed:
                self.result_queue.put((floor,0))
            else:
                self.result_queue.put((floor,1))

        self.result_queue.put(None)
