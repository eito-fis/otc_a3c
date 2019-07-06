import os
import gin
import logging
import argparse

from src.a2c.envs.lstm_env import LSTMEnv
from src.a2c.models.lstm_actor_critic_model import LSTMActorCriticModel

import gym
import pickle
import numpy as np
from functools import reduce

import tensorflow as tf
from tensorflow import keras

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

class Memory:
    def __init__(self):
        self.obs = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.floor = []
        self.probs = []
        self.dones = []

    def store(self, obs, action, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.obs = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.floor = []
        self.probs = []
        self.dones = []

class LSTM_Eval():
    def __init__(self,
                env=None,
                model=None,
                memory_dir=None,
                max_episodes=100,
                max_floor=5):
        self.env = env
        self.model = model
        self.memory_dir = memory_dir
        self.max_episodes = max_episodes
        self.max_floor = max_floor
        self.all_floors = np.array([[0,0] for _ in range(self.max_floor)])

        self.done = np.zeros((1,)).astype(np.float32)
        self.done[0] = 1
        self.state = np.zeros((1, model.lstm_size * 2)).astype(np.float32)

    def run(self):
        mem = Memory()

        current_episode = 0
        while current_episode < self.max_episodes:
            seed = np.random.randint(0, 100)
            self.env._obstacle_tower_env.seed(seed)
            floor = 0 # np.random.randint(0, self.max_floor)
            self.env.floor(floor)
            self.ob, self.info  = self.env.reset()

            mem.clear()
            time_count = 0
            reward = 0
            total_reward = 0
            new_done = 0
            passed = 0
            while not new_done or reward > .95:
                action, value, prob, self.state = self.model.step(np.asarray([self.ob]), self.state, self.done)
                # inputs = self.model.process_inputs(np.asarray([self.ob]))
                # logits, value, self.state = self.model([inputs, self.state, self.done])
                # action = np.argmax(tf.squeeze(logits))

                new_ob, reward, new_done, self.info = self.env.step(action)

                total_reward += reward
                if self.memory_dir is not None:
                    mem.store(self.ob, action, reward, self.done)
                if reward > .95:
                    passed = 1
                    floor += 1

                self.ob = new_ob
                self.done = np.array([new_done])
                time_count += 1

                if floor == 10:
                    break

            print("| Episode {} | Seed {} | Floor {} | Steps {} | Reward {} |".format(current_episode, seed, floor, time_count, total_reward))
            self.all_floors[floor,passed] += 1
            floor_hist = [p / (p + not_p) if p + not_p > 0 else -1 for not_p, p in self.all_floors]
            print("Floor histogram: {}".format(floor_hist))
            print("Floors overall: {}".format([p + not_p for not_p, p in self.all_floors]))
            average_level(floor_hist)
            current_episode += 1

            if self.memory_dir is not None:
                if passed:
                    output_filepath = os.path.join(self.memory_dir, "pass_floor{}_steps{}_episode{}".format(floor, time_count, current_episode))
                else:
                    output_filepath = os.path.join(self.memory_dir, "fail_floor{}_steps{}_episode{}".format(floor, time_count, current_episode))
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                print("Saving memory to output filepath {}".format(output_filepath))
                output_file = open(output_filepath, 'wb+')
                pickle.dump(mem, output_file)

if __name__ == '__main__':
    #COMMAND LINE ARGUMENTS#
    parser = argparse.ArgumentParser('OTC - A2C Evaluation')
    parser.add_argument('--env-filename', type=str, default='../ObstacleTower/obstacletower')
    parser.add_argument('--memory-dir', type=str, default=None)
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--gray', default=False, action='store_true')
    parser.add_argument('--mobilenet', default=False, action='store_true')
    parser.add_argument('--autoencoder', type=str, default=None)
    args = parser.parse_args()

    #INITIALIZE ENVIRONMENT#
    env = LSTMEnv(args.env_filename,
                  stack_size=1,
                  worker_id=0,
                  realtime_mode=args.render,
                  retro=True)

    #BUILD MODEL#
    print("Building model...")
    model = LSTMActorCriticModel(num_actions=6,
                                 state_size=env.state_size,
                                 stack_size=env.stack_size,
                                 num_steps=40,
                                 num_envs=1,
                                 max_floor=25,
                                 before_fc=[256],
                                 actor_fc=[128],
                                 conv_size="quake",
                                 critic_fc=[128],
                                 lstm_size=256)
    if args.restore is not None:
        model.load_weights(args.restore)
    print("Model built!")
    
    #RUN EVAL#
    agent = LSTM_Eval(env=env,
                     model=model,
                     memory_dir=args.memory_dir,
                     max_episodes=1000,
                     max_floor=15)
    agent.run()
