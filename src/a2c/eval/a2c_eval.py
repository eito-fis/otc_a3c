import os
import gin
import logging
import argparse

from src.a2c.envs.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv
from src.a2c.models.actor_critic_model import ActorCriticModel

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
        self.states = []
        self.actions = []
        self.rewards = []
        self.floor = []
        self.obs = []
        self.probs = []

    def store(self, state, action, reward, floor):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.floor.append(floor)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.floor = []
        self.obs = []
        self.probs = []

class A2C_Eval():
    def __init__(self,
                env=None,
                model=None,
                memory_dir=None,
                max_episodes=100,
                max_floor=5):
        self.env = env
        self.model = model
        self.state = self.env.reset()
        self.memory_dir = memory_dir
        self.max_episodes = max_episodes
        self.max_floor = max_floor
        self.all_floors = np.array([[0,0] for _ in range(self.max_floor)])

    def run(self):
        mem = Memory()

        current_episode = 0
        while current_episode < self.max_episodes:
            seed = np.random.randint(0, 100)
            # seed = 50
            self.env._obstacle_tower_env.seed(seed)
            # floor = np.random.randint(0, self.max_floor)
            floor = 0
            self.env.floor(floor)
            state = self.env.reset()
            mem.clear()

            time_count = 0
            total_reward = 0
            done = False
            passed = 0
            while not done:
                action, value, _ = self.model.step([state])
                # inputs = self.model.process_inputs([state])
                # logits, value = self.model.predict(inputs)
                # action = np.argmax(logits)

                new_state, reward, done, _ = self.env.step(action)

                total_reward += reward
                if self.memory_dir is not None:
                    mem.store(state, action, reward, floor)
                if reward > .95:
                    passed = 1
                    break

                time_count += 1
                state = new_state

            print("| Episode {} | Seed {} | Floor {} | Steps {} | Reward {} |".format(current_episode, seed, floor, time_count, total_reward))
            self.all_floors[floor,passed] += 1
            floor_hist = [p / (p + not_p) if p + not_p > 0 else -1 for not_p, p in self.all_floors]
            print("Floor histogram: {}".format(floor_hist))
            print("Floors overall: {}".format([p + not_p for not_p, p in self.all_floors]))
            average_level(floor_hist)

            current_episode += 1

            if self.memory_dir:
                if passed: 
                    output_filepath = os.path.join(self.memory_dir, "pass_floor{}_steps{}_episode{}".format(floor, time_count, current_episode))
                else:
                    output_filepath = os.path.join(self.memory_dir, "fail_floor{}_steps{}_episode{}".format(floor, time_count, current_episode))
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                print("Saving memory to output file path {}".format(output_filepath))
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
    env = WrappedObstacleTowerEnv(args.env_filename,
                                  stack_size=2,
                                  worker_id=0,
                                  mobilenet=args.mobilenet,
                                  gray_scale=args.gray,
                                  realtime_mode=args.render)

    #BUILD MODEL#
    print("Building model...")
    model = ActorCriticModel(num_actions=4,
                             state_size=[84,84,3],
                             stack_size=2,
                             actor_fc=(1024,512),
                             critic_fc=(1024,512),
                             conv_size=((8,4,16),(4,2,32),(3,1,64)))
    if args.restore is not None:
        model.load_weights(args.restore)
    print("Model built!")
    
    #RUN EVAL#
    agent = A2C_Eval(env=env,
                     model=model,
                     memory_dir=args.memory_dir,
                     max_episodes=1000,
                     max_floor=5)
    agent.run()
