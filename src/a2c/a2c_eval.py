import os
import gin
import logging
import argparse

from src.a2c.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv
from src.a2c.actor_critic_model import ActorCriticModel

import gym
import pickle
import numpy as np

import tensorflow as tf
from tensorflow import keras

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.obs = []
        self.probs = []
        self.values = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
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

    def run(self):
        mem = Memory()

        current_episode = 0
        while current_episode < self.max_episodes:
            seed = np.random.randint(0, 100)
            self.env._obstacle_tower_env.seed(seed)
            floor = np.random.randint(0, self.max_floor)
            # self.env.floor(floor)
            state = self.env.reset()
            mem.clear()

            time_count = 0
            total_reward = 0
            done = False
            passed = False
            while not done:
                action, value = self.model.step([state])

                new_state, reward, done, _ = self.env.step(action)

                total_reward += reward
                if self.memory_dir is not None:
                    mem.store(state, action, reward)
                    # mem.probs.append(probs)
                    # mem.obs.append(obs)
                if reward == 1:
                    passed = True
                    break

                time_count += 1
                state = new_state
                # obs = new_obs
            print("Episode {} | Seed {} | Floor {} | Reward {}".format(current_episode, seed, floor, total_reward))
            if self.memory_dir is not None:
                output_filepath = os.path.join(self.memory_dir, "memory_episode_{}".format(current_episode))
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                print("Saving memory to output file path {}".format(output_filepath))
                output_file = open(output_filepath, 'wb+')
                pickle.dump(mem, output_file)
            current_episode += 1

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
                                  stack_size=4,
                                  worker_id=0,
                                  mobilenet=args.mobilenet,
                                  gray_scale=args.gray,
                                  realtime_mode=args.render)

    #BUILD MODEL#
    print("Building model...")
    model = ActorCriticModel(num_actions=4,
                             state_size=[84,84,3],
                             stack_size=4,
                             actor_fc=(1024,512),
                             critic_fc=(1024,512),
                             conv_size=((8,4,32),(4,2,64),(3,1,64)))
    if args.restore is not None:
        model.load_weights(args.restore)
    print("Model built!")
    
    #RUN EVAL#
    agent = A2C_Eval(env=env,
                     model=model,
                     memory_dir=args.memory_dir,
                     max_episodes=100,
                     max_floor=5)
    agent.run()