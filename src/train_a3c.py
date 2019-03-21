# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gin
import logging
import argparse

from src.agents.a3c import MasterAgent, Memory
from src.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv

import gym
import numpy as np

import tensorflow as tf
from tensorflow import keras

def main(args,
         initial_train_steps=50,
         num_episodes=1000,
         log_period=5,
         save_period=5,
         actor_fc=(256, 128),
         critic_fc=(256, 128),
         num_actions=3,
         state_size=1280,
         realtime_mode=True):
    #env = gym.make('CartPole-v0')

    def env_func(idx):
        return WrappedObstacleTowerEnv(args.env_filename,
                                       worker_id=idx,
                                       realtime_mode=realtime_mode)

    log_dir = os.path.join(args.output_dir, "log")
    save_dir = os.path.join(args.output_dir, "checkpoints")
    summary_writer = tf.summary.create_file_writer(log_dir)

    master_agent = MasterAgent(num_episodes=num_episodes,
                               num_actions=num_actions,
                               state_size=state_size,
                               env_func=env_func,
                               actor_fc=actor_fc,
                               critic_fc=critic_fc,
                               summary_writer=summary_writer,
                               save_path=save_dir)

    
    if args.human_input != None:
        print("Starting train on human input...")
        master_agent.human_train(args.human_input, initial_train_steps)
        print("Train done!")

    print("Starting train...")
    master_agent.distributed_train()
    print("Train done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('slideshow rl')
    # Input data arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data')
    parser.add_argument(
        '--env-filename',
        type=str,
        default=None)
    parser.add_argument(
        '--human-input',
        type=str,
        default=None)
    parser.add_argument(
        '--job-dir',
        type=str,
        default='/tmp/slideshow_rl')
    parser.add_argument(
        '--n-epoch',
        type=int,
        default=1000)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    
    main(args)
