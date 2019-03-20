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

from src.agents.a3c import MasterAgent
from src.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv

import gym
import numpy as np

import tensorflow as tf
from tensorflow import keras

def main(args,
         log_period=5,
         save_period=5,
         actor_fc=(256, 128),
         critic_fc=(256, 128),
         num_actions=3,
         state_size=1280):
    #env = gym.make('CartPole-v0')

    def env_func(idx):
        return WrappedObstacleTowerEnv(args.env_filename, worker_id=idx)

    summary_writer = tf.summary.create_file_writer(args.logdir)

    master_agent = MasterAgent(num_actions=num_actions,
                               state_size=state_size,
                               env_func=env_func,
                               actor_fc=actor_fc,
                               critic_fc=critic_fc)

    print("Starting train...")
    master_agent.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('slideshow rl')
    # Input data arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data')
    parser.add_argument(
        '--logdir',
        type=str,
        default=None)
    parser.add_argument(
        '--env-filename',
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
