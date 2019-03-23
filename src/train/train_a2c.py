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

from src.agents.a2c import Model, A2CAgent
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
         num_actions=3):
    #env = gym.make('CartPole-v0')
    env = WrappedObstacleTowerEnv(args.env_filename)

    summary_writer = tf.summary.create_file_writer(args.logdir)

    model = Model(num_actions=num_actions,
                  actor_fc=actor_fc,
                  critic_fc=critic_fc)
    agent = A2CAgent(model=model,
                     env=env,
                     summary_writer=summary_writer,
                     log_period=log_period,
                     save_period=save_period,
                     save_path=args.output_dir)

    print("Starting train...")
    print(agent.train(args.n_epoch))

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
