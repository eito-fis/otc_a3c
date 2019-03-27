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

from src.agents.eval import MasterAgent, Memory
from src.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv

import gym
import numpy as np

import tensorflow as tf
from tensorflow import keras

# import time
# import matplotlib
# matplotlib.use('PS')
import matplotlib.pyplot as plt

def main(args,
         train_steps=500,
         actor_fc=(1024, 512),
         num_actions=3,
         stack_size=4,
         state_size=[1280]):
    realtime_mode = args.render

    def env_func(idx):
        return WrappedObstacleTowerEnv(args.env_filename,
                                       worker_id=idx,
                                       mobilenet=args.mobilenet,
                                       gray_scale=args.gray,
                                       realtime_mode=realtime_mode)

    master_agent = MasterAgent(train_steps=train_steps,
                               num_actions=num_actions,
                               state_size=state_size,
                               env_func=env_func,
                               stack_size=stack_size,
                               actor_fc=actor_fc,
                               memory_path=args.memory_dir,
                               load_path=args.restore)

    print("Starting evaluation...")
    master_agent.distributed_eval()
    print("Train done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('OTC Eval')
    # Input data arguments
    parser.add_argument(
        '--env-filename',
        type=str,
        default='../ObstacleTower/obstacletower')
    parser.add_argument(
        '--memory-dir',
        type=str,
        default=None)
    parser.add_argument(
        '--restore',
        required=True,
        type=str,
        default=None)
    parser.add_argument(
        '--job-dir',
        type=str,
        default='/tmp/slideshow_rl')
    parser.add_argument(
        '--render',
        default=False,
        action='store_true')
    parser.add_argument(
        '--gray',
        default=False,
        action='store_true')
    parser.add_argument(
        '--mobilenet',
        default=False,
        action='store_true')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    main(args)
