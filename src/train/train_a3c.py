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

# import time
# import matplotlib
# matplotlib.use('PS')
import matplotlib.pyplot as plt

def main(args,
         initial_train_steps=100,
         num_episodes=0,
         log_period=25,
         save_period=50,
         visual_period=1,
         actor_fc=(1024, 512),
         critic_fc=(1024, 512),
         conv_size=((8,4,32), (4,2,64), (3,1,64)),
         num_actions=4,
         stack_size=4,
         sparse_stack_size=0,
         sparse_update=5,
         action_stack_size=0,
         state_size=[84,84,1],
         batch_size=100,
         realtime_mode=True):
    realtime_mode = args.render

    def env_func(idx):
        return WrappedObstacleTowerEnv(args.env_filename,
                                       worker_id=idx,
                                       mobilenet=args.mobilenet,
                                       gray_scale=args.gray,
                                       realtime_mode=realtime_mode,
                                       autoencoder=args.autoencoder)

    log_dir = os.path.join(args.output_dir, "log")
    save_dir = os.path.join(args.output_dir, "checkpoints")
    summary_writer = tf.summary.create_file_writer(log_dir)

    master_agent = MasterAgent(num_episodes=num_episodes,
                               num_actions=num_actions,
                               state_size=state_size,
                               conv_size=conv_size,
                               env_func=env_func,
                               stack_size=stack_size,
                               sparse_stack_size=sparse_stack_size,
                               sparse_update=sparse_update,
                               action_stack_size=action_stack_size,
                               actor_fc=actor_fc,
                               critic_fc=critic_fc,
                               summary_writer=summary_writer,
                               save_path=save_dir,
                               memory_path=args.memory_dir,
                               visual_period=visual_period,
                               load_path=args.restore)

    if args.eval:
        reached_floors = []
        print("Starting evaluation...")
        env = env_func(0)
        for _ in range(100):
            reached_floors.append(master_agent.play(env))
            floors_hist = np.histogram(reached_floors, 5, (0,5))
            print(floors_hist)
        plt.hist(reached_floors, 5, (0,5))
        plt.show()
        print("Evaluation done!")
    else:
        if args.human_input != None:
            print("Starting train on human input...")
            master_agent.human_train(args.human_input, initial_train_steps, batch_size)
            print("Train done!")

        print("Starting train...")
        master_agent.distributed_train()
        print("Train done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('OTC - 42RoboLab')
    # Input data arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data')
    parser.add_argument(
        '--env-filename',
        type=str,
        default='../ObstacleTower/obstacletower')
    parser.add_argument(
        '--human-input',
        type=str,
        default=None)
    parser.add_argument(
        '--memory-dir',
        type=str,
        default=None)
    parser.add_argument(
        '--restore',
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
    parser.add_argument(
        '--render',
        default=False,
        action='store_true')
    parser.add_argument(
        '--eval',
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
    parser.add_argument(
        '--autoencoder',
        type=str,
        default=None)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    main(args)