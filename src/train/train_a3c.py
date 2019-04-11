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
         human_train_steps=500,
         human_batch_size=1000,
         rl_episodes=1000,
         num_actions=4,
         state_size=[84,84,1],
         stack_size=4,
         actor_fc=(1024, 512),
         critic_fc=(1024, 512),
         conv_size=((8,4,32), (4,2,64), (3,1,64)),
         learning_rate=0.00042,
         gamma=0.99,
         entropy_discount=0.01,
         value_discount=0.5,
         boredom_thresh=10,
         update_freq=650,
         log_period=25,
         checkpoint_period=50,
         memory_period=1):

    # Function that builds the environment
    def env_func(idx):
        return WrappedObstacleTowerEnv(args.env_filename,
                                       worker_id=idx,
                                       mobilenet=args.mobilenet,
                                       gray_scale=args.gray,
                                       realtime_mode=args.render)

    # Make sure out logging dirs exist and build the summary writer
    log_dir = os.path.join(args.output_dir, "log")
    save_dir = os.path.join(args.output_dir, "checkpoints")
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Build the agent!
    master_agent = MasterAgent(num_episodes=rl_episodes,
                               num_actions=num_actions,
                               state_size=state_size,
                               stack_size=stack_size,
                               actor_fc=actor_fc,
                               critic_fc=critic_fc,
                               conv_size=conv_size,
                               learning_rate=learning_rate,
                               env_func=env_func,
                               gamma=gamma,
                               entropy_discount=entropy_discount,
                               value_discount=value_discount,
                               boredom_thresh=boredom_thresh,
                               update_freq=update_freq,
                               summary_writer=summary_writer,
                               load_path=args.restore,
                               memory_path=args.memory_dir,
                               save_path=save_dir,
                               log_period=log_period,
                               checkpoint_period=checkpoint_period,
                               memory_period=memory_period)

    if args.eval:
        # If runnig in eval mode, only eval
        print("Starting evaluation...")
        env = env_func(0)
        master_agent.play(env)
        print("Evaluation done!")
    else:
        # Otherwise, train on human input if passed...
        if args.human_input != None:
            print("Starting train on human input...")
            master_agent.human_train(args.human_input, human_train_steps, human_batch_size)
            print("Train done!")

        # ...then train as RL
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
