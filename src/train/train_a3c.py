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
         human_train_steps=2500,
         human_batch_size=1000,
         rl_episodes=10000,
         num_actions=4,
         state_size=[1280],
         stack_size=10,
         actor_fc=(1024, 512),
         critic_fc=(1024, 512),
         learning_rate=0.00042,
         gamma=0.99,
         entropy_discount=0.05,
         value_discount=0.5,
         boredom_thresh=10,
         update_freq=650,
         log_period=25,
         checkpoint_period=50,
         memory_period=1):
    '''
    Main
    - Build and runs our agent
    Arguments:
    human_train_steps: Total steps to train on human input if there is human input
    human_batch_size: Batch size to use for training on human input if there is human input
    rl_episodes: Total number of episodes to be played by workers
    num_actions: Number of possible actions in the environment
    state_size: Expected size of state returned from environment
    stack_size: Number of stacked frames our model will consider
    actor_fc: Iterable containing the amount of neurons per layer for the actor model
    critic_fc: Iterable containing the amount of neurons per layer for the critic model
        ex: (1024, 512, 256) would make 3 fully connected layers, with 1024, 512 and 256
            layers respectively
    learning_rate: Learning rate
    env_func: Callable function that builds an environment for each worker. Will be passed an idx
    gamma: Decay coefficient used to discount future reward while calculating loss
    entropy_discount: Discount coefficient used to control our entropy loss
    value_discount: Discount coefficient used to control our critic model loss
    boredom_thresh: Threshold for the standard deviation of previous frames - controls how easily
        our model gets bored
    update_freq: Number of time steps each worker takes before calcualting gradient and updating
        global model
    log_period: Number of epochs to wait before writing log information
    checkpoint_period: Numer of epochs to wait before saving a model
    memory_period: Number of epochs to wait before saving a memory
    '''

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
    parser = argparse.ArgumentParser('A3C OTC')
    # Directory path arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data')
    parser.add_argument(
        '--memory-dir',
        type=str,
        default=None)

    # File path arguments
    parser.add_argument(
        '--restore',
        type=str,
        default=None)
    parser.add_argument(
        '--env-filename',
        type=str,
        default='../ObstacleTower/obstacletower')
    parser.add_argument(
        '--human-input',
        type=str,
        default=None)

    # Run mode arguments
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
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    main(args)
