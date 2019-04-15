
import os
import gin
import logging
import argparse

from src.a2c.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv
from src.a2c.a2c_agent import A2CAgent

import numpy as np
import tensorflow as tf

def main(args,
         train_steps=2500,
         entropy_discount=0.01,
         value_discount=0.1,
         learning_rate=0.00042,
         num_steps=650,
         num_envs=4,
         num_actions=4,
         stack_size=2,
         actor_fc=[1024,512],
         critic_fc=[1024,512],
         conv_size=((8,4,32), (4,2,64), (3,1,64)),
         logging_period=1,
         checkpoint_period=10):

    '''
    Train an A2C agent
    train_steps: Number of episodes to play and train on
    entropy_discount: Amount to discount entropy loss by relative to policy loss
    value_discount: Amount to discount value loss by relative to policy loss
    learning_rate: Learning_rate
    num_steps: Number of steps for each environment to take per rollout
    num_envs: Number of environments to run in parallel
    num_actions: Number of actions for model to output
    actor_fc: Actor model dense layers topology
    critic_fc: Critic model dense layers topology
    conv_size: Conv model topology
    '''

    def env_func(idx):
        return WrappedObstacleTowerEnv(args.env_filename,
                                       stack_size=stack_size,
                                       worker_id=idx,
                                       mobilenet=args.mobilenet,
                                       gray_scale=args.gray,
                                       realtime_mode=args.render)

    print("Building agent...")
    agent = A2CAgent(train_steps=train_steps,
                     entropy_discount=entropy_discount,
                     value_discount=value_discount,
                     learning_rate=learning_rate,
                     num_steps=num_steps,
                     env_func=env_func,
                     num_envs=num_envs,
                     num_actions=num_actions,
                     actor_fc=actor_fc,
                     critic_fc=critic_fc,
                     conv_size=conv_size,
                     logging_period=logging_period,
                     checkpoint_period=checkpoint_period,
                     output_dir=args.output_dir,
                     restore_dir=args.restore)
    print("Agent built!")

    print("Strating train...")
    agent.train()
    print("Train done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train A2C')
    # Directory path arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/tmp/a2c')

    # File path arguments
    parser.add_argument(
        '--restore',
        type=str,
        default=None)
    parser.add_argument(
        '--env-filename',
        type=str,
        default='../ObstacleTower/obstacletower')

    # Run mode arguments
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
