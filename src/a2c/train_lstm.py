
import os
import gin
import logging
import argparse

from src.a2c.envs.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv
from src.a2c.envs.lstm_env import LSTMEnv
from src.a2c.agents.lstm_agent import LSTMAgent

import numpy as np
import tensorflow as tf

def main(args,
         train_steps=2500,
         update_epochs=10,
         num_minibatches=8,
         learning_rate=0.00042,
         entropy_discount=0.001,
         value_discount=0.5,
         epsilon=0.1,
         num_steps=601,
         num_envs=16,
         num_actions=4,
         stack_size=1,
         actor_fc=[256],
         critic_fc=[256],
         before_fc=[512],
         lstm_size=256,
         conv_size="quake",
         logging_period=1,
         checkpoint_period=10):

    '''
    Train a PPO agent

    train_steps: Number of episodes to play and train on
    update_epochs: Number of update epochs to run per train step
    num_minibatches: Number of batches to split one train step batch into
    learning_rate: Learning_rate
    entropy_discount: Amount to discount entropy loss by relative to policy loss
    value_discount: Amount to discount value loss by relative to policy loss
    epsilon: Amount to clip action probability by
    num_steps: Number of steps for each environment to take per rollout
    num_envs: Number of environments to run in parallel
    num_actions: Number of actions for model to output
    actor_fc: Actor model dense layers topology
    critic_fc: Critic model dense layers topology
    conv_size: Conv model topology
    '''

    if args.wandb:
        import wandb
        if args.wandb_name != None:
            wandb.init(name=args.wandb_name,
                       project="obstacle-tower-challenge",
                       entity="42 Robolab")
        else:
            wandb.init(project="obstacle-tower-challenge",
                       entity="42 Robolab")
        wandb.config.update({"learning_rate": learning_rate,
                             "entropy_discount": entropy_discount,
                             "value_discount": value_discount,
                             "epsilon": epsilon,
                             "num_steps": num_steps,
                             "num_envs": num_envs,
                             "num_actions": num_actions,
                             "stack_size": stack_size,
                             "actor_fc": actor_fc,
                             "critic_fc": critic_fc,
                             "conv_size": conv_size,
                             "retro": args.retro,
                             "gae": args.gae})
    else: wandb = None

    def env_func(idx):
        return LSTMEnv(args.env_filename,
                       stack_size=stack_size,
                       worker_id=idx,
                       mobilenet=args.mobilenet,
                       gray_scale=args.gray,
                       realtime_mode=args.render,
                       retro=args.retro)

    print("Building agent...")
    agent = LSTMAgent(train_steps=train_steps,
                     update_epochs=update_epochs,
                     num_minibatches=num_minibatches,
                     learning_rate=learning_rate,
                     entropy_discount=entropy_discount,
                     value_discount=value_discount,
                     epsilon=epsilon,
                     num_steps=num_steps,
                     env_func=env_func,
                     num_envs=num_envs,
                     num_actions=num_actions,
                     actor_fc=actor_fc,
                     critic_fc=critic_fc,
                     conv_size=conv_size,
                     before_fc=before_fc,
                     lstm_size=lstm_size,
                     gae=args.gae,
                     retro=args.retro,
                     logging_period=logging_period,
                     checkpoint_period=checkpoint_period,
                     output_dir=args.output_dir,
                     restore_dir=args.restore,
                     wandb=wandb)
    print("Agent built!")

    print("Strating train...")
    agent.train()
    print("Train done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train PPO')
    # Directory path arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/tmp/ppo')

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
    parser.add_argument(
        '--gae',
        default=False,
        action='store_true')
    parser.add_argument(
        '--retro',
        default=False,
        action='store_true')

    # WandB flags
    parser.add_argument(
        '--wandb',
        default=False,
        action='store_true')
    parser.add_argument(
        '--wandb-name',
        type=str,
        default=None)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    main(args)
