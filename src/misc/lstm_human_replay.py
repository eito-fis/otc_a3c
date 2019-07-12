import os
import argparse

import getch
import pickle

from collections import Counter
import random
import numpy as np

from src.a2c.envs.lstm_env import LSTMEnv
from src.a2c.models.lstm_actor_critic_model import LSTMActorCriticModel
from src.a2c.eval.lstm_eval import Memory

def concatenate_memories(memories_dir, output_filepath):
    memory_files_list = os.listdir(memories_dir)
    if '.DS_Store' in memory_files_list:
        memory_files_list.remove('.DS_Store')
    complete_memory = []
    print("Concatenating memories from {} ...".format(memories_dir))
    for memory_filename in memory_files_list:
        memory_filepath = os.path.join(memories_dir, memory_filename)
        mem_file = open(memory_filepath, 'rb')
        memory_list = pickle.load(mem_file)
        mem_file.close()
        for memory in memory_list:
            complete_memory.append(memory)
    output_file = open(output_filepath, 'wb+')
    pickle.dump(complete_memory, output_file)
    output_file.close()
    print("Saved concatenated memory to {}".format(output_filepath))
    exit()

def input_action():  #   0    1    2    3    4    5    6
    possible_actions = ['w', 'k', 'l', ' ', ',', '.', 'p']
    while True:
        action = getch.getch()
        if action in possible_actions:
            return possible_actions.index(action)
        else:
            print("Invalid input.")

def run(env, floor, seed, model):
    seed = np.random.randint(0, 100) if seed == -1 else seed
    env._obstacle_tower_env.seed(seed)
    env.floor(floor)

    mem = Memory()
    observation, info = env.reset()
    new_done, done = np.zeros((1,)).astype(np.float32), np.zeros((1,)).astype(np.float32)
    done[0] = 1
    if model is not None:
        state = np.zeros((1, model.lstm_size * 2)).astype(np.float32)
    
    while not new_done:
        expert_action = input_action()
        if model is not None and expert_action == 6:
            while expert_action == 6:
                expert_action = input_action()
            action, _, _, state = model.step(np.asarray([observation]), state, new_done)
        else:
            action = expert_action
        new_observation, reward, new_done, info = env.step(action)
        mem.store(observation, expert_action, reward, done)
        observation = new_observation
        new_done = np.array([new_done])
        done = new_done
    return mem

if __name__ == '__main__':
    #PARSE COMMAND-LINE ARGUMENTS#
    parser = argparse.ArgumentParser('human_replay')
    parser.add_argument('--env-filepath', type=str, default='../ObstacleTower/obstacletower')
    parser.add_argument('--output-filepath', type=str, default='./human_replay/memory_human_replay')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--period', type=int, default=1)
    parser.add_argument('--floor', type=int, default=10)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--concat-dir', type=str, default=None)
    parser.add_argument('--restore', type=str, default=None)
    args = parser.parse_args()

    #INITIALIZE VARIABLES#
    env_filepath = args.env_filepath
    output_filepath = args.output_filepath
    episodes = args.episodes
    period = args.period

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    if args.concat_dir:
        concatenate_memories(args.concat_dir, output_filepath)

    #BUILD ENVIRONMENT#
    print("Building environment...")
    env = LSTMEnv(env_filepath,
                  stack_size=1,
                  worker_id=0,
                  realtime_mode=True,
                  retro=True)
    print("Environment built.")

    if args.restore:
        print("Restoring model...")
        model = LSTMActorCriticModel(num_actions=6,
                                     state_size=env.state_size,
                                     stack_size=env.stack_size,
                                     num_steps=40,
                                     num_envs=1,
                                     max_floor=25,
                                     before_fc=[256],
                                     actor_fc=[128],
                                     conv_size="quake",
                                     critic_fc=[128],
                                     lstm_size=256)
        model.load_weights(args.restore)
        print("Model restored.")
    else:
        model = None

    #INSTANTIATE MEMORY BUFFER#
    memory_buffer = []
    print("Time to play!")
    for episode in range(1,episodes+1):
        print("Current episode: {}".format(episode))
        mem = run(env, args.floor, args.seed, model)
        memory_buffer.append(mem)
        if episode % period == 0:
            output_file = open(output_filepath, 'wb+')
            pickle.dump(memory_buffer, output_file)
            print("Finished episode {}. Memory buffer saved to {}".format(episode, output_filepath))
            output_file.close()
    output_file = open(output_filepath, 'wb+')
    pickle.dump(memory_buffer, output_file)
    output_file.close()
    print("Finished human replay collection. Memory buffer saved to {}".format(episode, output_filepath))