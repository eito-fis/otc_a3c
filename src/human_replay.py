import os
import argparse

import getch
import pickle

from collections import Counter
import random
import numpy as np

from src.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv
from src.agents.a3c import Memory

def input_action():    # 0    1    2    3    4    5    6    7    8    9
    possible_actions = ['w', 'k', 'l', ' ', 'p', 's', 'd', 'a', ',', '.']
    while True:
        action = getch.getch()
        if action in possible_actions:
            return possible_actions.index(action)
        else:
            print("Invalid input.")

def run(env, save_obs):
    mem = Memory()
    current_state, observation = env.reset()
    mem.clear()

    done = False
    while not done:
        action = input_action()
        if action == 4:
            break
        elif action > 7:
            mem.store(current_state, action, reward)
            print("Custom action {} stored!".format(action))
            if save_obs: mem.obs.append(observation)
            continue
        (new_state, reward, done, _), new_observation = env.step(action)
        mem.store(current_state, action, reward)
        if save_obs: mem.obs.append(observation)
        current_state = new_state
        observation = new_observation

    return mem

def augment_data(memory_buffer, num_duplications, num_remove_actions):
    for memory in memory_buffer:
        for i in range(len(memory.actions)):
            if memory.actions[i] > 0:
                for _ in range(num_duplications):
                    memory.store(memory.states[i], memory.actions[i], memory.rewards[i])

    i = 0
    while i < num_remove_actions:
        memory = random.choice(memory_buffer)
        forward_actions = np.where(np.array(memory.actions) == 0)[0]
        if forward_actions.any():
            action = np.random.choice(forward_actions)
            del memory.actions[action]
            del memory.states[action]
            del memory.rewards[action]
            i += 1

def data_statistics(memory_buffer):
    count_forward = sum(action == 0 for memory in memory_buffer for action in memory.actions)
    count_left = sum(action == 1 for memory in memory_buffer for action in memory.actions)
    count_right = sum(action == 2 for memory in memory_buffer for action in memory.actions)
    print("forward: {}\nleft: {}\nright: {}".format(count_forward, count_left, count_right))

def concatenate_memories(memories_dir, output_filepath):
    memory_files_list = os.listdir(memories_dir)
    if '.DS_Store' in memory_files_list:
        memory_files_list.remove('.DS_Store')
    complete_memory = []
    print("Concatenating memories from {} ...".format(memories_dir))
    for memory_filename in memory_files_list:
        memory_filepath = os.path.join(memories_dir, memory_filename)
        mem_file = open(memory_filepath, 'rb')
        memory = pickle.load(mem_file)
        mem_file.close()
        complete_memory = complete_memory + memory
    output_file = open(output_filepath, 'wb+')
    pickle.dump(complete_memory, output_file)
    output_file.close()
    print("Saved concatenated memory to {}".format(output_filepath))
    exit()

def custom_memory(custom_filepath, output_filepath):
    print("Generating custom memory...")
    input_filepath = custom_filepath
    input_buffer_file = open(input_filepath, 'rb')
    memory_buffer = pickle.load(input_buffer_file)
    input_buffer_file.close()
    
    custom_memories = []
    for memory in memory_buffer:
        for i in range(len(memory.actions)):
            if memory.actions[i] > 7:
                custom_memory = Memory()
                custom_memory.store(memory.states[i], memory.actions[i] - 7, memory.rewards[i])
                custom_memories.append(custom_memory)
    output_file = open(output_filepath, 'wb+')
    pickle.dump(custom_memories, output_file)
    output_file.close()
    print("Saved custom memory to {}".format(output_filepath))
    exit()

if __name__ == '__main__':
    #PARSE COMMAND-LINE ARGUMENTS#
    parser = argparse.ArgumentParser('human_replay')
    parser.add_argument('--env-filepath', type=str, default='../ObstacleTower/obstacletower')
    parser.add_argument('--output-filepath', type=str, default='./human_replay/memory_human_replay')
    parser.add_argument('--input-filepath', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--period', type=int, default=50)
    parser.add_argument('--augment', default=False, action='store_true')
    parser.add_argument('--save-obs', default=False, action='store_true')
    parser.add_argument('--floor', type=int, default=0)
    parser.add_argument('--concat-dir', type=str, default=None)
    parser.add_argument('--custom-filepath', type=str, default=None)
    args = parser.parse_args()
    
    #INITIALIZE VARIABLES#
    env_filepath = args.env_filepath
    output_filepath = args.output_filepath
    episodes = args.episodes
    period = args.period

    if args.concat_dir:
        concatenate_memories(args.concat_dir, output_filepath)
    
    if args.custom_filepath:
        custom_memory(args.custom_filepath, output_filepath)

    if args.input_filepath:
        print("Loading input buffer file...")
        input_buffer_file = open(args.input_filepath, 'rb')
        memory_buffer = pickle.load(input_buffer_file)
        input_buffer_file.close()
        # for memory in memory_buffer:
        #     print(memory.actions)
        #     input()
        print("Input buffer loaded.")
    else:
        memory_buffer = []
        print("Created new memory buffer.")

    if args.augment:
        #DATA AUGMENTATION#
        if not args.input_filepath:
            print("Input file must be specified for augementation.")
            exit()
        output_file = open(output_filepath, 'wb+')
        num_duplications = 4
        num_remove_actions = 1500
        data_statistics(memory_buffer)
        augment_data(memory_buffer, num_duplications, num_remove_actions)
        data_statistics(memory_buffer)
        pickle.dump(memory_buffer, output_file)
    else:
        #BUILD ENVIRONMENT#
        print("Building environment...")
        env = WrappedObstacleTowerEnv(env_filepath, worker_id=0, realtime_mode=True, mobilenet=True, floor=args.floor)
        print("Environment built.")

        #INSTANTIATE MEMORY BUFFER#
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        print("Time to play!")
        for episode in range(1,episodes+1):
            mem = run(env, args.save_obs)
            memory_buffer.append(mem)
            if episode % period == 0:
                output_file = open(output_filepath, 'wb+')
                pickle.dump(memory_buffer, output_file)
                print("Finished episode {}. Memory buffer saved.".format(episode))
                output_file.close()
        if episode % period != 0:
            output_file = open(output_filepath, 'wb+')
            pickle.dump(memory_buffer, output_file)
        
    output_file.close()
