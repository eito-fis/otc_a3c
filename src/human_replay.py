import os
import argparse

import getch
import pickle

from src.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv
from src.agents.a3c import Memory

def input_action():
    possible_actions = ['w', 'k', 'l', ' ', 'p']
    while True:
        action = getch.getch()
        if action in possible_actions:
            return possible_actions.index(action)
        else:
            print("Invalid input.")

def run(env):
    mem = Memory()
    current_state = env.reset()
    mem.clear()

    done = False
    while not done:
        action = input_action()
        if action == 4:
            break
        new_state, reward, done, _ = env.step(action)
        mem.store(current_state, action, reward)

    return mem

if __name__ == '__main__':
    #PARSE COMMAND-LINE ARGUMENTS#
    parser = argparse.ArgumentParser('human_replay')
    parser.add_argument('--env-filepath', type=str, default='../ObstacleTower/obstacletower')
    parser.add_argument('--output-filepath', type=str, default='./buffers/human_replay_buffer')
    parser.add_argument('--input-filepath', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--period', type=int, default=50)
    args = parser.parse_args()
    
    #INITIALIZE VARIABLES#
    env_filepath = args.env_filepath
    output_filepath = args.output_filepath
    episodes = args.episodes
    period = args.period

    #BUILD ENVIRONMENT#
    print("Building environment...")
    env = WrappedObstacleTowerEnv(env_filepath, worker_id=0, realtime_mode=True, mobilenet=True)
    print("Environment built.")

    if args.input_filepath:
        print("Loading input buffer file...")
        input_filepath = args.input_filepath
        input_buffer_file = open(input_filepath, 'rb')
        memory_buffer = pickle.load(input_buffer_file)
        input_buffer_file.close()
        print("Input buffer loaded.")
    else:
        memory_buffer = []
        print("Created new memory buffer.")

    #INSTANTIATE MEMORY BUFFER#
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    print("Time to play!")
    for episode in range(1,episodes+1):
        mem = run(env)
        memory_buffer.append(mem)
        if episode % period == 0:
            output_file = open(output_filepath, 'wb+')
            pickle.dump(memory_buffer, output_file)
            print("Finished episode {}. Memory buffer saved.".format(episode))
            output_file.close()
    
    output_file.close()