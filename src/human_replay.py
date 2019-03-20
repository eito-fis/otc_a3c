import os
import argparse

import getch
import pickle

from src.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

def input_action():
    possible_actions = ['w', 'k', 'l', ' ', 'p']
    while True:
        action = getch.getch()
        if action in possible_actions:
            return possible_actions.index(action)
        else:
            print("Invalid input.")

def run(env, max_steps):
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
    parser.add_argument('--buffer-filepath', type=str, default='./buffers/human_replay_buffer')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=50)
    args = parser.parse_args()
    
    #INITIALIZE VARIABLES#
    env_filepath = args.env_filepath
    buffer_filepath = args.buffer_filepath
    episodes = args.episodes
    max_steps = args.max_steps

    #BUILD ENVIRONMENT#
    print("Building environment...")
    env = WrappedObstacleTowerEnv(env_filepath, worker_id=0, realtime_mode=True)
    print("Environment built.")

    #INSTANTIATE MEMORY BUFFER#
    buffer_file = open(buffer_filepath, 'wb+')
    memory_buffer = []
    for episode in range(episodes):
        mem = run(env, max_steps)
        memory_buffer.append(mem)
    pickle.dump(memory_buffer, buffer_file)

