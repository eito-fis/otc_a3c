
import numpy as np
from src.agents.a3c import Memory, ActorCriticModel
from src.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv
import os
import tensorflow as tf
import argparse
from PIL import Image

def run(env, model, state_size, stack_size, boredom_thresh, num_actions):
    mem = Memory()
    current_state, observation = env.reset()
    mem.clear()
    step_count = 0

    done = False
    rolling_average_state = np.zeros(state_size[0]) + (0.2 * current_state)
    while not done:
        _deviation = tf.reduce_sum(tf.math.squared_difference(rolling_average_state, current_state))
        if step_count > 10 and _deviation < boredom_thresh:
            possible_actions = np.delete(np.array([0, 1, 2]), action)
            action = np.random.choice(possible_actions)
            distribution = np.zeros(num_actions)
            value = 100
        else:
            stacked_state = [np.zeros_like(current_state) if step_count - i < 0
                                                  else mem.states[step_count - i].numpy()
                                                  for i in reversed(range(1, stack_size))]
            stacked_state.append(current_state)
            stacked_state = np.concatenate(stacked_state)
            action, _ = model.get_action_value(stacked_state[None, :])

        (new_state, reward, done, _), new_observation = env.step(action)

        mem.store(current_state, action, reward)
        mem.obs.append(observation)

        current_state = new_state
        rolling_average_state = rolling_average_state * 0.8 + new_state * 0.2
        observation = new_observation
        step_count += 1

        if reward == 1:
            return mem

    return None


if __name__ == '__main__':
    #PARSE COMMAND-LINE ARGUMENTS#
    parser = argparse.ArgumentParser('data distallation')
    parser.add_argument('--env-filepath', type=str, default='../ObstacleTower/obstacletower')
    parser.add_argument('--output-dir', type=str, default='./buffers/human_replay_buffer')
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--frames', type=int, default=600)
    parser.add_argument('--floor', type=int, default=0)
    parser.add_argument('--realtime', default=False, action='store_true')
    args = parser.parse_args()
    
    state_size = [1280]
    stack_size = 4
    num_actions = 3
    boredom_thresh = 10

    model = ActorCriticModel(num_actions=num_actions,
                             state_size=state_size,
                             stack_size=stack_size,
                             actor_fc=(1024, 512),
                             critic_fc=(1024, 512))
    if args.restore != None:
        model.load_weights(args.restore)


    forward_dir = os.path.join(args.output_dir, "forward/")
    left_dir = os.path.join(args.output_dir, "left/")
    right_dir = os.path.join(args.output_dir, "right/")
    directory_lookup = [forward_dir, left_dir, right_dir]
    os.makedirs(os.path.dirname(forward_dir), exist_ok=True)
    os.makedirs(os.path.dirname(left_dir), exist_ok=True)
    os.makedirs(os.path.dirname(right_dir), exist_ok=True)

    episodes = 0
    episodes_saved = 0
    total_frames = 0
    #memory_list = []
    print("Time to play!")
    for floor in range(args.floor):
        env = WrappedObstacleTowerEnv(args.env_filepath,
                                      worker_id=0,
                                      realtime_mode=args.realtime,
                                      mobilenet=True,
                                      floor=floor)
        episode_frames = 0
        while episode_frames < args.frames:
            print("Episode {} | Floor {} | Saved {} episodes and {} total frames so far, and {} frames this floor".format(episodes, floor, episodes_saved, total_frames, episode_frames))
            mem = run(env, model, state_size, stack_size, boredom_thresh, num_actions)
            if mem != None:
                for index, (action, state) in enumerate(zip(mem.actions, mem.obs)):
                    save_directory = directory_lookup[action]
                    save_path = os.path.join(save_directory, "{}_{}.jpeg".format(episodes_saved, index))
                    im = Image.fromarray(state)
                    im.save(save_path)
                    print("Saved {}".format(save_path))
                episodes_saved += 1
                episode_frames += len(mem.actions)
                total_frame += episode_frames
                #mem.obs = []
                #memory_list.append(mem)
            else: print("Failed!")
            episodes += 1

