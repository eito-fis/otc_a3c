import os
import pickle
import argparse

import threading
import multiprocessing

import gym
import numpy as np

from queue import Queue

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from collections import Counter
from functools import reduce
import cv2

from src.agents.a3c import ActorCriticModel as A3CModel
from src.agents.a3c import Memory, ProbabilityDistribution
from src.agents.curiosity import ActorCriticModel as CuriosityModel
from src.wrapped_obstacle_tower_env import WrappedKerasLayer

def average_level(histogram):
        inverse_histogram = list(map(lambda x: 1 - x, histogram))
        max_level = len(histogram)
        avg_level = 0
        for i, (ratio, inv_ratio) in enumerate(zip(histogram, inverse_histogram)):
            level = i
            fail_ratio = inv_ratio
            if i > 0:
                prev_ratios = histogram[:i]
                pass_ratio = reduce((lambda x, y: x * y), prev_ratios)
                fail_ratio = fail_ratio * pass_ratio
            avg_level += level * fail_ratio
        final_avg_level = avg_level + reduce((lambda x, y: x * y), histogram) * max_level
        print("Average level: {}".format(final_avg_level))

IMAGE_SHAPE = (84, 84)
BLUR_COEFF = 20
MASK_RADIUS = 55
STRIDE = 7

def blur_images(images, radius):
  return np.array([cv2.blur(image, (radius,radius)) for image in images])
#   return np.array([cv2.GaussianBlur(image, (5,5), 0) for image in source_images])

def generate_mask(image_shape, radius, x, y):
  mask_image = np.zeros(image_shape)
  mask_image[y,x] = 1.
  mask_image = cv2.GaussianBlur(mask_image, (radius,radius), 0)
  mask_image = cv2.normalize(mask_image, None, 1, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  return mask_image

def generate_masks(image_shape, radius, stride):
  masks = []
  for y in range(image_shape[0])[::stride]:
    for x in range(image_shape[0])[::stride]:
      masks.append(generate_mask(image_shape, radius, x, y))
  return np.array(masks)

def perturb_image(source_image, blurred_image, mask):
  result = np.zeros(IMAGE_SHAPE)
  for y in range(IMAGE_SHAPE[0]):
    for x in range(IMAGE_SHAPE[1]):
        if mask[y,x] > 0: result[y,x] = np.multiply(mask[y,x],blurred_image[y,x]) + np.multiply(1-mask[y,x],source_image[y,x])
  return result

def generate_perturbations(source_image, masks, blur_coeff, stride):
  perturbed_images = []
  blurred_image = cv2.blur(source_image, (blur_coeff,blur_coeff))
  for y in range(IMAGE_SHAPE[0])[::stride]:
    for x in range(IMAGE_SHAPE[1])[::stride]:
      print(x//stride + (y//stride)*(IMAGE_SHAPE[0]//stride))
      perturbed_image = perturb_image(source_image, blurred_image, masks[x//stride + y*IMAGE_SHAPE[0]//stride])
      perturbed_images.append(perturbed_image)
  return perturbed_images

def generate_saliency(model, source_image, prev_states, masks, blur_coeff, stride):
  saliency_map = np.zeros((84,84))
  stacked_state = np.concatenate(prev_states + [source_image])
  logits = np.squeeze(model.predict(stacked_state[None,:]))
  blurred_image = cv2.blur(source_image, (blur_coeff,blur_coeff))
  for y in range(IMAGE_SHAPE[0])[::stride]:
    for x in range(IMAGE_SHAPE[1])[::stride]:
      mask = masks[x//stride + (y//stride)*(IMAGE_SHAPE[0]//stride)]
      perturbed_image = perturb_image(source_image, blurred_image, mask)
      stacked_state = np.concatenate(prev_states + [perturbed_image])
      perturbed_logits = np.squeeze(model.predict(stacked_state[None,:]))
      saliency = np.square(sum(logits - perturbed_logits)) / 2
      saliency_map = saliency_map + np.multiply(saliency, mask)
    #   saliency_map[y,x] = saliency
  saliency_map = cv2.normalize(saliency_map, None, 1, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  rgb_saliency_map = (np.zeros(IMAGE_SHAPE) + saliency_map[:,:,None]) * [1,0,0]
  return rgb_saliency_map

masks = generate_masks(IMAGE_SHAPE, MASK_RADIUS, STRIDE)
image_module = WrappedKerasLayer(retro=False, mobilenet=True)

class MasterAgent():
    def __init__(self,
                 train_steps=1000,
                 env_func=None,
                 curiosity=False,
                 num_actions=2,
                 state_size=[4],
                 stack_size=6,
                 sparse_stack_size=4,
                 action_stack_size=4,
                 max_floor=5,
                 boredom_thresh=10,
                 actor_fc=None,
                 conv_size=None,
                 memory_path="/tmp/a3c/visuals",
                 save_path="/tmp/a3c",
                 load_path=None):

        self.memory_path = memory_path
        self.save_path = save_path
        self.max_floor = max_floor
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.train_steps = train_steps
        self.num_actions = num_actions
        self.state_size = state_size
        self.stack_size = stack_size
        self.sparse_stack_size = sparse_stack_size
        self.action_stack_size = action_stack_size
        self.actor_fc = actor_fc
        self.conv_size = conv_size

        self.boredom_thresh = boredom_thresh

        self.env_func = env_func
        self.curiosity = curiosity
        if curiosity:
            self.global_model = CuriosityModel(num_actions=self.num_actions,
                                               state_size=self.state_size,
                                               stack_size=self.stack_size,
                                               actor_fc=self.actor_fc,
                                               critic_fc=(1024,512),
                                               curiosity_fc=(1024,512))
        else:
            self.global_model = A3CModel(num_actions=self.num_actions,
                                         state_size=self.state_size,
                                         stack_size=self.stack_size,
                                         actor_fc=self.actor_fc,
                                         critic_fc=(1024,512),
                                         conv_size=self.conv_size)

        if load_path != None:
            self.global_model.load_weights(load_path)
            print("LOADED!")

    def distributed_eval(self):
        res_queue = Queue()

        workers = [Worker(idx=i,
                   num_actions=self.num_actions,
                   max_floor=self.max_floor,
                   state_size=self.state_size,
                   stack_size=self.stack_size,
                   sparse_stack_size=self.sparse_stack_size,
                   action_stack_size=self.action_stack_size,
                   actor_fc=self.actor_fc,
                   conv_size=self.conv_size,
                   boredom_thresh=self.boredom_thresh,
                   global_model=self.global_model,
                   result_queue=res_queue,
                   env_func=self.env_func,
                   curiosity=self.curiosity,
                   max_episodes=self.train_steps,
                   memory_path=self.memory_path,
                   save_path=self.save_path) for i in range(multiprocessing.cpu_count())]
                   #save_path=self.save_path) for i in range(1)]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

        all_floors = np.array([[0,0] for _ in range(self.max_floor)])
        while True:
            data = res_queue.get()
            if data is not None:
                floor, passed = data

                all_floors[floor,passed] += 1
                floor_hist = [p / (p + not_p) if p + not_p > 0 else -1 for p, not_p in all_floors]
                print("Floor histogram: {}".format(floor_hist))
                print("Floors overall: {}".format([p + not_p for p, not_p in all_floors]))
                average_level(floor_hist)
            else:
                break
        [w.join() for w in workers]
        print("Done!")
        # floors_hist = np.histogram(all_floors, 10, (0,10))
        floor_hist = [p / (p + not_p) if p + not_p > 0 else -1 for p, not_p in all_floors]
        print("Final Floor histogram: {}".format(floor_hist))
        print("Final floors overall: {}".format([p + not_p for p, not_p in all_floors]))
        average_level(floor_hist)
        return all_floors

class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0.0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self,
                 idx=0,
                 num_actions=2,
                 max_floor=None,
                 state_size=[4],
                 stack_size=None,
                 sparse_stack_size=None,
                 action_stack_size=None,
                 actor_fc=None,
                 conv_size=None,
                 boredom_thresh=None,
                 global_model=None,
                 result_queue=None,
                 env_func=None,
                 curiosity=False,
                 max_episodes=None,
                 memory_path='/tmp/a3c/visuals',
                 save_path='/tmp/a3c/workers'):
        super(Worker, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        self.stack_size = stack_size
        self.sparse_stack_size = sparse_stack_size
        self.action_stack_size = action_stack_size
        self.max_floor = max_floor

        self.global_model = global_model
        if curiosity:
            self.local_model = CuriosityModel(num_actions=self.num_actions,
                                              state_size=self.state_size,
                                              stack_size=self.stack_size,
                                              actor_fc=actor_fc,
                                              critic_fc=(1024,512),
                                              curiosity_fc=(1024,512))
        else:
            self.local_model = A3CModel(num_actions=self.num_actions,
                                        state_size=self.state_size,
                                        stack_size=self.stack_size,
                                        sparse_stack_size=sparse_stack_size,
                                        action_stack_size=self.action_stack_size,
                                        actor_fc=actor_fc,
                                        critic_fc=(1024,512),
                                        conv_size=conv_size)
        self.local_model.set_weights(self.global_model.get_weights())

        print("Building environment")
        self.env = env_func(idx)
        print("Environment built!")

        self.boredom_thresh = boredom_thresh

        self.save_path = save_path
        self.memory_path = memory_path

        self.result_queue = result_queue

        self.worker_idx = idx
        self.max_episodes = max_episodes

    def run(self):
        mem = Memory()

        while Worker.global_episode < self.max_episodes:
            self.env._obstacle_tower_env.seed(np.random.randint(0, 100))
            # floor = np.random.randint(0, self.max_floor)
            floor = 0
            self.env.floor(floor)
            state, obs = self.env.reset()
            rolling_average_state = state
            mem.clear()
            current_episode = Worker.global_episode

            time_count = 0
            total_reward = 0
            done = False
            prev_states = [np.random.random(state.shape) for _ in range(self.stack_size)]
            sparse_states = [np.random.random(state.shape) for _ in range(self.sparse_stack_size)]
            prev_actions = [np.zeros(self.num_actions) for _ in range(self.action_stack_size)]
            boredom_actions = []
            passed = False
            while not done:
                prev_states = prev_states[1:] + [state]
                if time_count > 0 and self.action_stack_size > 0:
                    one_hot_action = np.zeros(self.num_actions)
                    one_hot_action[action] = 1
                    prev_actions = prev_actions[1:] + [one_hot_action]
                if self.sparse_stack_size > 0 and time_count % self.sparse_update == 0:
                    sparse_states = sparse_states[1:] + [state]
                _deviation = tf.reduce_sum(tf.math.squared_difference(rolling_average_state, state))
                if time_count > 10 and _deviation < self.boredom_thresh:
                    possible_actions = np.delete(np.array([range(self.num_actions)]), action)
                    action = np.random.choice(possible_actions)
                else:
                    stacked_state = np.concatenate(prev_states + sparse_states + prev_actions, axis=-1).astype(np.float32)
                    logits, _ = self.local_model(stacked_state[None, :])
                    probs = np.squeeze(tf.nn.softmax(logits).numpy())
                    action = np.argmax(probs)

                print("Generating saliency...")
                saliency_map = generate_saliency(self.local_model.actor_model, state, prev_states[:-1], masks, BLUR_COEFF, STRIDE)
                normed_map = cv2.normalize(saliency_map, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite('conv_saliency_output/saliency_map_{}.png'.format(time_count), cv2.cvtColor(normed_map, cv2.COLOR_BGR2RGB))
                print("Done.")

                (new_state, reward, done, _), new_obs = self.env.step(action)

                total_reward += reward
                mem.obs.append(state)
                mem.probs.append(probs)
                mem.novelty.append(saliency_map)
                if reward == 1:
                    passed = True
                    break


                time_count += 1
                state = new_state
                rolling_average_state = rolling_average_state * 0.8 + new_state * 0.2
                obs = new_obs
            print("Environment Seed: {}".format(self.env._obstacle_tower_env._seed))
            print("Episode {} | Floor {} | Reward {}".format(current_episode, floor, total_reward))
            if passed:
                self.result_queue.put((floor,0))
            else:
                self.result_queue.put((floor,1))

        self.result_queue.put(None)
