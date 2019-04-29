import numpy as np

from obstacle_tower_env import ObstacleTowerEnv
from obstacle_tower_env import ActionFlattener

import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image

class WrappedKerasLayer(tf.keras.layers.Layer):
    def __init__(self, retro, mobilenet):
        super(WrappedKerasLayer, self).__init__()
        self.layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=False)
        self.input_spec = (1, 224, 224, 3)

    def __call__(self, _input):
        _input = np.reshape(np.array(_input), self.input_spec)
        _input = tf.convert_to_tensor(_input, dtype=tf.float32)
        tensor_var = tf.convert_to_tensor(np.array(self.layer(_input)))
        tensor_var = tf.squeeze(tensor_var)
        return tensor_var

class WrappedObstacleTowerEnv():

    def __init__(
        self,
        environment_filename=None,
        docker_training=False,
        worker_id=0,
        retro=False,
        timeout_wait=30,
        realtime_mode=False,
        num_actions=3,
        mobilenet=False,
        gray_scale=False,
        floor=0,
        ):
        '''
        Arguments:
          environment_filename: The file path to the Unity executable.  Does not require the extension.
          docker_training: Whether this is running within a docker environment and should use a virtual
            frame buffer (xvfb).
          worker_id: The index of the worker in the case where multiple environments are running.  Each
            environment reserves port (5005 + worker_id) for communication with the Unity executable.
          retro: Resize visual observation to 84x84 (int8) and flattens action space.
          timeout_wait: Time for python interface to wait for environment to connect.
          realtime_mode: Whether to render the environment window image and run environment at realtime.
          num_actions: Number of actions our model will output
          mobilenet: Whether or not to use mobilenet to compress images
          gray_scale: Whether or not to convert images to grayscale and resize to 84x84
          floor: Floor to start on
        '''

        # Build environment
        self._obstacle_tower_env = ObstacleTowerEnv(environment_filename, docker_training, worker_id, retro, timeout_wait, realtime_mode)

        # Set default floor is floor isn't 0
        if floor != 0: self._obstacle_tower_env.floor(floor)

        self.mobilenet = mobilenet
        self.gray_scale = gray_scale
        if mobilenet:
            self.image_module = WrappedKerasLayer(retro, self.mobilenet)
        self._done = False
        self.id = worker_id

    def gray_process_observation(self, observation):
        '''
        Converts to grayscale and resizes to 84 x 84. Copies environments retro mode
        '''
        observation = observation[0]
        observation = (observation * 255).astype(np.uint8)
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((84, 84), Image.NEAREST)
        gray_observation = np.mean(np.array(obs_image),axis=-1,keepdims=True)
        return gray_observation / 255

    def _preprocess_observation(self, observation):
        """
        Resizes to 224x224 for mobilenet
        """
        observation = observation[0]
        observation = (observation * 255).astype(np.uint8)
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((224, 224), Image.NEAREST)
        return np.array(obs_image).astype(np.float32)

    def reset(self):
        observation = self._obstacle_tower_env.reset()
        self._done = False
        if self.mobilenet:
            mobile_observation = self._preprocess_observation(observation)
            observation, key, time = observation
            return self.image_module(mobile_observation), observation, key, time
        elif self.gray_scale:
            return self.gray_process_observation(observation), observation[0]
        else:
            return self._preprocess_observation(observation), observation

    def step(self, action):

        # Convert our scalar outputs to vectors for the environment
        if action == 0: # forward
            action = [1, 0, 0, 0]
        elif action == 1: # rotate camera left
            action = [1, 1, 0, 0]
        elif action == 2: # rotate camera right
            action = [1, 2, 0, 0]
        elif action == 3: # jump forward
            action = [1, 0, 1, 0]
        elif action == 5:
            action = [2, 0, 0, 0]
        elif action == 6:
            action = [0, 0, 0, 1]
        elif action == 7:
            action = [0, 0, 0, 2]


        observation, reward, done, info = self._obstacle_tower_env.step(action)
        self._done = done

        if self.mobilenet:
            mobile_observation = self._preprocess_observation(observation)
            observation, key, time = observation
            return (self.image_module(mobile_observation), reward, done, info), observation, key, time
        elif self.gray_scale:
            return (self.gray_process_observation(observation), reward, done, info), observation[0]
        else:
            return (self._preprocess_observation(observation), reward, done, info), observation

    def close(self):
        self._obstacle_tower_env.close()

    def floor(self, floor):
        self._obstacle_tower_env.floor(floor)
