import numpy as np

from obstacle_tower_env import ObstacleTowerEnv
from obstacle_tower_env import ActionFlattener

import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image

class AutoEncoder(tf.keras.Model):
  def __init__(self, embedding_size=128, input_size=1280):
    super(AutoEncoder, self).__init__()
    self.dense2 = tf.keras.layers.Dense(embedding_size, activation='relu')
    self.dense4 = tf.keras.layers.Dense(input_size, activation='sigmoid')
    
  def call(self, data):
    data = self.dense2(data)
    _ = self.dense4(data)
    return data

class WrappedKerasLayer(tf.keras.layers.Layer):
    def __init__(self, retro, mobilenet, deep_module_path):
        super(WrappedKerasLayer, self).__init__()
        self.layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=False)
        if deep_module_path:
            self.deep_module = AutoEncoder()
            self.deep_module(np.random.random(1280)[None, :])
            self.deep_module.load_weights(deep_module_path)
        else: self.deep_module = None
        if mobilenet:
            self.input_spec = (1, 224, 224, 3)
        else:
            self.input_spec = (1, 84, 84, 3) if retro == True else (1, 168, 168, 3)

    def __call__(self, _input):
        _input = np.reshape(np.array(_input), self.input_spec)
        _input = tf.convert_to_tensor(_input, dtype=tf.float32)
        tensor_var = tf.convert_to_tensor(np.array(self.layer(_input)))
        #tensor_var = tensorvar / tf.maximum(tensor_var)
        #if self.deep_module:
        #    tensor_var = self.deep_module(tensor_var)
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
        deep_module_path=None
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
        '''

        self._obstacle_tower_env = ObstacleTowerEnv(environment_filename, docker_training, worker_id, retro, timeout_wait, realtime_mode)
        self._obstacle_tower_env.floor(floor)
        self._flattener = ActionFlattener([3,3,2,3])
        self._action_space = self._flattener.action_space
        self.mobilenet = mobilenet
        self.gray_scale = gray_scale
        if mobilenet:
            self.image_module = WrappedKerasLayer(retro, self.mobilenet, deep_module_path)
            self.state_size = [1280]
        elif gray_scale:
            self.state_size = [84, 84, 1]
        else:
            self.state_size = [168, 168, 3]
        self._done = False
        self.id = worker_id

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def gray_process_observation(self, observation):
        observation = observation[0]
        observation = (observation * 255).astype(np.uint8)
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((84, 84), Image.NEAREST)
        grey_observation = np.mean(np.array(obs_image),axis=-1,keepdims=True)
        return grey_observation / 255

    def _preprocess_observation(self, observation):
        """
        Re-sizes visual observation to 84x84
        """
        observation = observation[0]
        observation = (observation * 255).astype(np.uint8)
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((224, 224), Image.NEAREST)
        return np.array(obs_image)

    def reset(self):
        observation = self._obstacle_tower_env.reset()
        self._done = False
        if self.mobilenet:
            #gray_observation = self.gray_process_observation(observation)
            image_observation = self._preprocess_observation(observation)
            observation = (observation[0] * 255).astype(np.uint8)
            return self.image_module(image_observation), observation[0]
        elif self.gray_scale:
            return self.grey_process_observation(observation), observation[0]
        else:
            return self._preprocess_observation(observation), observation

    def step(self, action):
        #if self._done:
        #    return self.reset()

        if action == 0: # forward
            action = [1, 0, 0, 0]
        elif action == 1: # rotate camera left
            action = [0, 1, 0, 0]
        elif action == 2: # rotate camera right
            action = [0, 2, 0, 0]
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

        if self.mobilenet: # OBSERVATION MUST BE RESIZED BEFORE PASSING TO image_module
            #gray_observation = self.gray_process_observation(observation)
            mobile_observation = self._preprocess_observation(observation)
            return self.image_module(mobile_observation), reward, done, info, observation[0]
        elif self.gray_scale:
            return self.gray_process_observation(observation), reward, done, info, observation[0]
        else:
            return self._preprocess_observation(observation), reward, done, info, observation

    def close(self):
        self._obstacle_tower_env.close()

    def floor(self, floor):
        self._obstacle_tower_env.floor(floor)
