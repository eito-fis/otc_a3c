import numpy as np

from obstacle_tower_env import ObstacleTowerEnv
from obstacle_tower_env import ActionFlattener

import tensorflow as tf
import tensorflow_hub as hub

from src.a2c.envs.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv

from PIL import Image

class WrappedKerasLayer(tf.keras.layers.Layer):
    def __init__(self, retro, mobilenet):
        super(WrappedKerasLayer, self).__init__()
        self.layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2",
                                    output_shape=[1280],
                                    trainable=False)
        if mobilenet:
            self.input_spec = (1, 224, 224, 3)
        else:
            self.input_spec = (1, 84, 84, 3) if retro == True else (1, 168, 168, 3)

    def __call__(self, _input):
        _input = np.reshape(np.array(_input), self.input_spec)
        _input = tf.convert_to_tensor(_input, dtype=tf.float32)
        tensor_var = tf.convert_to_tensor(np.array(self.layer(_input)))
        tensor_var = tf.squeeze(tensor_var)
        return tensor_var

class AuxModel(tf.keras.Model):
    def __init__(self, num_aux):
        super().__init__()
        self.conv = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2",
                                    output_shape=[1280],
                                    trainable=False)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.result = tf.keras.layers.Dense(num_aux, activation='softmax')

        self.call(np.zeros((1, 224, 224, 3)).astype(np.float32))

    def call(self, data):
        data = self.conv(data)
        data = self.dense1(data)
        data = self.result(data)
        return data


class AuxEnv(WrappedObstacleTowerEnv):

    def __init__(
        self,
        environment_filename=None,
        docker_training=False,
        worker_id=0,
        retro=False,
        timeout_wait=30,
        realtime_mode=False,
        num_actions=3,
        stack_size=4,
        mobilenet=False,
        gray_scale=False,
        floor=0,
        num_aux=9,
        aux_dir=None
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
        print("Init called!")
        super().__init__(environment_filename=environment_filename,
                         docker_training=docker_training,
                         worker_id=worker_id,
                         retro=retro,
                         timeout_wait=timeout_wait,
                         realtime_mode=realtime_mode,
                         num_actions=num_actions,
                         stack_size=stack_size,
                         mobilenet=mobilenet,
                         gray_scale=gray_scale,
                         floor=floor)

        # Build ground-truth aux output model
        self.aux = AuxModel(num_aux)
        if aux_dir != None:
            self.aux.load_weights(aux_dir)
        else:
            raise ValueError('Weights for the aux ground truth model must be specified')

    def scale_up_observation(self, observation):
        """
        Re-sizes obs to 224x224 for mobilenet
        """
        observation = (observation * 255).astype(np.uint8)
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((224, 224), Image.NEAREST)
        return np.array(obs_image).astype(np.float32) / 255.

    def reset(self):
        ret_state, info = super().reset()

        if self.retro:
            observation = (self.state / 255).astype(np.float32)
        else:
            observation, _, _ = self.state
        scaled_obs = self.scale_up_observation(observation)
        aux = np.squeeze(self.aux(scaled_obs[None, :]).numpy())

        info["aux"] = aux

        return ret_state, info

    def step(self, action):
        ret_state, reward, done, info = super().step(action)

        if self.retro:
            observation = (self.state / 255).astype(np.float32)
        else:
            observation, _, _ = self.state
        scaled_obs = self.scale_up_observation(observation)
        aux = np.squeeze(self.aux(scaled_obs[None, :]).numpy())

        info["aux"] = aux

        return ret_state, reward, done, info

    def close(self):
        self._obstacle_tower_env.close()

    def floor(self, floor):
        self._obstacle_tower_env.floor(floor)
        self.start_floor = floor
