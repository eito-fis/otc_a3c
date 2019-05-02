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

    def reset(self):
        ret_state, info = super().reset()

        info["reset"] = True

        return ret_state, info

    def step(self, action):
        ret_state, reward, done, info = super().step(action)

        if "episode_info" in info:
            info["reset"] = True
        else:
            info["reset"] = False

        return ret_state, reward, done, info
