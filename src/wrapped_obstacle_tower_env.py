import gin
import numpy as np

from obstacle_tower_env import ObstacleTowerEnv
from obstacle_tower_env import ActionFlattener

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

from PIL import Image

class WrappedKerasLayer(tf.keras.layers.Layer):
    def __init__(self, retro, mobilenet):
        super(WrappedKerasLayer, self).__init__()
        self.layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280],
        trainable=False)
        if mobilenet:
            self.input_spec = (1, 224, 224, 3)
        else:
            self.input_spec = (1, 84, 84, 3) if retro == True else (1, 168, 168, 3)

    def __call__(self, _input):
        _input = np.reshape(np.array(_input), self.input_spec)
        _input = tf.convert_to_tensor(_input, dtype=tf.float32)
        tensor_var = tf.convert_to_tensor(np.array(self.layer(_input)))
        # print("tensor_var: {}".format(tensor_var.shape))
        tensor_var = tf.squeeze(tensor_var)
        return tensor_var

INPUT_SHAPE = (168, 168, 3)

def build_autoencoder(model_path):
    _input = Input(shape=INPUT_SHAPE)

    # conv_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(_input)
    # max_1 = MaxPooling2D((2, 2), padding='same')(conv_1)
    # conv_2 = Conv2D(8, (2, 2), activation='relu', padding='same')(max_1)
    # encoded = MaxPooling2D((2, 2), padding='same')(conv_2)

    # upconv_2 = Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same')(encoded)
    # concat_2 = concatenate([upconv_2, conv_2])
    # conv_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(concat_2)

    # upconv_1 = Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same')(conv_3)
    # concat_1 = concatenate([upconv_1, conv_1])
    # conv_4 = Conv2D(16, (3, 3), activation='relu', padding='same')(concat_1)

    # _output = Conv2D(1, (1, 1), activation='relu', padding='same')(conv_4)

    conv_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(_input)
    max_1 = MaxPooling2D((2, 2), padding='same')(conv_1)
    conv_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(max_1)
    max_2 = MaxPooling2D((2, 2), padding='same')(conv_2)
    conv_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(max_2)
    encoded = MaxPooling2D((2, 2), padding='same')(conv_3)

    upconv_3 = Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same')(encoded)
    concat_3 = concatenate([upconv_3, conv_3])
    conv_4 = Conv2D(8, (3, 3), activation='relu', padding='same')(concat_3)

    upconv_2 = Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same')(conv_4)
    concat_2 = concatenate([upconv_2, conv_2])
    conv_5 = Conv2D(8, (3, 3), activation='relu', padding='same')(concat_2)

    upconv_1 = Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same')(conv_5)
    concat_1 = concatenate([upconv_1, conv_1])
    conv_6 = Conv2D(16, (3, 3), activation='relu', padding='same')(concat_1)

    _output = Conv2D(3, (1, 1), activation='relu', padding='same')(conv_6)

    autoencoder = Model(_input, _output)
    autoencoder.load_weights(model_path)
    return autoencoder

@gin.configurable
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
        autoencoder=None,
        floor=0
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
        if floor != 0:
            self._obstacle_tower_env.floor(floor)
        self._flattener = ActionFlattener([3,3,2,3])
        self._action_space = self._flattener.action_space
        self.mobilenet = mobilenet
        self.gray_scale = gray_scale
        if mobilenet:
            self.image_module = WrappedKerasLayer(retro, self.mobilenet)
        self._done = False
        if autoencoder:
            print("Loading autoencoder from {}".format(autoencoder))
            self.autoencoder = build_autoencoder(autoencoder)
            print("Done.")
        else:
            self.autoencoder = None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def gray_process_observation(self, observation):
        observation = (observation * 255).astype(np.uint8)
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((84, 84), Image.NEAREST)
        gray_observation = np.mean(np.array(obs_image),axis=-1,keepdims=True)
        gray_observation = (gray_observation / 255)

        # gray_observation = self.autoencoder.predict(gray_observation)
        return gray_observation

    def _preprocess_observation(self, observation):
        """
        Re-sizes visual observation to 84x84
        """
        observation = (observation * 255).astype(np.uint8)
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((224, 224), Image.NEAREST)
        return np.array(obs_image)

    def reset(self):
        observation = self._obstacle_tower_env.reset()
        observation = observation[0]
        self._done = False
        if self.mobilenet:
            if self.autoencoder:
                observation = self.autoencoder.predict(observation[None,:])[0]
            return self.image_module(self._preprocess_observation(observation)), observation
        elif self.gray_scale:
            gray_observation = self.gray_process_observation(observation)
            if self.autoencoder:
                gray_observation = self.autoencoder.predict(gray_observation[None,:])[0]
            return gray_observation, observation
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
        # elif action == 5:
        #     action = [2, 0, 0, 0]
        # elif action == 6:
        #     action = [0, 0, 0, 1]
        # elif action == 7:
        #     action = [0, 0, 0, 2]


        observation, reward, done, info = self._obstacle_tower_env.step(action)
        observation = observation[0]
        self._done = done

        if self.mobilenet:
            if self.autoencoder:
                observation = self.autoencoder.predict(observation[None,:])[0]
            return (self.image_module(self._preprocess_observation(observation)), reward, done, info), observation
        elif self.gray_scale:
            gray_observation = self.gray_process_observation(observation)
            if self.autoencoder:
                gray_observation = self.autoencoder.predict(gray_observation[None,:])[0]
            return (gray_observation, reward, done, info), observation
        else:
            return (self._preprocess_observation(observation), reward, done, info), observation

    def close(self):
        self._obstacle_tower_env.close()

    def floor(self, floor):
        self._obstacle_tower_env.floor(floor)
