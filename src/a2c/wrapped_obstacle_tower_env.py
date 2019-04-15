import numpy as np

from obstacle_tower_env import ObstacleTowerEnv
from obstacle_tower_env import ActionFlattener

import tensorflow as tf
import tensorflow_hub as hub

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

class WrappedObstacleTowerEnv():

    def __init__(
        self,
        environment_filename=None,
        docker_training=False,
        worker_id=0,
        retro=True,
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

        self._obstacle_tower_env = ObstacleTowerEnv(environment_filename,
                                                    docker_training,
                                                    worker_id,
                                                    retro,
                                                    timeout_wait,
                                                    realtime_mode)
        if floor is not 0:
            self._obstacle_tower_env.floor(floor)
        self.start_floor = floor
        self.floor = floor

        self.mobilenet = mobilenet
        self.gray_scale = gray_scale
        self.retro = retro
        if mobilenet:
            self.mobilenet = WrappedKerasLayer(retro, self.mobilenet)
            self.state_size = [1280]
        elif gray_scale:
            self.state_size = [84, 84, 1]
        elif retro:
            self.state_size = [84, 84, 3]
        else:
            self.state_size = [168, 168, 3]

        self.stack_size = stack_size
        self.stack = [np.random.random(self.state_size).astype(np.float32) for _ in range(self.stack_size)]
        self.total_reward = 0

        self.id = worker_id

    def gray_preprocess_observation(self, observation):
        '''
        Re-sizes obs to 84x84 and compresses to grayscale
        '''
        observation = (observation * 255).astype(np.uint8)
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((84, 84), Image.NEAREST)
        gray_observation = np.mean(np.array(obs_image),axis=-1,keepdims=True)
        return gray_observation / 255

    def mobile_preprocess_observation(self, observation):
        """
        Re-sizes obs to 224x224 for mobilenet
        """
        observation = (observation * 255).astype(np.uint8)
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((224, 224), Image.NEAREST)
        return np.array(obs_image)

    def reset(self):
        # Reset env, stack and floor
        state = self._obstacle_tower_env.reset()
        self.floor = self.start_floor
        self.stack = [np.random.random(self.state_size).astype(np.float32) for _ in range(self.stack_size)]
        self.total_reward = 0

        # Preprocess current obs and add to stack
        if self.retro is not True:
            observation = state[0]
        else:
            observation = (state / 255).astype(np.float32)
        if self.mobilenet:
            observation = self.mobile_preprocess_observation(observation)
        elif self.gray_scale:
            observation = self.gray_preprocess_observation(observation)
        self.stack = self.stack[1:] + [observation]

        # Convert floor into an array so it can be fed into the network
        _floor_arry = np.array([self.floor]).astype(np.float32)
        return (np.concatenate(self.stack, axis=-1).astype(np.float32), _floor_arry)

    def step(self, action):
        # Convert int action to vector required by the env
        if self.retro:
            if action == 0: # forward
                action = 18
            elif action == 1: # rotate camera left
                action = 24
            elif action == 2: # rotate camera right
                action = 30
            elif action == 3: # jump forward
                action = 21
        else:
            if action == 0: # forward
                action = [1, 0, 0, 0]
            elif action == 1: # rotate camera left
                action = [1, 1, 0, 0]
            elif action == 2: # rotate camera right
                action = [1, 2, 0, 0]
            elif action == 3: # jump forward
                action = [1, 0, 1, 0]


        # Take the step and record data
        state, reward, done, info = self._obstacle_tower_env.step(action)

        if reward >= 1: self.floor += 1
        self.total_reward += reward
        
        if done:
            # Save info and reset when an episode ends
            info["episode_info"] = {"floor": self.floor, "total_reward": self.total_reward}
            state = self.reset()
        else:
            # Preprocess current obs and add to stack
            if self.retro is not True:
                observation = state[0]
            else:
                observation = (state / 255).astype(np.float32)
            if self.mobilenet:
                observation = self.mobile_preprocess_observation(observation)
            elif self.gray_scale:
                observation = self.gray_preprocess_observation(observation)
            self.stack = self.stack[1:] + [observation]

            # Convert floor into an array so it can be fed into the network
            _floor_arry = np.array([self.floor]).astype(np.float32)
            # Build our state
            state = (np.concatenate(self.stack, axis=-1).astype(np.float32), _floor_arry)

        return state, reward, done, info

    def close(self):
        self._obstacle_tower_env.close()

    def floor(self, floor):
        self._obstacle_tower_env.floor(floor)
        self.start_floor = floor
