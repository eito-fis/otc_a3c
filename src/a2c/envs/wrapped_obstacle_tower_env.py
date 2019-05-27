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

        self._obstacle_tower_env = ObstacleTowerEnv(environment_filename,
                                                    docker_training,
                                                    worker_id,
                                                    retro,
                                                    timeout_wait,
                                                    realtime_mode)
        if floor is not 0:
            self._obstacle_tower_env.floor(floor)
        self.start_floor = floor
        self.current_floor = floor

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
        self.current_reward = 0
        self.max_floor = 25

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
        return self.mobilenet(np.array(obs_image))

    def reset(self):
        # Reset env, stack and floor
        # (We save state as an attribute so child objects can access it)
        config = {"total-floors": 11}
        self.state = self._obstacle_tower_env.reset(config)
        self.current_floor = self.start_floor
        self.stack = [np.random.random(self.state_size).astype(np.float32) for _ in range(self.stack_size)]
        self.total_reward = 0
        self.current_reward = 0

        # Preprocess current obs and add to stack
        if self.retro:
            observation = (self.state / 255).astype(np.float32)
        else:
            observation, key, time = self.state

        if self.mobilenet:
            observation = self.mobile_preprocess_observation(observation)
        elif self.gray_scale:
            observation = self.gray_preprocess_observation(observation)

        self.stack = self.stack[1:] + [observation]

        # Build our state (MUST BE A TUPLE)
        one_hot_floor = tf.one_hot(self.current_floor, self.max_floor).numpy()
        floor_data = np.append(one_hot_floor, self.current_reward).astype(np.float32)
        stacked_state = np.concatenate(self.stack, axis=-1).astype(np.float32)
        if self.retro is True:
            ret_state = (stacked_state, floor_data)
        else:
            # Clip time to 2000, then normalize
            time = (2000. if time > 2000 else time) / 2000.
            key_time_data = np.array([key, time]).astype(np.float32)
            #key_time_data = np.array([key]).astype(np.float32)
            ret_state = (stacked_state, floor_data, key_time_data)

        # Empty info dict for any children to add to
        info = {}

        return ret_state, info

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
        # (We save state as an attribute so child objects can access it)
        self.state, reward, done, info = self._obstacle_tower_env.step(action)

        # Keep track of current floor reward and total reward
        if reward >= 0.95:
            self.current_floor += 1
            self.current_reward = 0
            done = True
        else:
            self.current_reward += reward
        self.total_reward += reward
        
        if done and reward < 0.95:
            # Save info and reset when an episode ends
            info["episode_info"] = {"floor": self.current_floor, "total_reward": self.total_reward}
            ret_state, _ = self.reset()
        else:
            # Preprocess current obs and add to stack
            if self.retro:
                observation = (self.state / 255).astype(np.float32)
            else:
                observation, key, time = self.state

            if self.mobilenet:
                observation = self.mobile_preprocess_observation(observation)
            elif self.gray_scale:
                observation = self.gray_preprocess_observation(observation)

            self.stack = self.stack[1:] + [observation]

            # Build our state (MUST BE A TUPLE)
            one_hot_floor = tf.one_hot(self.current_floor, self.max_floor).numpy()
            floor_data = np.append(one_hot_floor, self.current_reward).astype(np.float32)
            stacked_state = np.concatenate(self.stack, axis=-1).astype(np.float32)
            if self.retro is True:
                ret_state = (stacked_state, floor_data)
            else:
                # Clip time to 2000, then normalize
                time = (2000. if time > 2000 else time) / 2000.
                key_time_data = np.array([key, time]).astype(np.float32)
                #key_time_data = np.array([key]).astype(np.float32)
                ret_state = (stacked_state, floor_data, key_time_data)

        return ret_state, reward, done, info

    def close(self):
        self._obstacle_tower_env.close()

    def floor(self, floor):
        self._obstacle_tower_env.floor(floor)
        self.start_floor = floor
