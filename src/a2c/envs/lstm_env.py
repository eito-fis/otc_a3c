import numpy as np

from src.a2c.envs.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv

class LSTMEnv(WrappedObstacleTowerEnv):

    def __init__(
        self,
        environment_filename=None,
        docker_training=False,
        worker_id=0,
        retro=False,
        timeout_wait=30000,
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

        #ret_state = ret_state[0]

        return ret_state, info

    def step(self, action):
        ret_state, reward, done, info = super().step(action)

        #if not "episode_info" in info:
        #    ret_state = ret_state[0]
        done = np.float32(done)

        return ret_state, reward, done, info
