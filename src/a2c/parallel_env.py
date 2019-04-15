
import numpy as np
import multiprocessing
from multiprocessing import Process, Pipe
import os

def worker(remote, env_fn_wrapper, idx):
    '''
    The function that each of our processes runs. Takes commands from the
    main thread through the pipe and takes asynchronously executes the command,
    then returns the results through the pipe.
    '''
    # Build environment at start
    env = env_fn_wrapper.x(idx)
    # Loop until environment is closed
    while True:
        # Recieve the command and any associated data
        cmd, data = remote.recv()
        if cmd == 'step':
            state, reward, done, info = env.step(data)
            total_info = info
            remote.send((state, reward, done, total_info))
        elif cmd == 'reset':
            state = env.reset()
            remote.send(state)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_size':
            remote.send((env.state_size, env.stack_size))
        else:
            raise NotImplementedError

class CloudpickleWrapper(object):
    '''
    Wrapper for transfering data between each sub-process and
    the main process.
    '''
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class ParallelEnv():
    '''
    Multi-environment wrapper. Takes a list of env build functions and maintains
    them in individual processes.
    '''
    def __init__(self, env_fns):
        nenvs = len(env_fns)
        
        # Multi-processing is by default forked, but Tensorflow isn't fork safe
        multiprocessing.set_start_method("spawn")

        # Build pipes for communication to and from processed
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        # Make sure our sub-processes don't use GPU 
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        # Build processes
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn), idx))
                   for idx, (work_remote, env_fn) in enumerate(zip(self.work_remotes, env_fns))]

        # Start processes
        for p in self.ps:
            p.daemon = True
            p.start()

        # Re-enable gpu for our main process
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # Get env state size
        self.remotes[0].send(('get_size', None))
        self.state_size, self.stack_size = self.remotes[0].recv()

    def step(self, actions):
        # Send each environment it's action...
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        # ..then wait for the response from each one
        results = [remote.recv() for remote in self.remotes]
        
        # Split responses up into individual lists
        _states, rewards, dones, infos = zip(*results)

        # Convert list of tuples into array of tuples
        states = np.empty(len(_states), dtype=object)
        states[:] = _states

        return states, np.stack(rewards), np.stack(dones), infos

    def reset(self):
        # Send each env a reset command...
        for remote in self.remotes:
            remote.send(('reset', None))

        # ...then wait for the response from each one
        _states = [remote.recv() for remote in self.remotes]

        # Convert list of tuples into array of tuples
        states = np.empty(len(_states), dtype=object)
        states[:] = _states
        return states

    def close(self):
        # Close each env, then end each process
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)





if __name__ == '__main__':
    import tensorflow as tf
    from src.a2c.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv

    env_filename = "../ObstacleTower/obstacletower"
    def env_func(idx):
        return WrappedObstacleTowerEnv(env_filename,
                                       worker_id=idx,
                                       realtime_mode=True)
    func_list = [env_func for _ in range(4)]
    parallel_env = ParallelEnv(func_list)

    states = parallel_env.reset()
    print(states)
    print(states.shape)
    input()

    while True:
        states, rewards, dones, infos = parallel_env.step([1 for _ in range(4)])
        print(states)
        print("State shape: {}".format(states.shape))
        print("Rewards shape: {}".format(rewards.shape))
        print("Dones shape: {}".format(dones.shape))
        input()

