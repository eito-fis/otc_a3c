
import numpy as np
from multiprocessing import Process, Pipe

# This class is to run multiple environments at the same time.

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
            state, reward, done, info, ob = env.step(data)
            total_info = info.copy()
            if done:
                state, ob = env.reset()
            remote.send((state, reward, done, total_info, ob))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_state_size':
            remote.send(env.state_size)
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
        
        # Build pipes for communication to and from processed
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        # Build processes
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn), idx))
                   for idx, (work_remote, env_fn) in enumerate(zip(self.work_remotes, env_fns))]
        # Start processes
        for p in self.ps:
            p.start()

        # Get env state size
        self.remotes[0].send(('get_state_size', None))
        self.state_size = self.remotes[0].recv()

    def step(self, actions):
        # Send each environment it's action...
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        # ..then wait for the response from each one
        results = [remote.recv() for remote in self.remotes]
        
        # Split responses up into individual lists and return
        states, rewards, dones, infos, obs = zip(*results)
        return np.stack(states), np.stack(rewards), np.stack(dones), infos, np.stack(obs)

    def reset(self):
        # Send each env a reset command...
        for remote in self.remotes:
            remote.send(('reset', None))

        # ...then wait for the response from each one
        results = [remote.recv() for remote in self.remotes]

        # Split responses up into individual lists and return
        states, obs = zip(*results)
        return np.stack(states), np.stack(obs)

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
                                       mobilenet=True,
                                       realtime_mode=True)

    func_list = [env_func for _ in range(16)]
    parallel_env = ParallelEnv(func_list)
    while True:
        parallel_env.step([1 for _ in range(16)])


