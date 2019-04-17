
import numpy as np
from tqdm import tqdm

class Runner():
    def __init__(self,
                 env=None,
                 model=None,
                 num_steps=None,
                 gamma=0.99):
        """
        A runner to learn the policy of an environment for a model
        env: The environment to learn from
        model: The model to learn
        num_steps: The number of steps to run for each environment
        gamma: Discount factor
        """
        self.env = env
        self.model = model
        self.num_steps = num_steps
        self.gamma = gamma
        self.states  = self.env.reset()
        self.dones = np.zeros(self.env.num_envs)

    def run(self):
        """
        Run a learning step of the model
        returns:
            - observations
            - rewards
            - actions
            - values
            - infos
        """
        # Init lists
        b_states, b_rewards, b_actions, b_values, b_dones, b_probs = [], [], [], [], [], []
        ep_infos = []

        # Rollout on each env for num_steps
        for _ in tqdm(range(self.num_steps), "Rollout"):
            # Generate actions, values, and probabilities of the actions sampled
            actions, values = self.model.step(self.states)

            b_states.append(self.states)
            b_actions.append(actions)
            b_values.append(values)
            b_dones.append(self.dones)

            # Take actions
            self.states[:], rewards, self.dones[:], infos = self.env.step(actions)

            b_rewards.append(rewards)

            # Check if any episode finished, and save the metrics if one did
            for info in infos:
                ep_info = info.get("episode_info")
                if ep_info is not None:
                    ep_infos.append(ep_info)
        b_dones.append(self.dones)

        # Convert to numpy array and change shape from (num_steps, n_envs) to (n_envs, num_steps)
        b_states = np.asarray(b_states, dtype=self.states.dtype).swapaxes(0, 1)
        b_rewards = np.asarray(b_rewards, dtype=np.float32).swapaxes(0, 1)
        b_actions = np.asarray(b_actions).swapaxes(0, 1)
        b_values = np.asarray(b_values, dtype=np.float32).swapaxes(0, 1)
        b_dones = np.asarray(b_dones, dtype=np.bool).swapaxes(0, 1)
        b_dones = b_dones[:, 1:]
        true_rewards = np.copy(b_rewards)
        last_values = self.model.get_values(self.states).tolist()

        # Calculate future discounted reward
        for n, (rewards, dones, value) in enumerate(zip(b_rewards, b_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = self.discount(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = self.discount(rewards, dones, self.gamma)
            b_rewards[n] = rewards

        # Flatten (num_env, num_steps) to (num_envs * num_steps) so we have one big batch
        b_states, b_rewards, b_dones, b_actions, b_values, true_rewards = \
            map(self.flatten, (b_states, b_rewards, b_dones, b_actions, b_values, true_rewards))
        return b_states, b_rewards, b_dones, b_actions, b_values, true_rewards, ep_infos

    def discount(self, rewards, dones, gamma):
        """
        Apply the discount value to the reward, where the environment is not done
        """
        discounted = []
        ret = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            if reward == 0 or reward <= 0.01:
                ret = reward + gamma * ret * (1. - done)
            else:
                ret = reward
            discounted.append(ret)
        return discounted[::-1]

    def flatten(self, arr):
        """
        Flatten the first two axis
        """
        shape = arr.shape
        return arr.reshape(shape[0] * shape[1], *shape[2:])





if __name__ == '__main__':
    from src.a2c.actor_critic_model import ActorCriticModel
    from src.a2c.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv
    from src.a2c.parallel_env import ParallelEnv

    print("Building env...")
    env_filename = "../ObstacleTower/obstacletower"
    def env_func(idx):
        return WrappedObstacleTowerEnv(env_filename,
                                       worker_id=idx,
                                       gray_scale=True,
                                       realtime_mode=True)
    func_list = [env_func for _ in range(4)]
    parallel_env = ParallelEnv(func_list)
    print("Env built!")

    print("Building model...")
    model = ActorCriticModel(num_actions=4,
                             state_size=parallel_env.state_size,
                             stack_size=4,
                             actor_fc=[1024,512],
                             critic_fc=[1024,512],
                             conv_size=((8,4,32), (4,2,64), (3,1,64)))
    print("Model built!")

    print("Building runner...")
    runner = Runner(env=parallel_env, model=model, num_steps=650)
    print("Runner made!")

    print("Running runner!")
    states, rewards, dones, actions, values, true_reward, ep_infos = runner.run()
    print("States: {}, shape: {}".format(states, states.shape))
    print("Rewards: {}, shape: {}".format(rewards, rewards.shape))
    print("Dones: {}, shape: {}".format(dones, dones.shape))
    print("Actions: {}, shape: {}".format(actions, actions.shape))
    print("Values: {}, shape: {}".format(values, values.shape))
    print("True Reward: {}, shape: {}".format(true_reward, true_reward.shape))
    print("Episode Infos: {}".format(ep_infos))
