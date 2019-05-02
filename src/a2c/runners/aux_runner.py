
import numpy as np
from tqdm import tqdm
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub

from src.a2c.runners.runner import Runner

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

class AuxRunner(Runner):
    def __init__(self,
                 env=None,
                 model=None,
                 num_steps=None,
                 gamma=0.99,
                 num_aux=9,
                 aux_dir=None):
        """
        A runner to learn the policy of an environment for a model
        env: The environment to learn from
        model: The model to learn
        num_steps: The number of steps to run for each environment
        gamma: Discount factor
        """
        super().__init__(env=env,
                         model=model,
                         num_steps=num_steps,
                         gamma=gamma)

        # Build ground-truth aux output model
        self.aux = AuxModel(num_aux)
        if aux_dir != None:
            self.aux.load_weights(aux_dir)
        else:
            raise ValueError('Weights for the aux ground truth model must be specified')

    def generate_batch(self):
        """
        Generates a batch
        returns:
            - observations
            - rewards
            - actions
            - values
            - infos
        """

        b_states, b_rewards, b_dones, b_actions, b_values, b_probs, b_aux, ep_infos = self.rollout()
        b_dones.append(self.dones)

        # Convert to numpy array and change shape from (num_steps, n_envs) to (n_envs, num_steps)
        b_states = np.asarray(b_states, dtype=self.states.dtype).swapaxes(0, 1)
        b_rewards = np.asarray(b_rewards, dtype=np.float32).swapaxes(0, 1)
        b_actions = np.asarray(b_actions).swapaxes(0, 1)
        b_values = np.asarray(b_values, dtype=np.float32).swapaxes(0, 1)
        b_probs = np.asarray(b_probs, dtype=np.float32).swapaxes(0, 1)
        b_dones = np.asarray(b_dones, dtype=np.bool).swapaxes(0, 1)
        b_dones = b_dones[:, 1:]
        b_aux = np.asarray(b_aux).swapaxes(0, 1)
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
        b_states, b_rewards, b_dones, b_actions, b_values, b_probs, b_aux, true_rewards = \
            map(self.flatten, (b_states, b_rewards, b_dones, b_actions,
                               b_values, b_probs, b_aux, true_rewards))
        return b_states, b_rewards, b_dones, b_actions, b_values, b_probs, b_aux, true_rewards, ep_infos

    def rollout(self):
        # Init lists
        b_states, b_rewards, b_actions, b_values, b_dones, b_probs, b_aux = [], [], [], [], [], [], []
        ep_infos = []

        # Rollout on each env for num_steps
        for _ in tqdm(range(self.num_steps), "Rollout"):
            # Generate actions, values, and probabilities of the actions sampled
            actions, values, probs = self.model.step(self.states)

            b_states.append(self.states)
            b_actions.append(actions)
            b_values.append(values)
            b_probs.append(probs)
            b_dones.append(self.dones)

            # Get auxillary outputs for each env
            aux_obs = []
            reset_indicies = []
            for i,info in enumerate(self.infos):
                if info["reset"]:
                    aux_obs.append(np.zeros((224,224,3)))
                    reset_indicies.append(i)
                else:
                    aux_obs.append(self.scale_up_obs(info["brain_info"].visual_observations[0][0]))
            aux_obs = np.array(aux_obs)
            aux = np.squeeze(self.aux.predict(aux_obs))
            if len(reset_indicies) > 0:
                aux[reset_indicies] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
            b_aux.append(aux)

            # Take actions
            self.states, rewards, self.dones, self.infos = self.env.step(actions)

            b_rewards.append(rewards)

            # Check if any episode finished, and save the metrics if one did
            for info in self.infos:
                ep_info = info.get("episode_info")
                if ep_info is not None:
                    ep_infos.append(ep_info)

        return b_states, b_rewards, b_dones, b_actions, b_values, b_probs, b_aux, ep_infos

    def scale_up_obs(self, observation):
        """
        Re-sizes obs to 224x224 for mobilenet
        """
        observation = (observation * 255).astype(np.uint8)
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((224, 224), Image.NEAREST)
        return np.array(obs_image).astype(np.float32) / 255.




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
