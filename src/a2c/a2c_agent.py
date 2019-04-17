
import os
from collections import deque

import numpy as np
import tensorflow as tf

from src.a2c.parallel_env import ParallelEnv
from src.a2c.actor_critic_model import ActorCriticModel
from src.a2c.runner import Runner

class A2CAgent():
    '''
    A2C Agent class. Trains the model
    train_steps: Number of episodes to play and train on
    entropy_discount: Amount to discount entropy loss by relative to policy loss
    value_discount: Amount to discount value loss by relative to policy loss
    learning_rate: Learning_rate
    num_steps: Number of steps for each environment to take per rollout
    env_func: Function that builds one instance of an environment. Will be passed an idx as arg
    num_envs: Number of environments to run in parallel
    num_actions: Number of actions for model to output
    actor_fc: Actor model dense layers topology
    critic_fc: Critic model dense layers topology
    conv_size: Conv model topology
    '''
    def __init__(self,
                 train_steps=None,
                 entropy_discount=0.01,
                 value_discount=0.5,
                 learning_rate=0.00042,
                 num_steps=None,
                 env_func=None,
                 num_envs=None,
                 num_actions=None,
                 actor_fc=None,
                 critic_fc=None,
                 conv_size=None,
                 logging_period=25,
                 checkpoint_period=50,
                 output_dir="/tmp/a2c",
                 restore_dir=None):

        # Build environment
        env_func_list = [env_func for _ in range(num_envs)]
        self.env = ParallelEnv(env_func_list)

        # Build model
        self.model = ActorCriticModel(num_actions=num_actions,
                                      state_size=self.env.state_size,
                                      stack_size=self.env.stack_size,
                                      actor_fc=actor_fc,
                                      critic_fc=critic_fc,
                                      conv_size=conv_size)
        if restore_dir != None:
            self.model.load_weights(restore_dir)
        
        # Build runner
        self.runner = Runner(env=self.env,
                             model=self.model,
                             num_steps=num_steps)

        # Build optimizer
        self.opt = tf.keras.optimizers.Adam(learning_rate)

        # Setup training parameters
        self.train_steps = train_steps
        self.entropy_discount = entropy_discount
        self.value_discount = value_discount

        # Setup logging parameters
        self.floor_queue = deque(maxlen=100)
        self.reward_queue = deque(maxlen=100)
        self.logging_period = logging_period
        self.checkpoint_period = checkpoint_period
        self.episodes = 0
        
        # Build logging directories
        self.log_dir = os.path.join(output_dir, "logs/")
        os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints/")
        os.makedirs(os.path.dirname(self.checkpoint_dir), exist_ok=True)

        # Build summary writer
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def train(self):
        for i in range(self.train_steps):
            # Generate a batch from one rollout
            b_states, b_rewards, b_dones, b_actions, b_values, true_reward, ep_infos = self.runner.run()
            # Calculate advantages
            advs = b_rewards - b_values
            
            # Calculate loss
            with tf.GradientTape() as tape:
                # Get actions probabilities and values for all states
                processed_states = self.model.process_inputs(b_states)
                logits, values = self.model(processed_states)

                # Model returns un-softmaxed logits
                logits = tf.nn.softmax(logits)

                # Calculate our entropy loss
                cce = tf.keras.losses.CategoricalCrossentropy()
                entropy_loss = cce(tf.stop_gradient(logits), logits) * -1
                # Calculate our policy loss
                wce = tf.keras.losses.SparseCategoricalCrossentropy()
                policy_loss = wce(b_actions, logits, sample_weight=advs)
                # Calculate our value loss
                mse = tf.keras.losses.MeanSquaredError()
                value_loss = mse(b_rewards[:, None], values)

                total_loss = policy_loss + \
                             (entropy_loss * self.entropy_discount) + \
                             (value_loss * self.value_discount)

            # Calculate and apply gradient
            total_grads = tape.gradient(total_loss, self.model.trainable_weights)
            self.opt.apply_gradients(zip(total_grads, self.model.trainable_weights))

            # Log data
            self.logging(b_rewards, b_values, ep_infos, entropy_loss, policy_loss, value_loss, i)

            print("\n")

    def logging(self, rewards, values, ep_infos, entropy_loss, policy_loss, value_loss, i):
        # Pull specific info from info array and store in queue
        for info in ep_infos:
            self.floor_queue.append(info["floor"])
            self.reward_queue.append(info["total_reward"])
            self.episodes += 1
        
        avg_floor = 0 if len(self.floor_queue) == 0 else sum(self.floor_queue) / len(self.floor_queue)
        avg_reward = 0 if len(self.reward_queue) == 0 else sum(self.reward_queue) / len(self.reward_queue)
        explained_variance = self.explained_variance(values, rewards)

        print("| Iteration: {} |".format(i)) 
        print("| Episodes: {} | Average Floor: {} | Average Reward: {} |".format(self.episodes, avg_floor, avg_reward))
        print("| Entropy Loss: {} | Policy Loss: {} | Value Loss: {} |".format(entropy_loss, policy_loss, value_loss))
        print("| Explained Variance: {} | Environment Variance: {} |".format(explained_variance, np.var(rewards)))

        # Periodically log
        if i % self.logging_period == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar("Average Floor", avg_floor, i)
                tf.summary.scalar("Average Reward", avg_reward, i)
                tf.summary.scalar("Entropy Loss", entropy_loss, i)
                tf.summary.scalar("Policy Loss", policy_loss, i)
                tf.summary.scalar("Value Loss", value_loss, i)
                tf.summary.scalar("Explained Variance", explained_variance, i)
        # Periodically save checkoints
        if i % self.checkpoint_period == 0:
            model_save_path = os.path.join(self.checkpoint_dir, "model_{}.h5".format(i))
            self.model.save_weights(model_save_path)
            print("Model saved to {}".format(model_save_path))

    def explained_variance(self, y_pred, y_true):
        """
        Computes fraction of variance that ypred explains about y.
        Returns 1 - Var[y-ypred] / Var[y]
        interpretation:
            ev=0  =>  might as well have predicted zero
            ev=1  =>  perfect prediction
            ev<0  =>  worse than just predicting zero
        """
        assert y_true.ndim == 1 and y_pred.ndim == 1
        var_y = np.var(y_true)
        return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y





if __name__ == '__main__':
    from src.a2c.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv
    env_filename = "../ObstacleTower/obstacletower"
    def env_func(idx):
        return WrappedObstacleTowerEnv(env_filename,
                                       worker_id=idx,
                                       gray_scale=True,
                                       realtime_mode=True)

    print("Building agent...")
    agent = A2CAgent(train_steps=1,
                     entropy_discount=0.01,
                     value_discount=0.5,
                     learning_rate=0.00000042,
                     num_steps=5,
                     env_func=env_func,
                     num_envs=4,
                     num_actions=4,
                     actor_fc=[1024,512],
                     critic_fc=[1024,512],
                     conv_size=((8,4,32), (4,2,64), (3,1,64)),
                     output_dir="./agent_test")
    print("Agent built!")

    print("Starting train...")
    agent.train()
    print("Train done!")
    
